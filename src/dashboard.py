"""Streamlit evaluation dashboard for the Air-Threat Cascade engine.

This page is the primary demonstration surface for judges. It lets a reviewer:

* pick an observation instant ``T`` (the temporal cutoff) and watch every
  metric, graph and prediction recompute from *only* the alarms that started
  strictly before ``T`` (see :mod:`src.loader` for the leakage guarantees);
* filter by kinematic wave class (fast-wave ballistic / aviation versus
  slow-wave UAV / Shahed) and individual threat types;
* switch between a real CSV export and the synthetic fallback generator, with
  an automatic fallback when the CSV is missing so the app always runs;
* inspect the regional propagation graph (``networkx`` + ``matplotlib``) and
  run a cascade simulation that ranks the top-K regions most likely to be
  alarmed next, with arrival-time estimates.

Run it with::

    python run_dashboard.py            # convenience runner (project root)
    streamlit run src/dashboard.py     # equivalent

Module layout
-------------
1. Constants and small value objects.
2. Cached data access (thin wrappers around :mod:`src.loader`).
3. Pure analytics: active alarms, cascade velocity, propagation graph,
   top-K cascade prediction. These never touch ``st`` and are unit-tested
   directly in ``tests/test_dashboard.py``.
4. Rendering helpers: sidebar, status banner, the three tabs.
5. ``main()`` composing the page; executed when Streamlit runs the script.

All timestamps are handled and displayed in UTC.
"""

from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Final, Mapping, Optional, Sequence

# Make ``src.loader`` importable when Streamlit runs this file directly
# (``streamlit run src/dashboard.py`` puts ``src/`` on sys.path, not the root).
_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")  # Headless backend; must precede pyplot import.

import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from src.loader import (  # noqa: E402
    THREAT_TYPES,
    UKRAINIAN_REGIONS,
    AirThreatLoader,
    SchemaValidationError,
    TemporalLeakageError,
    ThreatWave,
    generate_synthetic_alarms,
    load_and_filter_alarms,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# 1. Constants
# --------------------------------------------------------------------------- #
UTC: Final = timezone.utc
DEFAULT_CSV_PATH: Final[str] = "data/alarms.csv"
# An alarm with no recorded end is treated as active only if it started within
# this many hours before T; older open-ended rows are feed gaps. (Kept as a
# comment rather than a bare docstring: Streamlit "magic" would render it.)
ACTIVE_WINDOW_HOURS: Final[float] = 8.0

SOURCE_CSV: Final[str] = "csv"
SOURCE_SYNTHETIC: Final[str] = "synthetic"
SOURCE_LABELS: Final[Mapping[str, str]] = {
    SOURCE_CSV: "Real CSV",
    SOURCE_SYNTHETIC: "Synthetic fallback",
}

WAVE_LABELS: Final[Mapping[str, str]] = {
    ThreatWave.FAST.value: "Fast-wave (ballistic / aviation)",
    ThreatWave.SLOW.value: "Slow-wave (UAV / Shahed)",
    ThreatWave.UNKNOWN.value: "Unclassified",
}
WAVE_COLORS: Final[Mapping[str, str]] = {
    ThreatWave.FAST.value: "#D1495B",
    ThreatWave.SLOW.value: "#2E86AB",
    ThreatWave.UNKNOWN.value: "#8D99AE",
}

# Approximate oblast-centre coordinates (longitude, latitude) used as a fixed
# geographic layout for the propagation graph, so it reads like a map.
REGION_COORDS: Final[Mapping[str, tuple[float, float]]] = {
    "Kyiv City": (30.52, 50.45),
    "Kyiv Oblast": (29.95, 49.95),
    "Kharkiv Oblast": (36.23, 49.99),
    "Odesa Oblast": (30.73, 46.48),
    "Dnipropetrovsk Oblast": (35.05, 48.46),
    "Zaporizhzhia Oblast": (35.14, 47.84),
    "Mykolaiv Oblast": (32.00, 46.97),
    "Kherson Oblast": (32.62, 46.64),
    "Sumy Oblast": (34.80, 50.91),
    "Chernihiv Oblast": (31.29, 51.49),
    "Poltava Oblast": (34.55, 49.59),
    "Cherkasy Oblast": (32.06, 49.44),
    "Kirovohrad Oblast": (32.26, 48.51),
    "Vinnytsia Oblast": (28.47, 49.23),
    "Zhytomyr Oblast": (28.66, 50.25),
    "Khmelnytskyi Oblast": (26.98, 49.42),
    "Rivne Oblast": (26.25, 50.62),
    "Volyn Oblast": (25.33, 50.75),
    "Lviv Oblast": (24.03, 49.84),
    "Ternopil Oblast": (25.59, 49.55),
    "Ivano-Frankivsk Oblast": (24.71, 48.92),
    "Chernivtsi Oblast": (25.94, 48.29),
    "Zakarpattia Oblast": (22.30, 48.62),
    "Donetsk Oblast": (37.80, 48.00),
    "Luhansk Oblast": (39.30, 48.57),
}

PREDICTION_COLUMNS: Final[Sequence[str]] = (
    "region",
    "probability",
    "eta_min",
    "eta_time",
    "via",
    "hops",
    "paths",
)


@dataclass(frozen=True)
class CascadeParams:
    """Tunable parameters of the propagation model.

    Attributes:
        max_lag_min: Two alarms of the same threat type form a directed edge
            when the second starts within this many minutes of the first.
        min_edge_weight: Edges observed fewer times than this are hidden in
            the graph view (they still contribute to predictions).
        decay_min: E-folding time (minutes) by which a trigger's influence
            fades once its typical lag to a target has already elapsed.
        hops: Propagation depth considered by the simulator (1 or 2).
    """

    max_lag_min: float = 90.0
    min_edge_weight: int = 2
    decay_min: float = 60.0
    hops: int = 1


@dataclass(frozen=True)
class DataSelection:
    """Everything the sidebar decided about *which* data to show.

    Attributes:
        source: Requested source, ``"csv"`` or ``"synthetic"``.
        csv_path: Path to the CSV (only meaningful for the CSV source).
        num_records: Synthetic row count.
        seed: Synthetic random seed.
        start_date: Synthetic first-alarm date (ISO ``YYYY-MM-DD``).
    """

    source: str
    csv_path: str
    num_records: int
    seed: int
    start_date: str


@dataclass(frozen=True)
class LoadOutcome:
    """Result of loading data through the temporal safeguard.

    Attributes:
        frame: Cutoff-filtered alarms (``start_time < T``).
        source_used: Source actually used after any fallback.
        fallback_reason: Human-readable reason when the CSV source fell back
            to synthetic data, else ``None``.
        history_rows: Rows in the full (uncut) history.
        withheld_rows: Rows removed because ``start_time >= T``.
        masked_end_rows: Rows whose ``end_time`` was masked because the alarm
            was still active at ``T``.
    """

    frame: pd.DataFrame
    source_used: str
    fallback_reason: Optional[str]
    history_rows: int
    withheld_rows: int
    masked_end_rows: int


# --------------------------------------------------------------------------- #
# 2. Cached data access
# --------------------------------------------------------------------------- #
def resolve_source(requested: str, csv_path: str) -> tuple[str, Optional[str]]:
    """Decide which source to use, falling back to synthetic if the CSV is absent.

    Args:
        requested: ``"csv"`` or ``"synthetic"``.
        csv_path: Candidate CSV path.

    Returns:
        ``(source_used, fallback_reason)``; the reason is ``None`` unless a
        fallback happened.
    """
    if requested == SOURCE_CSV:
        if Path(csv_path).is_file():
            return SOURCE_CSV, None
        return (
            SOURCE_SYNTHETIC,
            f"No CSV found at `{csv_path}`. The synthetic generator was engaged "
            "automatically so the dashboard stays operational.",
        )
    return SOURCE_SYNTHETIC, None


@st.cache_data(ttl="1h", max_entries=8, show_spinner="Loading alarm history…")
def load_history(
    source: str, csv_path: str, num_records: int, seed: int, start_date: str
) -> pd.DataFrame:
    """Load the full cleaned history **without** a cutoff.

    Used only for widget defaults (data range) and for counting how many rows
    the temporal safeguard withholds. Nothing displayed as analysis is derived
    from this frame.

    Args:
        source: Resolved source (``"csv"`` or ``"synthetic"``).
        csv_path: CSV path for the CSV source.
        num_records: Synthetic row count.
        seed: Synthetic seed.
        start_date: Synthetic start date (ISO).

    Returns:
        Cleaned alarm history in the loader's canonical schema.
    """
    if source == SOURCE_CSV:
        loader = AirThreatLoader()
        return loader.clean(loader.load_csv(csv_path))
    return generate_synthetic_alarms(num_records=num_records, start_date=start_date, seed=seed)


@st.cache_data(ttl="1h", max_entries=64, show_spinner="Applying temporal cutoff…")
def load_filtered(
    source: str,
    csv_path: str,
    cutoff_iso: str,
    num_records: int,
    seed: int,
    start_date: str,
) -> pd.DataFrame:
    """Load alarms visible at ``T`` through the loader's temporal safeguard.

    Args:
        source: Resolved source (``"csv"`` or ``"synthetic"``).
        csv_path: CSV path for the CSV source.
        cutoff_iso: ``T`` as an ISO-8601 string (hashable cache key).
        num_records: Synthetic row count.
        seed: Synthetic seed.
        start_date: Synthetic start date (ISO).

    Returns:
        A frame where every ``start_time < T`` and every surviving
        ``end_time < T`` (later ends are masked to ``NaT``).
    """
    cutoff = datetime.fromisoformat(cutoff_iso)
    if source == SOURCE_CSV:
        return load_and_filter_alarms(csv_path, cutoff)
    synthetic = generate_synthetic_alarms(num_records=num_records, start_date=start_date, seed=seed)
    return AirThreatLoader().apply_temporal_cutoff(synthetic, cutoff)


def count_masked_ends(filtered: pd.DataFrame, history: pd.DataFrame) -> int:
    """Count rows whose end time was masked by the cutoff.

    Args:
        filtered: Cutoff-filtered frame.
        history: Full history frame.

    Returns:
        Number of rows with a known end in ``history`` but ``NaT`` in
        ``filtered`` (alarms still running at ``T``).
    """
    if filtered.empty or history.empty:
        return 0
    hist_end = (
        history.drop_duplicates(subset=["region", "start_time"])
        .set_index(["region", "start_time"])["end_time"]
        .rename("hist_end")
    )
    joined = filtered.join(hist_end, on=["region", "start_time"])
    return int((joined["end_time"].isna() & joined["hist_end"].notna()).sum())


# --------------------------------------------------------------------------- #
# 3. Pure analytics
# --------------------------------------------------------------------------- #
def apply_threat_filters(
    df: pd.DataFrame, waves: Sequence[str], threat_types: Sequence[str]
) -> pd.DataFrame:
    """Restrict a frame to selected wave classes and threat types.

    Args:
        df: Alarm frame in canonical schema.
        waves: Wave classes to keep (``"fast"``, ``"slow"``, ``"unknown"``).
        threat_types: Threat types to keep; an empty sequence means *all*.

    Returns:
        Filtered copy with a fresh index.
    """
    mask = df["wave_class"].isin(list(waves))
    if threat_types:
        mask &= df["threat_type"].isin(list(threat_types))
    return df.loc[mask].reset_index(drop=True)


def active_alarms(
    df: pd.DataFrame, cutoff: pd.Timestamp, *, max_age_hours: float = ACTIVE_WINDOW_HOURS
) -> pd.DataFrame:
    """Return alarms considered active at ``T`` with an ``elapsed_min`` column.

    Because the loader masks every ``end_time >= T`` to ``NaT``, an active
    alarm is one whose end is unknown. Open-ended rows older than
    ``max_age_hours`` are treated as feed gaps rather than live alarms.

    Args:
        df: Cutoff-filtered frame.
        cutoff: Observation instant ``T`` (UTC).
        max_age_hours: Staleness window for open-ended rows.

    Returns:
        Active rows sorted by most recent start, with ``elapsed_min``.
    """
    if df.empty:
        return df.assign(elapsed_min=pd.Series(dtype="float64"))
    start = df["start_time"]
    end = df["end_time"]
    fresh = start >= cutoff - pd.Timedelta(hours=max_age_hours)
    live = (end.notna() & (end > cutoff)) | (end.isna() & fresh)
    out = df.loc[live].copy()
    out["elapsed_min"] = (cutoff - out["start_time"]).dt.total_seconds() / 60.0
    return out.sort_values("start_time", ascending=False).reset_index(drop=True)


def cascade_velocity(
    df: pd.DataFrame, cutoff: pd.Timestamp, *, window_hours: float = 6.0
) -> tuple[float, float]:
    """Alarm onset rate (alarms per hour) in the window before ``T``.

    Args:
        df: Cutoff-filtered frame.
        cutoff: Observation instant ``T``.
        window_hours: Width of the trailing window.

    Returns:
        ``(current, previous)`` rates for ``[T-w, T)`` and ``[T-2w, T-w)``.
    """
    if df.empty or window_hours <= 0:
        return 0.0, 0.0
    w = pd.Timedelta(hours=window_hours)
    start = df["start_time"]
    current = int(((start >= cutoff - w) & (start < cutoff)).sum())
    previous = int(((start >= cutoff - 2 * w) & (start < cutoff - w)).sum())
    return current / window_hours, previous / window_hours


def hourly_onsets(df: pd.DataFrame, cutoff: pd.Timestamp, *, hours: int = 24) -> list[int]:
    """Alarm onsets per hour for the ``hours`` hours ending at ``T``.

    Args:
        df: Cutoff-filtered frame.
        cutoff: Observation instant ``T``.
        hours: Number of hourly buckets.

    Returns:
        A list of ``hours`` integer counts, oldest first (sparkline input).
    """
    # The last bucket is the hour containing the final instant before T, so a
    # cutoff exactly on the hour does not produce a trailing empty bucket.
    end = (cutoff - pd.Timedelta(nanoseconds=1)).floor("h")
    buckets = pd.date_range(end=end, periods=hours, freq="h", tz="UTC")
    if df.empty:
        return [0] * hours
    counts = df["start_time"].dt.floor("h").value_counts()
    return [int(counts.get(b, 0)) for b in buckets]


def daily_wave_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Alarm onsets per UTC day, one column per wave class.

    Args:
        df: Alarm frame.

    Returns:
        Frame indexed by naive UTC date with integer columns per wave.
    """
    if df.empty:
        return pd.DataFrame()
    day = df["start_time"].dt.floor("D").dt.tz_localize(None)
    table = pd.crosstab(day, df["wave_class"])
    table.index.name = "day"
    return table


def hour_of_day_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Alarm onsets by UTC hour of day, one column per wave class.

    Args:
        df: Alarm frame.

    Returns:
        Frame indexed 0..23 with integer columns per wave.
    """
    if df.empty:
        return pd.DataFrame()
    table = pd.crosstab(df["start_time"].dt.hour, df["wave_class"])
    table = table.reindex(range(24), fill_value=0)
    table.index.name = "hour (UTC)"
    return table


def threat_type_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Alarm counts per threat type with the wave class attached.

    Args:
        df: Alarm frame.

    Returns:
        Columns ``threat_type``, ``wave_class``, ``alarms`` sorted descending.
    """
    if df.empty:
        return pd.DataFrame(columns=["threat_type", "wave_class", "alarms"])
    out = (
        df.groupby(["threat_type", "wave_class"], observed=True)
        .size()
        .rename("alarms")
        .reset_index()
        .sort_values("alarms", ascending=False)
        .reset_index(drop=True)
    )
    return out


def build_propagation_graph(df: pd.DataFrame, *, max_lag_min: float = 90.0) -> nx.DiGraph:
    """Build the regional propagation graph from cutoff-filtered alarms.

    A directed edge ``A -> B`` is recorded every time region ``B`` raises an
    alarm of the same threat type within ``max_lag_min`` minutes after region
    ``A``. Edge attributes:

    * ``weight``: number of such co-occurrences;
    * ``median_lag_min`` / ``mean_lag_min``: lag statistics in minutes;
    * ``probability``: ``weight / alarms(A)``, the empirical chance that an
      alarm in ``A`` is followed by one in ``B`` within the lag window.

    Node attribute ``alarms`` holds the per-region alarm count.

    Args:
        df: Alarm frame (must already be temporally safeguarded).
        max_lag_min: Co-occurrence window in minutes.

    Returns:
        A :class:`networkx.DiGraph`; empty when ``df`` is empty.
    """
    graph = nx.DiGraph()
    if df.empty:
        return graph
    for region, count in df["region"].value_counts().items():
        graph.add_node(str(region), alarms=int(count))

    lags: dict[tuple[str, str], list[float]] = {}
    lag_ns = np.timedelta64(int(max_lag_min * 60_000_000_000), "ns")
    for _, group in df.groupby("threat_type", observed=True):
        group = group.sort_values("start_time")
        starts = group["start_time"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
        regions = group["region"].to_numpy()
        upper = np.searchsorted(starts, starts + lag_ns, side="right")
        for i in range(len(group)):
            for j in range(i + 1, int(upper[i])):
                if regions[j] == regions[i]:
                    continue
                lag = float((starts[j] - starts[i]) / np.timedelta64(1, "m"))
                lags.setdefault((str(regions[i]), str(regions[j])), []).append(lag)

    for (src, dst), values in lags.items():
        arr = np.asarray(values, dtype=float)
        src_alarms = max(int(graph.nodes[src]["alarms"]), 1)
        graph.add_edge(
            src,
            dst,
            weight=int(arr.size),
            median_lag_min=float(np.median(arr)),
            mean_lag_min=float(arr.mean()),
            probability=float(min(1.0, arr.size / src_alarms)),
        )
    return graph


def edge_table(graph: nx.DiGraph, *, min_weight: int = 1, limit: int = 25) -> pd.DataFrame:
    """Tabulate the strongest propagation corridors.

    Args:
        graph: Propagation graph.
        min_weight: Minimum co-occurrence count.
        limit: Maximum rows.

    Returns:
        Columns ``source``, ``target``, ``co_occurrences``, ``median_lag_min``,
        ``probability`` sorted by count then probability.
    """
    rows = [
        {
            "source": u,
            "target": v,
            "co_occurrences": d["weight"],
            "median_lag_min": d["median_lag_min"],
            "probability": d["probability"],
        }
        for u, v, d in graph.edges(data=True)
        if d["weight"] >= min_weight
    ]
    columns = ["source", "target", "co_occurrences", "median_lag_min", "probability"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["co_occurrences", "probability"], ascending=False)
        .head(limit)
        .reset_index(drop=True)
    )


def node_roles(graph: nx.DiGraph, *, top_n: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank regions as cascade *sources* and *sinks* by weighted degree.

    Args:
        graph: Propagation graph.
        top_n: Rows per table.

    Returns:
        ``(sources, sinks)`` frames with ``region`` and ``weighted_degree``.
    """
    cols = ["region", "weighted_degree"]
    if graph.number_of_nodes() == 0:
        return pd.DataFrame(columns=cols), pd.DataFrame(columns=cols)
    out_deg = pd.DataFrame(graph.out_degree(weight="weight"), columns=cols)
    in_deg = pd.DataFrame(graph.in_degree(weight="weight"), columns=cols)
    sources = out_deg.sort_values("weighted_degree", ascending=False).head(top_n)
    sinks = in_deg.sort_values("weighted_degree", ascending=False).head(top_n)
    return sources.reset_index(drop=True), sinks.reset_index(drop=True)


def predict_cascade(
    graph: nx.DiGraph,
    triggers: Sequence[tuple[str, float]],
    cutoff: pd.Timestamp,
    *,
    top_k: int = 8,
    hops: int = 1,
    decay_min: float = 60.0,
) -> pd.DataFrame:
    """Rank regions most likely to be alarmed next given active triggers.

    For each trigger region ``r`` that started ``e`` minutes before ``T`` and
    each successor ``s`` in the graph, the one-hop contribution is::

        p = P(s | r) * exp(-max(0, e - median_lag(r, s)) / decay_min)
        eta = max(0, median_lag(r, s) - e)

    so a trigger whose typical lag has already passed without ``s`` alarming
    loses influence. With ``hops == 2``, second-order paths ``r -> s -> t``
    contribute ``p(r, s) * p(s, t)`` with ``eta(s) + median_lag(s, t)``.
    Contributions from several triggers are combined with a noisy-OR
    (``1 - prod(1 - p_i)``); the ETA shown is that of the strongest path.
    Trigger regions themselves are never predicted.

    Args:
        graph: Propagation graph built from cutoff-filtered data.
        triggers: ``(region, elapsed_minutes)`` pairs active at ``T``.
        cutoff: Observation instant ``T``.
        top_k: Number of rows to return.
        hops: Propagation depth, ``1`` or ``2``.
        decay_min: Influence decay constant in minutes.

    Returns:
        Frame with :data:`PREDICTION_COLUMNS`, best candidates first.
    """
    trigger_regions = {region for region, _ in triggers}
    paths: dict[str, list[tuple[float, float, str, int]]] = {}

    def successors(region: str) -> list[tuple[str, float, float]]:
        return [
            (succ, float(data["probability"]), float(data["median_lag_min"]))
            for succ, data in graph[region].items()
        ]

    for region, elapsed in triggers:
        if region not in graph:
            continue
        for succ, p1, lag1 in successors(region):
            if succ in trigger_regions:
                continue
            overdue = max(0.0, elapsed - lag1)
            p = p1 * math.exp(-overdue / max(decay_min, 1e-9))
            eta = max(0.0, lag1 - elapsed)
            paths.setdefault(succ, []).append((p, eta, region, 1))
            if hops < 2:
                continue
            for succ2, p2, lag2 in successors(succ):
                if succ2 in trigger_regions or succ2 == region:
                    continue
                paths.setdefault(succ2, []).append((p * p2, eta + lag2, region, 2))

    rows = []
    for target, items in paths.items():
        survival = 1.0
        for p, _, _, _ in items:
            survival *= 1.0 - min(max(p, 0.0), 1.0)
        best = max(items, key=lambda item: item[0])
        rows.append(
            {
                "region": target,
                "probability": float(min(1.0, 1.0 - survival)),
                "eta_min": float(best[1]),
                "eta_time": cutoff + pd.Timedelta(minutes=best[1]),
                "via": best[2],
                "hops": int(best[3]),
                "paths": len(items),
            }
        )
    if not rows:
        return pd.DataFrame(columns=list(PREDICTION_COLUMNS))
    return (
        pd.DataFrame(rows, columns=list(PREDICTION_COLUMNS))
        .sort_values(["probability", "eta_min"], ascending=[False, True])
        .head(top_k)
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------- #
# 3b. Graph rendering (matplotlib)
# --------------------------------------------------------------------------- #
def short_region_name(region: str) -> str:
    """Compact label for graph nodes (``"Kyiv Oblast"`` -> ``"Kyiv reg."``)."""
    if region == "Kyiv City":
        return "Kyiv"
    if region == "Kyiv Oblast":
        return "Kyiv reg."
    return region.replace(" Oblast", "")


def layout_positions(graph: nx.Graph) -> dict[str, tuple[float, float]]:
    """Geographic layout with a spring-layout fallback for unknown regions.

    Args:
        graph: Any graph whose nodes are region names.

    Returns:
        Mapping of node to ``(x, y)``.
    """
    known = {n: REGION_COORDS[n] for n in graph.nodes if n in REGION_COORDS}
    if len(known) == graph.number_of_nodes():
        return known
    if known:
        return nx.spring_layout(graph, pos=known, fixed=list(known), seed=42)
    return nx.spring_layout(graph, seed=42)


def render_graph_figure(
    graph: nx.DiGraph,
    *,
    highlight: Optional[Mapping[str, tuple[str, float]]] = None,
    min_weight: int = 1,
    max_edges: int = 80,
    show_labels: bool = True,
    title: str = "",
    dark: bool = False,
) -> Figure:
    """Draw the propagation graph on a geographic layout.

    Args:
        graph: Propagation graph.
        highlight: Optional ``region -> (color, alpha)`` overrides for nodes
            (used to mark triggers and predicted targets).
        min_weight: Hide edges observed fewer times than this.
        max_edges: Keep only the strongest ``max_edges`` edges.
        show_labels: Draw region labels.
        title: Figure title.
        dark: Use light text for dark app themes.

    Returns:
        A matplotlib :class:`~matplotlib.figure.Figure` with a transparent
        background. The caller owns the figure and should close it.
    """
    text_color = "#E6E6E6" if dark else "#222222"
    fig, ax = plt.subplots(figsize=(10.5, 6.6))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    ax.set_axis_off()
    if title:
        ax.set_title(title, color=text_color, fontsize=12, loc="left")

    if graph.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No alarms before T", ha="center", va="center", color=text_color)
        return fig

    pos = layout_positions(graph)
    nodes = list(graph.nodes)
    alarms = np.asarray([graph.nodes[n].get("alarms", 1) for n in nodes], dtype=float)
    sizes = 140 + 900 * (alarms / alarms.max() if alarms.max() > 0 else alarms)
    highlight = highlight or {}
    colors = [highlight.get(n, ("#9AA5B1", 0.9))[0] for n in nodes]
    alphas = [highlight.get(n, ("#9AA5B1", 0.9))[1] for n in nodes]

    edges = sorted(
        (e for e in graph.edges(data=True) if e[2]["weight"] >= min_weight),
        key=lambda e: e[2]["weight"],
        reverse=True,
    )[:max_edges]
    if edges:
        weights = np.asarray([e[2]["weight"] for e in edges], dtype=float)
        widths = 0.6 + 4.0 * weights / weights.max()
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=[(u, v) for u, v, _ in edges],
            width=list(widths),
            edge_color="#6C7A89",
            alpha=0.45,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=11,
            connectionstyle="arc3,rad=0.15",
            node_size=list(sizes),
            ax=ax,
        )
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=nodes,
        node_size=list(sizes),
        node_color=colors,
        alpha=alphas,
        linewidths=0.8,
        edgecolors=text_color,
        ax=ax,
    )
    if show_labels:
        nx.draw_networkx_labels(
            graph,
            pos,
            labels={n: short_region_name(n) for n in nodes},
            font_size=8,
            font_color=text_color,
            ax=ax,
        )
    ax.margins(0.08)

    legend_items = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#9AA5B1", markersize=9,
               label="Region (size = alarms before T)"),
        Line2D([0], [0], color="#6C7A89", lw=3, alpha=0.6, label="Propagation edge (width = count)"),
    ]
    legend_colors = {c for c, _ in highlight.values()}
    if "#D1495B" in legend_colors:
        legend_items.append(Line2D([0], [0], marker="o", color="none", markerfacecolor="#D1495B",
                                   markersize=9, label="Active trigger at T"))
    if "#F2A541" in legend_colors:
        legend_items.append(Line2D([0], [0], marker="o", color="none", markerfacecolor="#F2A541",
                                   markersize=9, label="Predicted next target"))
    ax.legend(handles=legend_items, loc="lower left", fontsize=8, frameon=False,
              labelcolor=text_color)
    return fig


def _theme_is_dark() -> bool:
    """Best-effort detection of the active Streamlit theme."""
    try:
        return getattr(st.context.theme, "type", "light") == "dark"
    except Exception:  # pragma: no cover - defensive; theme API may be absent
        return False


# --------------------------------------------------------------------------- #
# 4. Rendering helpers
# --------------------------------------------------------------------------- #
def default_cutoff(history: pd.DataFrame) -> datetime:
    """Pick a default ``T`` about 60% through the history, during live activity.

    ``T`` is placed ten minutes after the onset of the alarm at the 60th
    percentile of the chronological order (rounded down to 5 minutes), so the
    page opens with at least one alarm active and the simulator has triggers.

    Args:
        history: Full history frame.

    Returns:
        A naive UTC datetime.
    """
    if history.empty:
        return datetime(2026, 2, 1, 12, 0)
    starts = history["start_time"].sort_values().reset_index(drop=True)
    anchor = starts.iloc[min(int(len(starts) * 0.6), len(starts) - 1)]
    point = (anchor + pd.Timedelta(minutes=10)).floor("5min")
    return point.tz_convert("UTC").tz_localize(None).to_pydatetime()


def _set_cutoff_state(value: datetime) -> None:
    """Write a naive UTC datetime into the cutoff widgets' session state."""
    st.session_state["cutoff_date"] = value.date()
    st.session_state["cutoff_time"] = value.time().replace(second=0, microsecond=0)


def _shift_cutoff(delta: timedelta) -> None:
    """Button callback: move ``T`` by ``delta``."""
    current = datetime.combine(st.session_state["cutoff_date"], st.session_state["cutoff_time"])
    _set_cutoff_state(current + delta)


def _reset_cutoff(history_default: datetime) -> None:
    """Button callback: reset ``T`` to the dataset midpoint."""
    _set_cutoff_state(history_default)


def render_source_controls() -> DataSelection:
    """Sidebar block: data source toggle and generator parameters.

    Returns:
        The user's data selection.
    """
    st.subheader("Data source", icon=":material/database:")
    default_source = SOURCE_CSV if Path(DEFAULT_CSV_PATH).is_file() else SOURCE_SYNTHETIC
    source = st.segmented_control(
        "Source",
        options=[SOURCE_CSV, SOURCE_SYNTHETIC],
        format_func=SOURCE_LABELS.get,
        default=default_source,
        key="source",
        required=True,
        label_visibility="collapsed",
        width="stretch",
    )
    csv_path = DEFAULT_CSV_PATH
    if source == SOURCE_CSV:
        csv_path = st.text_input(
            "CSV path",
            value=DEFAULT_CSV_PATH,
            key="csv_path",
            help="Columns such as region/start/end/threat are auto-detected.",
        ).strip() or DEFAULT_CSV_PATH

    with st.expander("Synthetic generator", icon=":material/science:", expanded=False):
        num_records = int(
            st.number_input("Records", 100, 20_000, 2_000, step=100, key="synthetic_records")
        )
        seed = int(st.number_input("Seed", 0, 99_999, 42, step=1, key="synthetic_seed"))
        start = st.date_input("First alarm date", date(2026, 1, 1), key="synthetic_start")
        start_date = (start if isinstance(start, date) else date(2026, 1, 1)).isoformat()
    return DataSelection(
        source=str(source),
        csv_path=csv_path,
        num_records=num_records,
        seed=seed,
        start_date=start_date,
    )


def render_cutoff_controls(history: pd.DataFrame, dataset_signature: str) -> pd.Timestamp:
    """Sidebar block: date & time picker for the temporal cutoff ``T``.

    When the underlying dataset changes, ``T`` is reset to the new data's
    midpoint so the page never opens on an empty slice.

    Args:
        history: Full history (for the range caption and default).
        dataset_signature: Opaque string identifying the loaded dataset.

    Returns:
        ``T`` as a UTC :class:`pandas.Timestamp`.
    """
    st.subheader("Temporal cutoff T", icon=":material/schedule:")
    midpoint = default_cutoff(history)
    if st.session_state.get("_dataset_signature") != dataset_signature:
        st.session_state["_dataset_signature"] = dataset_signature
        _set_cutoff_state(midpoint)
    st.session_state.setdefault("cutoff_date", midpoint.date())
    st.session_state.setdefault("cutoff_time", midpoint.time())

    if not history.empty:
        first = history["start_time"].min()
        last = history["start_time"].max()
        st.caption(f"Data spans {first:%Y-%m-%d %H:%M} → {last:%Y-%m-%d %H:%M} UTC")

    cutoff_date = st.date_input("Date (UTC)", key="cutoff_date")
    cutoff_time = st.time_input("Time (UTC)", key="cutoff_time", step=timedelta(minutes=15))
    with st.container(horizontal=True, gap="xsmall"):
        st.button("−1 d", on_click=_shift_cutoff, args=(timedelta(days=-1),), key="nudge_m1d")
        st.button("−6 h", on_click=_shift_cutoff, args=(timedelta(hours=-6),), key="nudge_m6h")
        st.button("+6 h", on_click=_shift_cutoff, args=(timedelta(hours=6),), key="nudge_p6h")
        st.button("+1 d", on_click=_shift_cutoff, args=(timedelta(days=1),), key="nudge_p1d")
    st.button(
        "Reset to data midpoint",
        icon=":material/restart_alt:",
        on_click=_reset_cutoff,
        args=(midpoint,),
        key="reset_cutoff",
        width="stretch",
    )
    picked_date = cutoff_date if isinstance(cutoff_date, date) else midpoint.date()
    picked_time = cutoff_time if isinstance(cutoff_time, time) else midpoint.time()
    return pd.Timestamp(datetime.combine(picked_date, picked_time, tzinfo=UTC))


def render_threat_filters(history: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Sidebar block: wave-class pills and threat-type multiselect.

    Args:
        history: Full history, used to list the threat types that exist.

    Returns:
        ``(waves, threat_types)``; an empty ``threat_types`` means *all*.
    """
    st.subheader("Threat filter", icon=":material/filter_alt:")
    present_waves = [w for w in WAVE_LABELS if w in set(history["wave_class"])] or ["fast", "slow"]
    picked = st.pills(
        "Wave class",
        options=present_waves,
        default=present_waves,
        selection_mode="multi",
        format_func=WAVE_LABELS.get,
        key="wave_filter",
        width="stretch",
        help="Deselect everything to show all wave classes.",
    )
    waves = list(picked) if isinstance(picked, list) and picked else list(present_waves)
    options = sorted(
        t for t in history["threat_type"].unique()
        if THREAT_TYPES.get(t, ThreatWave.UNKNOWN).value in waves
    )
    threat_types = st.multiselect(
        "Threat types",
        options=options,
        default=[],
        placeholder="All threat types",
        key="threat_type_filter",
        format_func=lambda t: t.replace("_", " "),
    )
    return waves, [str(t) for t in threat_types]


def render_cascade_controls() -> CascadeParams:
    """Sidebar block: propagation-model parameters shared by graph & simulator.

    Returns:
        The chosen :class:`CascadeParams`.
    """
    with st.expander("Cascade model", icon=":material/hub:", expanded=False):
        max_lag = st.slider("Max propagation lag (min)", 15, 240, 90, step=15, key="max_lag")
        min_weight = st.slider("Min edge count shown", 1, 15, 2, key="min_edge_weight")
        decay = st.slider("Trigger influence decay (min)", 15, 240, 60, step=15, key="decay_min")
        hops = st.segmented_control(
            "Propagation depth", options=[1, 2], default=1, required=True,
            format_func=lambda h: f"{h} hop" if h == 1 else f"{h} hops", key="hops",
        )
    return CascadeParams(
        max_lag_min=float(max_lag),
        min_edge_weight=int(min_weight),
        decay_min=float(decay),
        hops=int(hops) if hops in (1, 2) else 1,
    )


def load_data(selection: DataSelection, cutoff: pd.Timestamp, history: pd.DataFrame,
              source_used: str, fallback_reason: Optional[str]) -> LoadOutcome:
    """Run the temporal safeguard for ``T`` and collect audit counters.

    Args:
        selection: Sidebar data selection.
        cutoff: Observation instant ``T``.
        history: Full history frame (for withheld / masked counts).
        source_used: Resolved source.
        fallback_reason: Reason text if a fallback occurred.

    Returns:
        A :class:`LoadOutcome`.
    """
    frame = load_filtered(
        source_used,
        selection.csv_path,
        cutoff.isoformat(),
        selection.num_records,
        selection.seed,
        selection.start_date,
    )
    return LoadOutcome(
        frame=frame,
        source_used=source_used,
        fallback_reason=fallback_reason,
        history_rows=len(history),
        withheld_rows=max(len(history) - len(frame), 0),
        masked_end_rows=count_masked_ends(frame, history),
    )


def render_status_banner(outcome: LoadOutcome, cutoff: pd.Timestamp, history: pd.DataFrame,
                         visible_rows: int) -> None:
    """Show safeguard status, synthetic/fallback warnings and edge cases.

    Args:
        outcome: Load outcome for the current ``T``.
        cutoff: Observation instant ``T``.
        history: Full history frame.
        visible_rows: Rows left after the threat filter.
    """
    if outcome.fallback_reason:
        st.warning(outcome.fallback_reason, icon=":material/warning:",
                   title="Automatic fallback to synthetic data")
    if outcome.source_used == SOURCE_SYNTHETIC:
        st.warning(
            "All figures below come from the **synthetic generator**, not from "
            "observed alarms. Use them to evaluate the pipeline mechanics, not "
            "real-world threat levels.",
            icon=":material/science:",
            title="Synthetic mode active",
        )

    edge_cases: list[str] = []
    if not history.empty:
        first, last = history["start_time"].min(), history["start_time"].max()
        if cutoff <= first:
            edge_cases.append(f"T is at or before the first recorded alarm ({first:%Y-%m-%d %H:%M} UTC); "
                              "nothing is observable yet.")
        elif cutoff > last:
            edge_cases.append(f"T is after the last recorded alarm ({last:%Y-%m-%d %H:%M} UTC); the full "
                              "history is visible and no rows are withheld.")
    if outcome.masked_end_rows:
        edge_cases.append(f"{outcome.masked_end_rows} alarm(s) were still running at T; their end times "
                          "were masked so no post-T information leaks into durations.")
    unknown = int((outcome.frame["wave_class"] == ThreatWave.UNKNOWN.value).sum())
    if unknown:
        edge_cases.append(f"{unknown} row(s) carry an unrecognised threat type and are classed as "
                          "'unknown'.")
    if visible_rows == 0 and not outcome.frame.empty:
        edge_cases.append("The current threat filter hides every alarm before T.")
    if edge_cases:
        st.warning("\n".join(f"- {item}" for item in edge_cases), icon=":material/report:",
                   title="Temporal-cutoff edge cases")

    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.badge("Temporal safeguard verified", icon=":material/verified_user:", color="green")
        st.badge(f"T = {cutoff:%Y-%m-%d %H:%M} UTC", icon=":material/schedule:", color="blue")
        st.badge(
            SOURCE_LABELS[outcome.source_used],
            icon=":material/science:" if outcome.source_used == SOURCE_SYNTHETIC else ":material/table:",
            color="orange" if outcome.source_used == SOURCE_SYNTHETIC else "gray",
        )
        st.caption(
            f"{len(outcome.frame):,} alarms visible · {outcome.withheld_rows:,} withheld "
            f"(start ≥ T) · {visible_rows:,} after filters"
        )


def render_empty_state(
    cutoff: pd.Timestamp, history: pd.DataFrame, *, reason: str, slot: str
) -> None:
    """Friendly empty state with a one-click fix.

    Args:
        cutoff: Observation instant ``T``.
        history: Full history frame.
        reason: Short explanation to show.
        slot: Unique suffix (e.g. the tab name) so the button key is unique
            when the empty state is rendered in several tabs.
    """
    st.info(f"{reason} (T = {cutoff:%Y-%m-%d %H:%M} UTC)", icon=":material/hourglass_empty:")
    if not history.empty:
        st.button(
            "Jump to data midpoint",
            icon=":material/center_focus_strong:",
            on_click=_reset_cutoff,
            args=(default_cutoff(history),),
            key=f"empty_reset_{slot}",
        )


def render_overview_tab(df: pd.DataFrame, active: pd.DataFrame, cutoff: pd.Timestamp) -> None:
    """Tab 1: KPI row, temporal charts, wave mix and recent alarms.

    Args:
        df: Filtered, cutoff-safe alarm frame (non-empty).
        active: Alarms active at ``T``.
        cutoff: Observation instant ``T``.
    """
    current, previous = cascade_velocity(df, cutoff)
    delta = current - previous
    wave_share = df["wave_class"].value_counts(normalize=True)
    fast_share = float(wave_share.get(ThreatWave.FAST.value, 0.0))

    with st.container(horizontal=True):
        st.metric("Alarms before T", f"{len(df):,}", border=True,
                  help="Rows with start_time strictly before T after filters.")
        st.metric("Active at T", f"{len(active):,}", border=True,
                  delta=f"{active['region'].nunique()} regions" if len(active) else None,
                  delta_arrow="off", help=f"Open-ended alarms started within the last "
                  f"{ACTIVE_WINDOW_HOURS:.0f} h before T.")
        st.metric("Cascade velocity", f"{current:.1f} /h", delta=f"{delta:+.1f} vs prior 6 h",
                  border=True, chart_data=hourly_onsets(df, cutoff), chart_type="bar",
                  help="Alarm onsets per hour in the 6 h before T; sparkline shows the last 24 h.")
        st.metric("Fast-wave share", f"{fast_share:.0%}", border=True,
                  delta=f"{1 - fast_share:.0%} slow-wave", delta_arrow="off",
                  help="Share of ballistic / aviation / cruise alarms among visible rows.")

    col_left, col_right = st.columns((3, 2))
    with col_left, st.container(border=True):
        st.subheader("Alarm onsets per day", icon=":material/calendar_month:")
        daily = daily_wave_counts(df)
        st.bar_chart(daily, stack=True, color=[WAVE_COLORS[c] for c in daily.columns],
                     y_label="alarms", height=280)
    with col_right, st.container(border=True):
        st.subheader("Threat mix", icon=":material/pie_chart:")
        mix = threat_type_counts(df)
        mix["threat_type"] = mix["threat_type"].str.replace("_", " ")
        st.bar_chart(mix, x="threat_type", y="alarms", color="wave_class", horizontal=True,
                     x_label="alarms", y_label="", height=280)

    col_a, col_b = st.columns((2, 3))
    with col_a, st.container(border=True):
        st.subheader("Hour-of-day profile", icon=":material/nightlight:")
        profile = hour_of_day_profile(df)
        st.bar_chart(profile, stack=True, color=[WAVE_COLORS[c] for c in profile.columns],
                     y_label="alarms", height=240)
        st.caption("Slow-wave UAV activity concentrates overnight; fast-wave events are flatter.")
    with col_b, st.container(border=True):
        st.subheader("Latest alarms before T", icon=":material/history:")
        recent = df.sort_values("start_time", ascending=False).head(12)
        st.dataframe(
            recent[["region", "start_time", "end_time", "threat_type", "wave_class", "duration_min"]],
            hide_index=True,
            height=240,
            column_config={
                "region": st.column_config.TextColumn("Region"),
                "start_time": st.column_config.DatetimeColumn("Start (UTC)", format="MMM DD, HH:mm"),
                "end_time": st.column_config.DatetimeColumn("End (UTC)", format="MMM DD, HH:mm"),
                "threat_type": st.column_config.TextColumn("Threat"),
                "wave_class": st.column_config.TextColumn("Wave"),
                "duration_min": st.column_config.NumberColumn("Duration (min)", format="%.0f"),
            },
        )

    with st.expander("Filtered rows & export", icon=":material/download:"):
        st.dataframe(df, hide_index=True, height=300)
        st.download_button(
            "Download visible rows as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"alarms_before_{cutoff:%Y%m%dT%H%M}Z.csv",
            mime="text/csv",
            icon=":material/download:",
        )


def render_graph_tab(graph: nx.DiGraph, active: pd.DataFrame, params: CascadeParams) -> None:
    """Tab 2: propagation graph figure plus corridor and role tables.

    Args:
        graph: Propagation graph from the visible data.
        active: Alarms active at ``T`` (highlighted in red).
        params: Cascade parameters from the sidebar.
    """
    st.caption(
        f"Edge A → B counts how often B alarmed within {params.max_lag_min:.0f} min after A for "
        "the same threat type. Layout uses oblast-centre coordinates. Only pre-T alarms feed the graph."
    )
    with st.container(horizontal=True, vertical_alignment="bottom"):
        max_edges = st.slider("Strongest edges shown", 10, 200, 60, step=10, key="graph_max_edges")
        show_labels = st.toggle("Region labels", value=True, key="graph_labels")

    if graph.number_of_edges() == 0:
        st.info("No propagation edges yet: fewer than two same-type alarms fall inside the lag window "
                "before T. Widen the lag in the sidebar or move T later.", icon=":material/hub:")
    highlight = {str(r): ("#D1495B", 1.0) for r in active["region"].unique()}
    fig = render_graph_figure(
        graph, highlight=highlight, min_weight=params.min_edge_weight, max_edges=max_edges,
        show_labels=show_labels, title="Regional propagation graph (before T)", dark=_theme_is_dark(),
    )
    st.pyplot(fig)
    plt.close(fig)

    col_edges, col_roles = st.columns((3, 2))
    with col_edges, st.container(border=True):
        st.subheader("Strongest corridors", icon=":material/alt_route:")
        corridors = edge_table(graph, min_weight=params.min_edge_weight, limit=20)
        st.dataframe(
            corridors,
            hide_index=True,
            column_config={
                "source": "From",
                "target": "To",
                "co_occurrences": st.column_config.NumberColumn("Count", format="%d"),
                "median_lag_min": st.column_config.NumberColumn("Median lag (min)", format="%.0f"),
                "probability": st.column_config.ProgressColumn("P(to | from)", min_value=0,
                                                               max_value=1, format="percent"),
            },
        )
    with col_roles, st.container(border=True):
        st.subheader("Cascade roles", icon=":material/swap_calls:")
        sources, sinks = node_roles(graph)
        role_cols = st.columns(2)
        with role_cols[0]:
            st.markdown("**Top sources**")
            st.dataframe(sources, hide_index=True,
                         column_config={"region": "Region",
                                        "weighted_degree": st.column_config.NumberColumn("Out", format="%d")})
        with role_cols[1]:
            st.markdown("**Top sinks**")
            st.dataframe(sinks, hide_index=True,
                         column_config={"region": "Region",
                                        "weighted_degree": st.column_config.NumberColumn("In", format="%d")})


def render_simulator_tab(graph: nx.DiGraph, active: pd.DataFrame, cutoff: pd.Timestamp,
                         params: CascadeParams, all_regions: Sequence[str]) -> None:
    """Tab 3: cascade simulator with top-K targets and arrival estimates.

    Args:
        graph: Propagation graph from the visible data.
        active: Alarms active at ``T`` (default triggers).
        cutoff: Observation instant ``T``.
        params: Cascade parameters from the sidebar.
        all_regions: Regions offered for manual what-if triggers.
    """
    st.caption(
        "Triggers are the alarms active at T. Each successor's chance is the empirical "
        "P(target | trigger) discounted once the typical lag has elapsed; multiple triggers combine "
        "with a noisy-OR. Arrival estimates are median historical lags measured from each trigger's start."
    )
    with st.container(horizontal=True, vertical_alignment="bottom"):
        use_observed = st.toggle("Use observed active alarms at T", value=True, key="sim_use_observed")
        top_k = st.slider("Top-K targets", 3, 15, 8, key="sim_top_k")

    if use_observed:
        triggers_df = (
            active.groupby("region", observed=True)["elapsed_min"].min().reset_index()
            if not active.empty else pd.DataFrame(columns=["region", "elapsed_min"])
        )
    else:
        with st.container(horizontal=True, vertical_alignment="bottom"):
            chosen = st.multiselect("What-if trigger regions", options=list(all_regions),
                                    default=list(all_regions[:1]), key="sim_manual_regions")
            elapsed = st.slider("Minutes since triggers started", 0, 180, 0, step=5, key="sim_elapsed")
        triggers_df = pd.DataFrame({"region": chosen, "elapsed_min": float(elapsed)})

    if triggers_df.empty:
        st.info("No alarms are active at T, so there is nothing to propagate. Move T to a busier moment "
                "or switch off *Use observed active alarms* to run a what-if scenario.",
                icon=":material/notifications_paused:")
        return

    triggers = [(str(r), float(e)) for r, e in triggers_df.itertuples(index=False)]
    predictions = predict_cascade(graph, triggers, cutoff, top_k=top_k, hops=params.hops,
                                  decay_min=params.decay_min)

    col_trig, col_pred = st.columns((1, 2))
    with col_trig, st.container(border=True):
        st.subheader("Active triggers", icon=":material/notifications_active:")
        trig_view = triggers_df.sort_values("elapsed_min")
        trig_view["in_graph"] = trig_view["region"].map(lambda r: r in graph)
        st.dataframe(
            trig_view, hide_index=True,
            column_config={
                "region": "Region",
                "elapsed_min": st.column_config.NumberColumn("Elapsed (min)", format="%.0f"),
                "in_graph": st.column_config.CheckboxColumn("Has history"),
            },
        )
        missing = [r for r in trig_view.loc[~trig_view["in_graph"], "region"]]
        if missing:
            st.caption("No pre-T history for: " + ", ".join(missing))
    with col_pred, st.container(border=True):
        st.subheader(f"Top-{top_k} predicted next regions", icon=":material/trending_up:")
        if predictions.empty:
            st.info("The active triggers have no outgoing propagation edges before T. Try a wider lag "
                    "window, two hops, or a later T.", icon=":material/hub:")
        else:
            st.dataframe(
                predictions,
                hide_index=True,
                column_config={
                    "region": "Region",
                    "probability": st.column_config.ProgressColumn("Probability", min_value=0,
                                                                   max_value=1, format="percent"),
                    "eta_min": st.column_config.NumberColumn("ETA (min)", format="%.0f"),
                    "eta_time": st.column_config.DatetimeColumn("Est. arrival (UTC)", format="HH:mm"),
                    "via": "Via trigger",
                    "hops": st.column_config.NumberColumn("Hops", format="%d"),
                    "paths": st.column_config.NumberColumn("Paths", format="%d"),
                },
            )

    highlight: dict[str, tuple[str, float]] = {r: ("#D1495B", 1.0) for r, _ in triggers}
    for row in predictions.itertuples(index=False):
        highlight[str(row.region)] = ("#F2A541", float(0.35 + 0.65 * row.probability))
    fig = render_graph_figure(
        graph, highlight=highlight, min_weight=params.min_edge_weight, max_edges=60,
        title="Simulated cascade at T", dark=_theme_is_dark(),
    )
    st.pyplot(fig)
    plt.close(fig)

    if not predictions.empty:
        with st.container(border=True):
            st.subheader("Arrival timeline", icon=":material/timeline:")
            timeline = predictions.set_index("region")[["eta_min"]].sort_values("eta_min")
            st.bar_chart(timeline, horizontal=True, x_label="minutes after T", y_label="",
                         color="#F2A541", height=max(160, 26 * len(timeline)))


# --------------------------------------------------------------------------- #
# 5. Page composition
# --------------------------------------------------------------------------- #
def main() -> None:
    """Compose the dashboard page. Called when Streamlit executes this file."""
    st.set_page_config(
        page_title="Air-threat cascade engine",
        page_icon=":material/radar:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Air-threat cascade engine", icon=":material/radar:")
    st.caption(
        "Evaluation console · every view is computed from alarms that started strictly before the "
        "cutoff T · all times UTC"
    )

    with st.sidebar:
        st.header("Controls", icon=":material/tune:")
        selection = render_source_controls()
        source_used, fallback_reason = resolve_source(selection.source, selection.csv_path)
        try:
            history = load_history(source_used, selection.csv_path, selection.num_records,
                                   selection.seed, selection.start_date)
        except (SchemaValidationError, FileNotFoundError, ValueError) as exc:
            st.error(f"Could not load alarm data: {exc}", icon=":material/error:")
            st.stop()
        signature = "|".join([source_used, selection.csv_path, str(selection.num_records),
                              str(selection.seed), selection.start_date, str(len(history))])
        cutoff = render_cutoff_controls(history, signature)
        waves, threat_types = render_threat_filters(history)
        params = render_cascade_controls()

    try:
        outcome = load_data(selection, cutoff, history, source_used, fallback_reason)
    except (SchemaValidationError, TemporalLeakageError, FileNotFoundError, ValueError) as exc:
        st.error(f"Temporal safeguard failed: {exc}", icon=":material/gpp_bad:")
        st.stop()

    visible = apply_threat_filters(outcome.frame, waves, threat_types)
    active = active_alarms(visible, cutoff)
    render_status_banner(outcome, cutoff, history, len(visible))

    tab_overview, tab_graph, tab_sim = st.tabs(
        [":material/dashboard: Overview & metrics",
         ":material/hub: Spatiotemporal graph",
         ":material/model_training: Cascade simulator"]
    )
    if visible.empty:
        reason = ("No alarms recorded before T" if outcome.frame.empty
                  else "The threat filter hides every alarm before T")
        for slot, tab in (("overview", tab_overview), ("graph", tab_graph), ("simulator", tab_sim)):
            with tab:
                render_empty_state(cutoff, history, reason=reason, slot=slot)
        return

    graph = build_propagation_graph(visible, max_lag_min=params.max_lag_min)
    regions = [r for r in UKRAINIAN_REGIONS if r in set(history["region"])] or sorted(history["region"].unique())
    with tab_overview:
        render_overview_tab(visible, active, cutoff)
    with tab_graph:
        render_graph_tab(graph, active, params)
    with tab_sim:
        render_simulator_tab(graph, active, cutoff, params, regions)


if __name__ == "__main__":
    main()
