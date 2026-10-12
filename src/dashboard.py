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

from src.config import get_settings  # noqa: E402
from src.llm_parser import GeminiOSINTParser, demo_messages  # noqa: E402
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
        max_lag_min: Slow-wave lag window. Two alarms of the same threat type
            form a directed edge when the second starts within this many
            minutes of the first. Also used for the mixed-wave simulator graph.
        fast_lag_min: Lag window for the fast-wave layer (ballistic / aviation
            alarms cascade within minutes, so a tight window avoids spurious
            long-range edges).
        min_edge_weight: Edges observed fewer times than this are hidden in
            the graph view (they still contribute to predictions).
        decay_min: E-folding time (minutes) by which a trigger's influence
            fades once its typical lag to a target has already elapsed.
        hops: Propagation depth considered by the simulator (1 or 2).
    """

    max_lag_min: float = 90.0
    fast_lag_min: float = 30.0
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
        triggers: ``(region, elapsed_minutes)`` pairs active at ``T``, or
            ``(region, elapsed_minutes, weight)`` triples where ``weight`` in
            ``[0, 1]`` scales the trigger's contributions (used for the
            confidence of OSINT-derived triggers).
        cutoff: Observation instant ``T``.
        top_k: Number of rows to return.
        hops: Propagation depth, ``1`` or ``2``.
        decay_min: Influence decay constant in minutes.

    Returns:
        Frame with :data:`PREDICTION_COLUMNS`, best candidates first.
    """
    trigger_regions = {str(item[0]) for item in triggers}
    paths: dict[str, list[tuple[float, float, str, int]]] = {}

    def successors(region: str) -> list[tuple[str, float, float]]:
        return [
            (succ, float(data["probability"]), float(data["median_lag_min"]))
            for succ, data in graph[region].items()
        ]

    for item in triggers:
        region, elapsed = str(item[0]), float(item[1])
        weight = min(1.0, max(0.0, float(item[2]))) if len(item) > 2 else 1.0
        if region not in graph or weight <= 0.0:
            continue
        for succ, p1, lag1 in successors(region):
            if succ in trigger_regions:
                continue
            overdue = max(0.0, elapsed - lag1)
            p = weight * p1 * math.exp(-overdue / max(decay_min, 1e-9))
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


# --------------------------------------------------------------------------- #
# 3c. Wave map: per-wave layers, default topology fallback, activity sizing
# --------------------------------------------------------------------------- #
WAVE_SPEED_KMH: Final[Mapping[str, float]] = {"fast": 900.0, "slow": 150.0}
"""Nominal propagation speed per wave class used for prior lag estimates."""

WAVE_PRIOR_PROBABILITY: Final[Mapping[str, float]] = {"fast": 0.35, "slow": 0.45}
"""Transition probability assigned to adjacency edges in the prior topology."""

# Undirected geographic adjacency between oblasts (Kyiv City sits inside Kyiv
# Oblast). Used only to build the fallback topology when no data exists.
_ADJACENT_PAIRS: Final[Sequence[tuple[str, str]]] = (
    ("Volyn Oblast", "Rivne Oblast"), ("Volyn Oblast", "Lviv Oblast"),
    ("Rivne Oblast", "Lviv Oblast"), ("Rivne Oblast", "Ternopil Oblast"),
    ("Rivne Oblast", "Khmelnytskyi Oblast"), ("Rivne Oblast", "Zhytomyr Oblast"),
    ("Lviv Oblast", "Ternopil Oblast"), ("Lviv Oblast", "Ivano-Frankivsk Oblast"),
    ("Lviv Oblast", "Zakarpattia Oblast"), ("Zakarpattia Oblast", "Ivano-Frankivsk Oblast"),
    ("Ivano-Frankivsk Oblast", "Ternopil Oblast"), ("Ivano-Frankivsk Oblast", "Chernivtsi Oblast"),
    ("Ternopil Oblast", "Khmelnytskyi Oblast"), ("Ternopil Oblast", "Chernivtsi Oblast"),
    ("Chernivtsi Oblast", "Khmelnytskyi Oblast"), ("Chernivtsi Oblast", "Vinnytsia Oblast"),
    ("Khmelnytskyi Oblast", "Zhytomyr Oblast"), ("Khmelnytskyi Oblast", "Vinnytsia Oblast"),
    ("Zhytomyr Oblast", "Vinnytsia Oblast"), ("Zhytomyr Oblast", "Kyiv Oblast"),
    ("Vinnytsia Oblast", "Kyiv Oblast"), ("Vinnytsia Oblast", "Cherkasy Oblast"),
    ("Vinnytsia Oblast", "Kirovohrad Oblast"), ("Vinnytsia Oblast", "Odesa Oblast"),
    ("Kyiv Oblast", "Kyiv City"), ("Kyiv Oblast", "Chernihiv Oblast"),
    ("Kyiv Oblast", "Poltava Oblast"), ("Kyiv Oblast", "Cherkasy Oblast"),
    ("Chernihiv Oblast", "Sumy Oblast"), ("Chernihiv Oblast", "Poltava Oblast"),
    ("Sumy Oblast", "Poltava Oblast"), ("Sumy Oblast", "Kharkiv Oblast"),
    ("Poltava Oblast", "Kharkiv Oblast"), ("Poltava Oblast", "Dnipropetrovsk Oblast"),
    ("Poltava Oblast", "Kirovohrad Oblast"), ("Poltava Oblast", "Cherkasy Oblast"),
    ("Cherkasy Oblast", "Kirovohrad Oblast"), ("Kirovohrad Oblast", "Dnipropetrovsk Oblast"),
    ("Kirovohrad Oblast", "Mykolaiv Oblast"), ("Kirovohrad Oblast", "Odesa Oblast"),
    ("Kharkiv Oblast", "Dnipropetrovsk Oblast"), ("Kharkiv Oblast", "Donetsk Oblast"),
    ("Kharkiv Oblast", "Luhansk Oblast"), ("Luhansk Oblast", "Donetsk Oblast"),
    ("Donetsk Oblast", "Dnipropetrovsk Oblast"), ("Donetsk Oblast", "Zaporizhzhia Oblast"),
    ("Dnipropetrovsk Oblast", "Zaporizhzhia Oblast"), ("Dnipropetrovsk Oblast", "Kherson Oblast"),
    ("Dnipropetrovsk Oblast", "Mykolaiv Oblast"), ("Zaporizhzhia Oblast", "Kherson Oblast"),
    ("Kherson Oblast", "Mykolaiv Oblast"), ("Mykolaiv Oblast", "Odesa Oblast"),
)


def _build_adjacency(pairs: Sequence[tuple[str, str]]) -> dict[str, tuple[str, ...]]:
    adjacency: dict[str, list[str]] = {region: [] for region in UKRAINIAN_REGIONS}
    for a, b in pairs:
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)
    return {region: tuple(sorted(set(neighbours))) for region, neighbours in adjacency.items()}


REGION_ADJACENCY: Final[Mapping[str, tuple[str, ...]]] = _build_adjacency(_ADJACENT_PAIRS)
"""Symmetric oblast adjacency derived from :data:`_ADJACENT_PAIRS`."""


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance between two ``(lon, lat)`` points in kilometres."""
    lon1, lat1 = math.radians(a[0]), math.radians(a[1])
    lon2, lat2 = math.radians(b[0]), math.radians(b[1])
    d_lat = lat2 - lat1
    d_lon = lon2 - lon1
    h = math.sin(d_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def default_topology_graph(wave: str) -> nx.DiGraph:
    """Prior propagation graph built from oblast adjacency.

    Used when no observed propagation exists before ``T`` (for example when
    ``T`` precedes the data, the CSV is missing, or the threat filter hides
    everything). Every adjacency pair becomes two directed edges with a
    distance-derived lag (``distance / WAVE_SPEED_KMH``) and the wave's prior
    transition probability. Edges carry ``prior=True`` and the graph carries
    ``graph.graph["source"] == "prior"`` so the UI can label them honestly.

    Args:
        wave: ``"fast"`` or ``"slow"`` (anything else uses mid values).

    Returns:
        A :class:`networkx.DiGraph` over all :data:`UKRAINIAN_REGIONS`.
    """
    speed = WAVE_SPEED_KMH.get(wave, 300.0)
    probability = WAVE_PRIOR_PROBABILITY.get(wave, 0.3)
    graph = nx.DiGraph(wave=wave, source="prior", lag_window_min=None)
    for region in UKRAINIAN_REGIONS:
        graph.add_node(region, alarms=0)
    for a, b in _ADJACENT_PAIRS:
        lag = max(2.0, haversine_km(REGION_COORDS[a], REGION_COORDS[b]) / speed * 60.0)
        for u, v in ((a, b), (b, a)):
            graph.add_edge(u, v, weight=1, median_lag_min=round(lag, 1), mean_lag_min=round(lag, 1),
                           probability=probability, prior=True)
    return graph


def with_fallback_topology(graph: nx.DiGraph, wave: str) -> nx.DiGraph:
    """Return ``graph`` if it has edges, else the prior topology for ``wave``.

    Observed node alarm counts are preserved on the prior graph so node sizing
    still reflects whatever activity exists.

    Args:
        graph: Observed propagation graph (possibly edgeless).
        wave: Wave class for the prior.

    Returns:
        A graph that always has edges; ``graph.graph["source"]`` is
        ``"observed"`` or ``"prior"``.
    """
    if graph.number_of_edges() > 0:
        graph.graph.setdefault("source", "observed")
        graph.graph.setdefault("wave", wave)
        return graph
    prior = default_topology_graph(wave)
    for node, data in graph.nodes(data=True):
        if node in prior:
            prior.nodes[node]["alarms"] = int(data.get("alarms", 0))
        else:
            prior.add_node(node, alarms=int(data.get("alarms", 0)))
    prior.graph["lag_window_min"] = graph.graph.get("lag_window_min")
    return prior


def build_wave_graphs(df: pd.DataFrame, params: CascadeParams) -> dict[str, nx.DiGraph]:
    """Build one propagation graph per wave class, each with a safe fallback.

    Args:
        df: Cutoff-filtered (and threat-filtered) alarm frame; may be empty.
        params: Lag windows per wave come from here.

    Returns:
        ``{"fast": graph, "slow": graph}``. Each graph has edges: observed
        ones when the data supports them, otherwise the prior topology.
    """
    graphs: dict[str, nx.DiGraph] = {}
    for wave, lag in ((ThreatWave.FAST.value, params.fast_lag_min),
                      (ThreatWave.SLOW.value, params.max_lag_min)):
        subset = df.loc[df["wave_class"] == wave] if not df.empty else df
        observed = build_propagation_graph(subset, max_lag_min=lag)
        observed.graph.update(wave=wave, source="observed", lag_window_min=lag)
        graphs[wave] = with_fallback_topology(observed, wave)
    return graphs


def node_activity(df: pd.DataFrame, cutoff: pd.Timestamp, *,
                  window_hours: Optional[float]) -> pd.Series:
    """Alarm onsets per region in the trailing window before ``T``.

    Args:
        df: Cutoff-filtered alarm frame.
        cutoff: Observation instant ``T``.
        window_hours: Trailing window; ``None`` counts all history before ``T``.

    Returns:
        Integer series indexed by region, descending, named ``alarms``.
    """
    if df.empty:
        return pd.Series(dtype="int64", name="alarms")
    frame = df if window_hours is None else df.loc[
        df["start_time"] >= cutoff - pd.Timedelta(hours=window_hours)
    ]
    return frame["region"].value_counts().rename("alarms")


def render_wave_map_figure(
    graphs: Mapping[str, nx.DiGraph],
    *,
    layers: Sequence[str],
    activity: pd.Series,
    active_regions: Sequence[str] = (),
    trigger: Optional[str] = None,
    predictions: Optional[pd.DataFrame] = None,
    min_probability: float = 0.0,
    max_edges_per_layer: int = 60,
    show_edge_labels: bool = True,
    max_labels: int = 18,
    show_node_labels: bool = True,
    title: str = "",
    dark: bool = False,
) -> Figure:
    """Draw fast- and slow-wave propagation layers on one geographic map.

    Nodes are oblasts sized by ``activity`` (alarm frequency before ``T``).
    Each layer draws its strongest edges in the wave colour, width scaled by
    transition probability and opacity by observation count; prior-topology
    edges are dashed. Edge labels show ``probability · lag``. Regions active
    at ``T`` are filled red, the selected trigger gets a star, and predicted
    targets are orange with their arrival estimate annotated.

    Args:
        graphs: Per-wave graphs from :func:`build_wave_graphs`.
        layers: Wave classes to draw (``"fast"``, ``"slow"``).
        activity: Region -> alarm count used for node sizing.
        active_regions: Regions with an alarm active at ``T``.
        trigger: Region selected as the simulation trigger.
        predictions: Output of :func:`predict_cascade` for the trigger.
        min_probability: Hide edges below this transition probability.
        max_edges_per_layer: Strongest edges kept per layer.
        show_edge_labels: Annotate the strongest edges.
        max_labels: Cap on annotated edges across layers.
        show_node_labels: Draw region names.
        title: Figure title.
        dark: Use light text for dark app themes.

    Returns:
        A transparent-background matplotlib figure owned by the caller.
    """
    text_color = "#E6E6E6" if dark else "#222222"
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    ax.set_axis_off()
    if title:
        ax.set_title(title, color=text_color, fontsize=12, loc="left")

    union = nx.DiGraph()
    for graph in graphs.values():
        union.add_nodes_from(graph.nodes)
    for region in UKRAINIAN_REGIONS:
        union.add_node(region)
    pos = layout_positions(union)
    nodes = list(union.nodes)

    counts = np.asarray([float(activity.get(n, 0.0)) if len(activity) else 0.0 for n in nodes])
    peak = counts.max() if counts.size and counts.max() > 0 else 1.0
    sizes = 110 + 1100 * (counts / peak)

    predicted: dict[str, tuple[float, float]] = {}
    if predictions is not None and not predictions.empty:
        predicted = {str(r.region): (float(r.probability), float(r.eta_min))
                     for r in predictions.itertuples(index=False)}
    active_set = set(active_regions)
    fill, alpha = [], []
    for node in nodes:
        if node == trigger:
            fill.append("#FFD166"); alpha.append(1.0)
        elif node in predicted:
            fill.append("#F2A541"); alpha.append(0.45 + 0.55 * predicted[node][0])
        elif node in active_set:
            fill.append("#D1495B"); alpha.append(0.95)
        else:
            fill.append("#B8C2CC"); alpha.append(0.85)

    label_budget = max_labels
    for wave in layers:
        graph = graphs.get(wave)
        if graph is None or graph.number_of_edges() == 0:
            continue
        edges = sorted(
            (e for e in graph.edges(data=True) if e[2].get("probability", 0.0) >= min_probability),
            key=lambda e: (e[2].get("weight", 1), e[2].get("probability", 0.0)),
            reverse=True,
        )[:max_edges_per_layer]
        if not edges:
            continue
        weights = np.asarray([e[2].get("weight", 1) for e in edges], dtype=float)
        probs = np.asarray([e[2].get("probability", 0.0) for e in edges], dtype=float)
        prior = graph.graph.get("source") == "prior"
        style = "dashed" if prior else "solid"
        rad = 0.08 if wave == "fast" else 0.22
        nx.draw_networkx_edges(
            graph, pos, edgelist=[(u, v) for u, v, _ in edges],
            width=list(0.6 + 3.4 * probs),
            alpha=0.25 if prior else None,
            edge_color=[(*matplotlib.colors.to_rgb(WAVE_COLORS[wave]),
                         0.3 + 0.6 * w / weights.max()) for w in weights] if not prior else WAVE_COLORS[wave],
            style=style, arrows=True, arrowstyle="-|>", arrowsize=10,
            connectionstyle=f"arc3,rad={rad}", node_size=list(sizes), ax=ax,
        )
        if show_edge_labels and label_budget > 0:
            labelled = edges[:label_budget]
            label_budget -= len(labelled)
            nx.draw_networkx_edge_labels(
                graph, pos,
                edge_labels={(u, v): f"{d['probability']:.0%}·{d['median_lag_min']:.0f}m"
                             for u, v, d in labelled},
                font_size=6.5, font_color=WAVE_COLORS[wave], label_pos=0.5,
                bbox={"alpha": 0.0}, connectionstyle=f"arc3,rad={rad}", ax=ax,
            )

    if trigger and predicted:
        route_graph = nx.DiGraph()
        route_graph.add_edges_from((trigger, target) for target in predicted if trigger in pos)
        nx.draw_networkx_edges(
            route_graph, pos, width=[1.5 + 3.0 * predicted[v][0] for _, v in route_graph.edges],
            edge_color="#F2A541", alpha=0.9, arrows=True, arrowstyle="-|>", arrowsize=14,
            connectionstyle="arc3,rad=0.12", node_size=list(sizes), ax=ax,
        )
        for target, (_, eta) in predicted.items():
            if target in pos:
                x, y = pos[target]
                ax.annotate(f"+{eta:.0f} min", (x, y), xytext=(0, 11), textcoords="offset points",
                            ha="center", fontsize=7.5, color="#B86B00", fontweight="bold")

    nx.draw_networkx_nodes(union, pos, nodelist=nodes, node_size=list(sizes), node_color=fill,
                           alpha=alpha, linewidths=0.8, edgecolors=text_color, ax=ax)
    if trigger and trigger in pos:
        ax.scatter([pos[trigger][0]], [pos[trigger][1]], marker="*", s=420, color="#FFD166",
                   edgecolors="#8A5A00", linewidths=0.9, zorder=5)
    if show_node_labels:
        nx.draw_networkx_labels(union, pos, labels={n: short_region_name(n) for n in nodes},
                                font_size=7.5, font_color=text_color, ax=ax)
    ax.margins(0.08)

    handles = [
        Line2D([0], [0], color=WAVE_COLORS["fast"], lw=3, label="Fast-wave edge (prob · lag)"),
        Line2D([0], [0], color=WAVE_COLORS["slow"], lw=3, label="Slow-wave edge (prob · lag)"),
        Line2D([0], [0], color="#777777", lw=2, linestyle="dashed", label="Prior topology (no data)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#B8C2CC", markersize=9,
               label="Region (size = alarm frequency before T)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#D1495B", markersize=9,
               label="Active at T"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#FFD166", markersize=13,
               label="Selected trigger"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#F2A541", markersize=9,
               label="Predicted target (+ETA)"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=7.5, frameon=False, labelcolor=text_color,
              ncol=2)
    return fig


def layer_summary(graphs: Mapping[str, nx.DiGraph]) -> pd.DataFrame:
    """One row per wave layer: node/edge counts, mean probability, median lag, source."""
    rows = []
    for wave, graph in graphs.items():
        probs = [d.get("probability", 0.0) for _, _, d in graph.edges(data=True)]
        lags = [d.get("median_lag_min", 0.0) for _, _, d in graph.edges(data=True)]
        rows.append({
            "layer": WAVE_LABELS.get(wave, wave),
            "source": "observed" if graph.graph.get("source") == "observed" else "prior topology",
            "lag_window_min": graph.graph.get("lag_window_min"),
            "regions": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "mean_probability": float(np.mean(probs)) if probs else 0.0,
            "median_lag_min": float(np.median(lags)) if lags else 0.0,
        })
    return pd.DataFrame(rows)


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
        max_lag = st.slider("Slow-wave lag window (min)", 15, 240, 90, step=15, key="max_lag",
                            help="UAV / Shahed alarms: window for A → B co-occurrence edges.")
        fast_lag = st.slider("Fast-wave lag window (min)", 5, 90, 30, step=5, key="fast_lag",
                             help="Ballistic / aviation alarms: window for A → B edges.")
        min_weight = st.slider("Min edge count shown", 1, 15, 2, key="min_edge_weight")
        decay = st.slider("Trigger influence decay (min)", 15, 240, 60, step=15, key="decay_min")
        hops = st.segmented_control(
            "Propagation depth", options=[1, 2], default=1, required=True,
            format_func=lambda h: f"{h} hop" if h == 1 else f"{h} hops", key="hops",
        )
    return CascadeParams(
        max_lag_min=float(max_lag),
        fast_lag_min=float(fast_lag),
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


_ACTIVITY_WINDOWS: Final[Mapping[str, Optional[float]]] = {
    "6 h": 6.0, "24 h": 24.0, "72 h": 72.0, "All before T": None,
}
_WHAT_IF_OPTION: Final[str] = "What-if: choose any region"


def _trigger_label(row: Any) -> str:
    return f"{row.region} · {str(row.threat_type).replace('_', ' ')} · +{row.elapsed_min:.0f} min"


def render_wave_map_tab(
    graphs: Mapping[str, nx.DiGraph],
    visible: pd.DataFrame,
    active: pd.DataFrame,
    cutoff: pd.Timestamp,
    params: CascadeParams,
    *,
    empty_reason: Optional[str] = None,
) -> None:
    """Tab 2: layered wave map with an interactive single-trigger simulation.

    Always renders: when no observed propagation exists for a layer the prior
    adjacency topology is drawn (dashed) and labelled as such, and when no
    alarm is active at ``T`` the trigger selector falls back to what-if mode.

    Args:
        graphs: Per-wave graphs from :func:`build_wave_graphs`.
        visible: Filtered, cutoff-safe alarm frame (may be empty).
        active: Alarms active at ``T`` (may be empty).
        cutoff: Observation instant ``T``.
        params: Cascade parameters from the sidebar.
        empty_reason: Explanation shown when ``visible`` is empty.
    """
    st.caption(
        "Directed edges A → B: empirical transition probability P(B | A) and median propagation "
        f"lag. Fast-wave window {params.fast_lag_min:.0f} min, slow-wave window "
        f"{params.max_lag_min:.0f} min. Nodes are oblasts sized by alarm frequency before T; "
        "layout uses oblast-centre coordinates. Only pre-T alarms feed the layers."
    )
    if empty_reason:
        st.info(f"{empty_reason}. The map shows the default adjacency topology so the model can "
                "still be explored.", icon=":material/map:")
    prior_layers = [WAVE_LABELS[w] for w, g in graphs.items() if g.graph.get("source") == "prior"]
    if prior_layers:
        st.warning(
            "No observed propagation before T for: " + "; ".join(prior_layers) + ". Those layers "
            "show the default topology (dashed edges) with prior probabilities and distance-based lags.",
            icon=":material/alt_route:", title="Default topology in use",
        )

    with st.container(horizontal=True, vertical_alignment="bottom", gap="medium"):
        chosen_layers = st.pills(
            "Layers", options=["fast", "slow"], default=["fast", "slow"], selection_mode="multi",
            format_func=lambda w: "Fast-wave" if w == "fast" else "Slow-wave", key="wm_layers",
        )
        window_label = st.segmented_control(
            "Node size: activity window", options=list(_ACTIVITY_WINDOWS), default="24 h",
            required=True, key="wm_activity_window",
        )
        min_probability = st.slider("Min transition probability", 0.0, 0.9, 0.1, step=0.05,
                                    key="wm_min_prob", format="%.2f")
        max_edges = st.slider("Edges per layer", 10, 150, 50, step=10, key="wm_max_edges")
    with st.container(horizontal=True, gap="medium"):
        edge_labels = st.toggle("Edge labels (prob · lag)", value=True, key="wm_edge_labels")
        node_labels = st.toggle("Region labels", value=True, key="wm_node_labels")

    layers = list(chosen_layers) if isinstance(chosen_layers, list) else []
    activity = node_activity(visible, cutoff, window_hours=_ACTIVITY_WINDOWS.get(str(window_label), 24.0))

    # ---- trigger selection ------------------------------------------------ #
    with st.container(border=True):
        st.subheader("Propagation wave from a trigger", icon=":material/waves:")
        trigger_rows = active.sort_values("elapsed_min") if not active.empty else active
        options = [_trigger_label(r) for r in trigger_rows.itertuples(index=False)] + [_WHAT_IF_OPTION]
        with st.container(horizontal=True, vertical_alignment="bottom", gap="medium"):
            choice = st.selectbox("Active trigger at T", options, index=0, key="wm_trigger",
                                  help="Alarms active at T, most recent first; or pick any region.")
            top_k = st.slider("Top-K targets", 3, 12, 6, key="wm_top_k")
        if choice == _WHAT_IF_OPTION or trigger_rows.empty:
            if trigger_rows.empty:
                st.caption("No alarm is active at T, so the simulation runs in what-if mode.")
            with st.container(horizontal=True, vertical_alignment="bottom", gap="medium"):
                trigger = st.selectbox("Trigger region", list(UKRAINIAN_REGIONS), key="wm_whatif_region")
                wave = st.segmented_control("Wave", options=["fast", "slow"], default="slow",
                                            required=True, key="wm_whatif_wave",
                                            format_func=lambda w: "Fast-wave" if w == "fast" else "Slow-wave")
                elapsed = float(st.slider("Minutes since onset", 0, 180, 0, step=5, key="wm_whatif_elapsed"))
            wave = str(wave) if wave in ("fast", "slow") else "slow"
        else:
            row = trigger_rows.iloc[options.index(choice)]
            trigger = str(row["region"])
            wave = str(row["wave_class"]) if row["wave_class"] in ("fast", "slow") else "slow"
            elapsed = float(row["elapsed_min"])

        graph = graphs[wave]
        predictions = predict_cascade(graph, [(trigger, elapsed)], cutoff, top_k=top_k,
                                      hops=params.hops, decay_min=params.decay_min)
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            st.badge(f"Trigger: {trigger}", icon=":material/my_location:", color="orange")
            st.badge("Fast-wave" if wave == "fast" else "Slow-wave", icon=":material/speed:",
                     color="red" if wave == "fast" else "blue")
            st.badge(f"+{elapsed:.0f} min since onset", icon=":material/timer:", color="gray")
            st.badge("observed graph" if graph.graph.get("source") == "observed" else "prior topology",
                     icon=":material/hub:", color="green" if graph.graph.get("source") == "observed" else "violet")

    fig = render_wave_map_figure(
        graphs, layers=layers, activity=activity, active_regions=list(active["region"].unique()),
        trigger=trigger, predictions=predictions, min_probability=float(min_probability),
        max_edges_per_layer=int(max_edges), show_edge_labels=bool(edge_labels),
        show_node_labels=bool(node_labels),
        title=f"Wave map at T = {cutoff:%Y-%m-%d %H:%M} UTC", dark=_theme_is_dark(),
    )
    st.pyplot(fig)
    plt.close(fig)

    col_pred, col_layers = st.columns((3, 2))
    with col_pred, st.container(border=True):
        st.subheader(f"Top-{top_k} predicted targets", icon=":material/trending_up:")
        if predictions.empty:
            st.info("The trigger has no outgoing edges in this layer. Lower the minimum probability, "
                    "widen the lag window, or choose two hops.", icon=":material/hub:")
        else:
            st.dataframe(
                predictions[["region", "probability", "eta_min", "eta_time", "hops", "paths"]],
                hide_index=True,
                column_config={
                    "region": "Region",
                    "probability": st.column_config.ProgressColumn("Probability", min_value=0,
                                                                   max_value=1, format="percent"),
                    "eta_min": st.column_config.NumberColumn("ETA (min)", format="%.0f"),
                    "eta_time": st.column_config.DatetimeColumn("Est. arrival (UTC)", format="HH:mm"),
                    "hops": st.column_config.NumberColumn("Hops", format="%d"),
                    "paths": st.column_config.NumberColumn("Paths", format="%d"),
                },
            )
    with col_layers, st.container(border=True):
        st.subheader("Layer summary", icon=":material/layers:")
        st.dataframe(
            layer_summary(graphs), hide_index=True,
            column_config={
                "layer": "Layer", "source": "Source",
                "lag_window_min": st.column_config.NumberColumn("Window (min)", format="%.0f"),
                "regions": st.column_config.NumberColumn("Regions", format="%d"),
                "edges": st.column_config.NumberColumn("Edges", format="%d"),
                "mean_probability": st.column_config.NumberColumn("Mean P", format="percent"),
                "median_lag_min": st.column_config.NumberColumn("Median lag (min)", format="%.0f"),
            },
        )

    with st.expander("Corridors and roles by layer", icon=":material/alt_route:"):
        for wave_key in ("fast", "slow"):
            layer_graph = graphs[wave_key]
            st.markdown(f"**{WAVE_LABELS[wave_key]}** · "
                        f"{'observed' if layer_graph.graph.get('source') == 'observed' else 'prior topology'}")
            corridors = edge_table(layer_graph, min_weight=1 if layer_graph.graph.get("source") == "prior"
                                   else params.min_edge_weight, limit=12)
            sources, sinks = node_roles(layer_graph, top_n=5)
            c1, c2 = st.columns((3, 2))
            with c1:
                st.dataframe(
                    corridors, hide_index=True,
                    column_config={
                        "source": "From", "target": "To",
                        "co_occurrences": st.column_config.NumberColumn("Count", format="%d"),
                        "median_lag_min": st.column_config.NumberColumn("Lag (min)", format="%.0f"),
                        "probability": st.column_config.ProgressColumn("P(to | from)", min_value=0,
                                                                       max_value=1, format="percent"),
                    },
                )
            with c2:
                roles = sources.rename(columns={"weighted_degree": "out"}).merge(
                    sinks.rename(columns={"weighted_degree": "in"}), on="region", how="outer").fillna(0)
                st.dataframe(roles, hide_index=True,
                             column_config={"region": "Region",
                                            "out": st.column_config.NumberColumn("Out", format="%d"),
                                            "in": st.column_config.NumberColumn("In", format="%d")})


_TRIGGER_MODES: Final[Mapping[str, str]] = {
    "observed": "Observed alarms at T",
    "whatif": "What-if regions",
    "osint": "OSINT message",
}


@st.cache_resource(show_spinner=False)
def get_osint_parser() -> GeminiOSINTParser:
    """Process-wide OSINT parser (keeps the Gemini client and circuit breaker).

    Returns:
        A :class:`GeminiOSINTParser` built from the current settings; it is
        offline automatically when no API key is configured.
    """
    return GeminiOSINTParser()


def _run_osint_pipeline() -> None:
    """Button callback: parse the pasted message and switch the simulator to it.

    Runs before the script reruns, so it may update the trigger-mode widget's
    session state. Never raises: parser failures degrade to rules, and any
    unexpected error is stored for display.
    """
    text = str(st.session_state.get("osint_text", "")).strip()
    st.session_state["osint_error"] = None
    if not text:
        st.session_state["osint_result"] = None
        st.session_state["osint_error"] = "Paste a monitoring message first."
        return
    try:
        parser = get_osint_parser()
        record = parser.parse(text)
        st.session_state["osint_result"] = record.model_dump(mode="json")
        st.session_state["osint_status"] = parser.status()
        st.session_state["sim_trigger_mode"] = "osint"
    except Exception as exc:  # noqa: BLE001 - UI must never crash on parser issues
        logger.exception("OSINT pipeline failed")
        st.session_state["osint_result"] = None
        st.session_state["osint_error"] = f"{type(exc).__name__}: {exc}"


def _load_osint_example() -> None:
    """Button callback: copy the selected demo message into the text area."""
    st.session_state["osint_text"] = st.session_state.get("osint_example", demo_messages()[0])


def osint_triggers(record: Mapping[str, Any], *, include_source: bool = True) -> pd.DataFrame:
    """Turn a parsed OSINT record into simulator triggers active at ``T``.

    Projected ``target_regions`` become triggers (the message says they are
    being, or about to be, alarmed). The ``source_region`` is added when it is
    a Ukrainian oblast, since the threat is currently there. External origins
    (Belgorod, Black Sea, ...) are not nodes and are skipped. Every trigger
    starts at ``T`` (elapsed 0) and is weighted by the record's confidence.

    Args:
        record: ``ParsedThreat.model_dump()`` payload.
        include_source: Add the source oblast as a trigger.

    Returns:
        Columns ``region``, ``elapsed_min``, ``weight``, ``role``.
    """
    confidence = float(record.get("confidence_score", 0.0) or 0.0)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    source = record.get("source_region")
    if include_source and source in UKRAINIAN_REGIONS:
        rows.append({"region": source, "elapsed_min": 0.0, "weight": confidence, "role": "source"})
        seen.add(str(source))
    for target in record.get("target_regions") or []:
        if target in UKRAINIAN_REGIONS and target not in seen:
            rows.append({"region": target, "elapsed_min": 0.0, "weight": confidence, "role": "target"})
            seen.add(str(target))
    return pd.DataFrame(rows, columns=["region", "elapsed_min", "weight", "role"])


def render_osint_input() -> Optional[dict[str, Any]]:
    """Simulator block: paste a Telegram message, parse it, inspect the record.

    Returns:
        The parsed record payload stored in session state, or ``None``.
    """
    with st.container(border=True):
        st.subheader("Multimodal OSINT input", icon=":material/translate:")
        st.caption(
            "Paste a raw monitoring-channel update (Ukrainian / Russian / English slang is fine). "
            "Gemini normalises it into threat type, source and target oblasts with a confidence "
            "score; without an API key the rule-based parser runs instead. Parsed regions become "
            "triggers active at T."
        )
        with st.container(horizontal=True, vertical_alignment="bottom", gap="small"):
            st.selectbox("Example message", demo_messages(), key="osint_example",
                         format_func=lambda m: m[:70] + ("…" if len(m) > 70 else ""))
            st.button("Use example", icon=":material/content_paste:", on_click=_load_osint_example,
                      key="osint_use_example")
        st.text_area(
            "Raw monitoring message", key="osint_text", height=110,
            placeholder="⚠️ Сумщина: 6 мопедів курсом на Полтавщину, швидкість ~150 км/год",
        )
        st.button("⚡ Parse OSINT & Run Cascade Simulation", type="primary", key="osint_run",
                  on_click=_run_osint_pipeline, width="stretch")

        error = st.session_state.get("osint_error")
        if error:
            st.warning(error, icon=":material/warning:")
        record = st.session_state.get("osint_result")
        status = st.session_state.get("osint_status") or {}
        if record:
            engine = str(record.get("parser", "rules"))
            with st.container(horizontal=True, vertical_alignment="center", gap="small"):
                st.badge(str(record.get("threat_type", "unknown")).replace("_", " "),
                         icon=":material/warning:", color="red" if record.get("wave_class") == "fast"
                         else "blue" if record.get("wave_class") == "slow" else "gray")
                st.badge(f"{record.get('wave_class', 'unknown')}-wave", icon=":material/speed:",
                         color="gray")
                st.badge(f"source: {record.get('source_region') or '—'}", icon=":material/flight_takeoff:",
                         color="orange")
                st.badge(f"targets: {', '.join(record.get('target_regions') or []) or '—'}",
                         icon=":material/my_location:", color="violet")
                st.badge(f"confidence {float(record.get('confidence_score', 0)):.0%}",
                         icon=":material/percent:", color="green")
                st.badge({"gemini": "Gemini", "rules": "rule-based fallback",
                          "synthetic": "synthetic scenario"}.get(engine, engine),
                         icon=":material/smart_toy:" if engine == "gemini" else ":material/rule:",
                         color="green" if engine == "gemini" else "orange")
            st.caption(record.get("raw_summary", ""))
            if record.get("unmapped_regions"):
                st.caption("Unmapped mentions: " + ", ".join(record["unmapped_regions"]))
            if status.get("offline_reason"):
                st.caption(f"Parser offline: {status['offline_reason']}")
            with st.expander("Parsed OSINT record (structured JSON)", icon=":material/data_object:",
                             expanded=True):
                st.json(record, expanded=2)
        return record if isinstance(record, dict) else None


def render_simulator_tab(graph: nx.DiGraph, active: pd.DataFrame, cutoff: pd.Timestamp,
                         params: CascadeParams, all_regions: Sequence[str],
                         wave_graphs: Optional[Mapping[str, nx.DiGraph]] = None) -> None:
    """Tab 3: cascade simulator with OSINT ingestion, top-K targets and ETAs.

    Trigger sources: alarms observed active at ``T``, hand-picked what-if
    regions, or a parsed OSINT message whose source / target oblasts are
    injected as triggers at ``T`` weighted by the parser's confidence.

    Args:
        graph: Mixed-wave propagation graph from the visible data.
        active: Alarms active at ``T`` (default triggers).
        cutoff: Observation instant ``T``.
        params: Cascade parameters from the sidebar.
        all_regions: Regions offered for manual what-if triggers.
        wave_graphs: Optional per-wave graphs; when an OSINT record has a known
            wave class its layer is used so the propagation view matches it.
    """
    st.caption(
        "Each successor's chance is the empirical P(target | trigger) discounted once the typical "
        "lag has elapsed; multiple triggers combine with a noisy-OR. Arrival estimates are median "
        "historical lags measured from each trigger's start."
    )
    record = render_osint_input()

    # Session state drives the widget (no ``default``) so the parse-button
    # callback may switch the mode to "osint" without a Streamlit warning.
    st.session_state.setdefault("sim_trigger_mode", "observed")
    with st.container(horizontal=True, vertical_alignment="bottom", gap="medium"):
        mode = st.segmented_control(
            "Trigger source", options=list(_TRIGGER_MODES), required=True,
            format_func=_TRIGGER_MODES.get, key="sim_trigger_mode",
        )
        top_k = st.slider("Top-K targets", 3, 15, 8, key="sim_top_k")
    mode = str(mode) if mode in _TRIGGER_MODES else "observed"

    sim_graph = graph
    graph_note = "mixed-wave graph"
    if mode == "observed":
        triggers_df = (
            active.groupby("region", observed=True)["elapsed_min"].min().reset_index()
            .assign(weight=1.0, role="active alarm")
            if not active.empty else pd.DataFrame(columns=["region", "elapsed_min", "weight", "role"])
        )
        empty_hint = ("No alarms are active at T, so there is nothing to propagate. Move T to a busier "
                      "moment, pick what-if regions, or parse an OSINT message.")
    elif mode == "whatif":
        with st.container(horizontal=True, vertical_alignment="bottom"):
            chosen = st.multiselect("What-if trigger regions", options=list(all_regions),
                                    default=list(all_regions[:1]), key="sim_manual_regions")
            elapsed = st.slider("Minutes since triggers started", 0, 180, 0, step=5, key="sim_elapsed")
        triggers_df = pd.DataFrame({"region": chosen, "elapsed_min": float(elapsed), "weight": 1.0,
                                    "role": "what-if"})
        empty_hint = "Pick at least one what-if region."
    else:
        if record is None:
            st.info("Parse an OSINT message above to inject its regions as triggers at T.",
                    icon=":material/translate:")
            return
        include_source = st.toggle("Include the source oblast as a trigger", value=True,
                                   key="osint_include_source")
        triggers_df = osint_triggers(record, include_source=include_source)
        wave = str(record.get("wave_class", "unknown"))
        if wave_graphs and wave in wave_graphs:
            sim_graph = wave_graphs[wave]
            graph_note = (f"{wave}-wave layer · "
                          f"{'observed' if sim_graph.graph.get('source') == 'observed' else 'prior topology'}")
        empty_hint = ("The message names no Ukrainian oblast (nationwide or external-origin threat), "
                      "so there is no regional trigger to inject. Try what-if mode for a hypothesis.")

    if triggers_df.empty:
        st.info(empty_hint, icon=":material/notifications_paused:")
        return

    triggers = [(str(r.region), float(r.elapsed_min), float(r.weight))
                for r in triggers_df.itertuples(index=False)]
    predictions = predict_cascade(sim_graph, triggers, cutoff, top_k=top_k, hops=params.hops,
                                  decay_min=params.decay_min)
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.badge(_TRIGGER_MODES[mode], icon=":material/input:", color="blue")
        st.badge(f"{len(triggers)} trigger{'s' if len(triggers) != 1 else ''} at T",
                 icon=":material/notifications_active:", color="red")
        st.badge(graph_note, icon=":material/hub:", color="gray")

    col_trig, col_pred = st.columns((1, 2))
    with col_trig, st.container(border=True):
        st.subheader("Active triggers", icon=":material/notifications_active:")
        trig_view = triggers_df.sort_values("elapsed_min")
        trig_view["in_graph"] = trig_view["region"].map(lambda r: r in sim_graph)
        st.dataframe(
            trig_view, hide_index=True,
            column_config={
                "region": "Region",
                "elapsed_min": st.column_config.NumberColumn("Elapsed (min)", format="%.0f"),
                "weight": st.column_config.ProgressColumn("Weight", min_value=0, max_value=1,
                                                          format="percent"),
                "role": "Role",
                "in_graph": st.column_config.CheckboxColumn("Has history"),
            },
        )
        missing = [r for r in trig_view.loc[~trig_view["in_graph"], "region"]]
        if missing:
            st.caption("No pre-T history for: " + ", ".join(missing))
    with col_pred, st.container(border=True):
        st.subheader(f"Top-{top_k} predicted next regions", icon=":material/trending_up:")
        if predictions.empty:
            st.info("The triggers have no outgoing propagation edges before T. Try a wider lag "
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

    highlight: dict[str, tuple[str, float]] = {r: ("#D1495B", 1.0) for r, _, _ in triggers}
    for row in predictions.itertuples(index=False):
        highlight[str(row.region)] = ("#F2A541", float(0.35 + 0.65 * row.probability))
    fig = render_graph_figure(
        sim_graph, highlight=highlight,
        min_weight=1 if sim_graph.graph.get("source") == "prior" else params.min_edge_weight,
        max_edges=60, title=f"Simulated cascade at T · {_TRIGGER_MODES[mode]}", dark=_theme_is_dark(),
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
# 4b. Methodology & architecture tab (judge-facing)
# --------------------------------------------------------------------------- #
_PIPELINE_MERMAID: Final[str] = """
flowchart LR
    subgraph Sources
        CSV[(Historical alarm CSV)]
        SYN[Synthetic generator]
        TG[Telegram OSINT text]
    end
    subgraph Safeguard["Temporal safeguard  (src/loader.py)"]
        LD[Clean + canonical schema]
        CUT{{"keep start_time < T<br/>mask end_time ≥ T"}}
        TRIP[Leakage tripwire]
    end
    subgraph Fusion["OSINT fusion  (src/llm_parser.py)"]
        GEM[Gemini 2.5 Flash<br/>JSON schema]
        RUL[Rule-based parser]
        VAL[Pydantic validation<br/>canonical oblasts]
    end
    subgraph Engine["Cascade engine  (src/dashboard.py)"]
        FAST[Fast-wave graph<br/>Δ ≈ 30 min]
        SLOW[Slow-wave graph<br/>Δ ≈ 90 min]
        PRED[Top-K prediction<br/>noisy-OR + ETA]
    end
    CSV --> LD
    SYN -. fallback .-> LD
    LD --> CUT --> TRIP
    TG --> GEM --> VAL
    GEM -. key missing / error .-> RUL --> VAL
    TRIP --> FAST & SLOW
    VAL -- source / targets / wave --> PRED
    FAST & SLOW --> PRED
    PRED --> UI[Streamlit evaluation console]
"""


@st.cache_data(ttl="1h", show_spinner=False)
def sample_osint_records() -> pd.DataFrame:
    """Parse the built-in demo messages offline for the methodology showcase.

    The parser is forced offline so this view never spends API quota or
    depends on the network; it demonstrates the schema and the fallback path.

    Returns:
        One row per demo message with the key structured fields.
    """
    parser = GeminiOSINTParser(force_offline=True)
    rows = []
    for message, record in zip(demo_messages(), parser.parse_many(demo_messages())):
        rows.append({
            "message": message,
            "threat_type": record.threat_type,
            "wave_class": record.wave_class,
            "source_region": record.source_region,
            "target_regions": record.target_regions,
            "confidence": record.confidence_score,
            "engine": record.parser,
        })
    return pd.DataFrame(rows)


def _layer_stat(graph: nx.DiGraph, key: str, reducer: Callable[[list[float]], float]) -> float:
    values = [float(d.get(key, 0.0)) for _, _, d in graph.edges(data=True)]
    return reducer(values) if values else 0.0


def render_methodology_tab(
    outcome: LoadOutcome,
    cutoff: pd.Timestamp,
    history: pd.DataFrame,
    visible: pd.DataFrame,
    wave_graphs: Mapping[str, nx.DiGraph],
    params: CascadeParams,
) -> None:
    """Tab 4: methodology and architecture write-up for judges.

    Every section pairs the explanation with live numbers from the current
    slice so the guarantees can be checked, not just read.

    Args:
        outcome: Load outcome for the current ``T``.
        cutoff: Observation instant ``T``.
        history: Full history frame.
        visible: Filtered, cutoff-safe alarm frame (may be empty).
        wave_graphs: Per-wave graphs from :func:`build_wave_graphs`.
        params: Cascade parameters from the sidebar.
    """
    settings = get_settings()
    fast, slow = wave_graphs["fast"], wave_graphs["slow"]
    observed_fast = fast.number_of_edges() if fast.graph.get("source") == "observed" else 0
    observed_slow = slow.number_of_edges() if slow.graph.get("source") == "observed" else 0
    max_start = visible["start_time"].max() if not visible.empty else None

    st.markdown(
        "**Multimodal Spatiotemporal Air-Threat Cascade & LLM Intelligence Engine** · a "
        "leakage-safe, graph-based model of how air-raid alarms propagate between Ukrainian oblasts, "
        "enriched with LLM-normalised OSINT. Everything on this page is recomputed from the sidebar "
        "cutoff **T**, so the numbers below are a live audit of the current slice."
    )

    kpi = st.columns(5)
    with kpi[0]:
        st.metric("Visible before T", f"{len(outcome.frame):,}", border=True,
                  help="Rows with start_time strictly before T (before threat filters).")
    with kpi[1]:
        st.metric("Withheld (start ≥ T)", f"{outcome.withheld_rows:,}", border=True,
                  help="Future rows removed by the loader; they never reach any computation.")
    with kpi[2]:
        st.metric("Ends masked at T", f"{outcome.masked_end_rows:,}", border=True,
                  help="Alarms still running at T whose end would be future knowledge.")
    with kpi[3]:
        st.metric("Edges fast · slow", f"{observed_fast:,} · {observed_slow:,}", border=True,
                  help="Directed propagation edges learned from pre-T alarms per wave layer.")
    with kpi[4]:
        st.metric("OSINT parser", "Gemini" if settings.llm_enabled else "Offline",
                  delta=settings.gemini_model if settings.llm_enabled else "rule-based (no API key)",
                  delta_arrow="off", border=True,
                  help="Live engine for Telegram OSINT parsing; falls back to rules automatically.")

    with st.container(border=True):
        st.subheader("End-to-end architecture", icon=":material/account_tree:")
        st.mermaid_chart(_PIPELINE_MERMAID)
        st.caption("Solid arrows: primary path. Dotted arrows: automatic fallbacks that keep the "
                   "system running offline (synthetic data, rule-based OSINT parsing).")

    # ---- 1. Graph diffusion ------------------------------------------------ #
    with st.expander("1 · Spatiotemporal graph diffusion", icon=":material/hub:", expanded=True):
        st.markdown(
            "Each oblast is a **node**; a directed **edge A → B** records that an alarm in B began "
            "within a wave-specific window Δ after an alarm in A for the same threat type. Edges "
            "carry the *conditional spread probability* and the *historical transition speed*:"
        )
        c1, c2 = st.columns(2)
        with c1:
            st.latex(r"P(B \mid A,\Delta_w)=\frac{N_{A\rightarrow B}(\Delta_w)}{N_A}")
            st.caption("Share of A's alarms followed by B inside the window.")
        with c2:
            st.latex(r"\tau_{AB}=\operatorname{median}\big(t^{\text{start}}_B-t^{\text{start}}_A\big)")
            st.caption("Median propagation lag in minutes (also mean and count).")
        st.markdown(
            "Given triggers active at T (region r, elapsed e_r minutes), each successor s receives a "
            "recency-discounted contribution; several triggers combine with a noisy-OR, and the "
            "arrival estimate is the remaining lag:"
        )
        st.latex(
            r"p_{r\rightarrow s}=P(s\mid r)\,e^{-\max(0,\,e_r-\tau_{rs})/\lambda},\qquad "
            r"P(s)=1-\prod_{r}\big(1-p_{r\rightarrow s}\big),\qquad "
            r"\widehat{\text{ETA}}_s=\max(0,\tau_{rs}-e_r)"
        )
        st.markdown(
            f"- λ (influence decay) is **{params.decay_min:.0f} min** and depth is "
            f"**{params.hops} hop{'s' if params.hops > 1 else ''}** in the sidebar; two-hop paths "
            "multiply probabilities and add lags.\n"
            "- Node layout uses oblast-centre coordinates, so the graph reads as a map; node size is "
            "alarm frequency before T.\n"
            "- When a layer has no observed edges before T, a **prior adjacency topology** (dashed) "
            "with distance-derived lags keeps the model explorable and is labelled as such."
        )

    # ---- 2. Velocity stratification ---------------------------------------- #
    with st.expander("2 · Velocity stratification: fast-wave vs slow-wave", icon=":material/speed:"):
        st.markdown(
            "Alarm cascades have two very different kinematics, so the engine learns **two graphs** "
            "instead of blending them:"
        )
        comparison = pd.DataFrame(
            {
                "Fast-wave": [
                    "Ballistic & cruise missiles, strategic/tactical aviation, guided bombs, artillery",
                    "Near-simultaneous multi-oblast activation (seconds to minutes)",
                    f"{params.fast_lag_min:.0f} min",
                    f"{_layer_stat(fast, 'median_lag_min', lambda v: float(np.median(v))):.0f} min",
                    f"{_layer_stat(fast, 'probability', lambda v: float(np.mean(v))):.0%}",
                    "Short alarms (≈15–60 min), few hops, radial from launch axis",
                ],
                "Slow-wave": [
                    "Shahed-type strike UAVs, reconnaissance UAVs",
                    "Progressive diffusion over 1–6 h as the group crosses oblasts",
                    f"{params.max_lag_min:.0f} min",
                    f"{_layer_stat(slow, 'median_lag_min', lambda v: float(np.median(v))):.0f} min",
                    f"{_layer_stat(slow, 'probability', lambda v: float(np.mean(v))):.0%}",
                    "Long alarms (1–3 h), multi-hop chains, strong night-time pattern",
                ],
            },
            index=["Threat types", "Dynamics", "Edge window Δ", "Median lag before T",
                   "Mean P(B | A) before T", "Signature"],
        )
        st.table(comparison)
        st.caption(
            "Live rows come from the current slice; layers marked *prior topology* on the wave map "
            "report prior values. The split matters for prediction: a fast-wave trigger that is "
            "already 20 min old has usually finished propagating, whereas a slow-wave trigger is "
            "still mid-cascade."
        )

    # ---- 3. Temporal safeguard --------------------------------------------- #
    with st.expander("3 · Strict temporal safeguard (no data leakage)", icon=":material/verified_user:"):
        st.markdown(
            "Evaluation at cutoff T only ever sees the dataset restricted to alarms that **started "
            "strictly before T**. The restriction is applied in the loader, before any feature, "
            "graph or prediction is computed, and is verified after the fact:"
        )
        st.latex(
            r"\mathcal{D}_T=\{\,x\in\mathcal{D}:\ t^{\text{start}}_x<T\,\},\qquad "
            r"t^{\text{end}}_x\leftarrow\text{NaT}\ \ \text{if}\ t^{\text{end}}_x\ge T,\qquad "
            r"\max_{x\in\mathcal{D}_T}t^{\text{start}}_x<T\ \ (\text{tripwire})"
        )
        st.markdown(
            "- **Strict inequality**: a row at exactly T is dropped (the boundary test asserts this).\n"
            "- **End-time masking**: an alarm still running at T has an unknown end; its end and "
            "duration are masked so no post-T information leaks through durations or 'active' status.\n"
            "- **Tripwire**: `TemporalLeakageError` is raised if any surviving row violates the "
            "inequality, so a regression fails loudly instead of silently inflating scores.\n"
            "- **Timezones**: all timestamps are normalised to UTC; naive cutoffs are treated as UTC.\n"
            "- The full history is loaded separately **only** for widget defaults and these audit "
            "counters; nothing analytical is derived from it."
        )
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            st.badge(f"T = {cutoff:%Y-%m-%d %H:%M} UTC", icon=":material/schedule:", color="blue")
            if max_start is not None:
                st.badge(f"latest visible start = {max_start:%Y-%m-%d %H:%M} UTC",
                         icon=":material/check_circle:", color="green")
                st.badge(f"margin {max(0.0, (cutoff - max_start).total_seconds() / 60):.0f} min",
                         icon=":material/straighten:", color="gray")
            else:
                st.badge("no rows before T", icon=":material/hourglass_empty:", color="gray")
        st.code(
            "keep = df['start_time'] < cutoff            # strict: rows at T are dropped\n"
            "out  = df.loc[keep]\n"
            "out.loc[out['end_time'] >= cutoff, 'end_time'] = pd.NaT   # still running at T\n"
            "assert (out['start_time'] < cutoff).all()    # TemporalLeakageError otherwise",
            language="python",
        )

    # ---- 4. Multimodal OSINT fusion ---------------------------------------- #
    with st.expander("4 · Multimodal OSINT fusion with Gemini", icon=":material/translate:"):
        st.markdown(
            "Monitoring channels post terse, slang-heavy updates (\"6 мопедів курсом на Полтавщину\"). "
            "The engine normalises them into the same vocabulary as the alarm log so text can enrich "
            "the tabular signal:\n\n"
            "1. **Gemini 2.5 Flash** via the official `google-genai` SDK, temperature 0, forced JSON "
            "output against a schema (`threat_type`, `source_region`, `target_regions`, "
            "`confidence_score`, `raw_summary`, heading, count).\n"
            "2. **Pydantic validation** canonicalises slang threat labels into the loader taxonomy, "
            "derives `wave_class`, maps Ukrainian/Russian/English region spellings to oblasts, drops "
            "external launch areas from targets, clamps confidence and records unmapped mentions.\n"
            "3. **Graceful degradation**: missing/invalid key, SDK absent, timeouts or malformed JSON "
            "fall back to a rule-based parser (and canned payloads for demo scenarios); a circuit "
            "breaker stops retrying after repeated failures. The pipeline never crashes.\n"
            "4. **Fusion**: parsed `source_region` / `target_regions` / `wave_class` become extra "
            "triggers and priors for the cascade predictor, and `confidence_score` weights them."
        )
        st.caption("Offline showcase (rule-based engine, no API calls) over the built-in demo messages:")
        st.dataframe(
            sample_osint_records(),
            hide_index=True,
            column_config={
                "message": st.column_config.TextColumn("Telegram message", width="large"),
                "threat_type": "Threat",
                "wave_class": "Wave",
                "source_region": "Source",
                "target_regions": st.column_config.ListColumn("Targets"),
                "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1,
                                                              format="percent"),
                "engine": "Engine",
            },
        )

    # ---- 5. Evaluation protocol & limitations ------------------------------ #
    with st.expander("5 · Evaluation protocol, fallbacks and limitations", icon=":material/fact_check:"):
        st.markdown(
            "**Backtest loop.** Pick T → load 𝒟_T → learn fast/slow graphs → take alarms active at T "
            "as triggers → rank top-K next regions with ETAs → compare with what actually happened "
            "after T (held-out by construction). The sidebar nudge buttons make this a one-click walk "
            "through time.\n\n"
            "**Fallbacks that keep the demo honest and alive.** Missing CSV → synthetic generator "
            "(banner shown). No observed edges → prior adjacency topology (dashed, labelled). No API "
            "key → rule-based OSINT parsing. No active alarms at T → what-if trigger mode.\n\n"
            "**Known limitations.** Synthetic data mimics spatial skew, burst structure and diurnal "
            "pattern but not real geopolitics; co-occurrence edges capture correlation, not causation; "
            "the prior topology uses nominal speeds and probabilities; OSINT confidence is calibrated "
            "heuristically in offline mode."
        )

    # ---- 6. Module map ------------------------------------------------------ #
    with st.expander("6 · Module map", icon=":material/folder_code:"):
        st.table(
            {
                ":material/database: `src/loader.py`":
                    "CSV ingestion, cleaning, canonical schema, synthetic generator, **temporal cutoff** "
                    "and leakage tripwire.",
                ":material/settings: `src/config.py`":
                    "Crash-proof settings from `.env`; missing `GEMINI_API_KEY` ⇒ offline mode.",
                ":material/translate: `src/llm_parser.py`":
                    "Gemini OSINT parser with schema-constrained JSON, Pydantic validation, rule-based "
                    "fallback and circuit breaker.",
                ":material/public: `src/dashboard.py`":
                    "Fast/slow propagation graphs, cascade predictor, wave map, simulator and this "
                    "methodology page.",
                ":material/science: `tests/`":
                    "Unit tests for every module plus headless end-to-end runs of the page via "
                    "`streamlit.testing`.",
            },
            border="horizontal",
        )


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

    tab_overview, tab_map, tab_sim, tab_method = st.tabs(
        [":material/dashboard: Overview & metrics",
         ":material/public: Wave map & spatiotemporal graph",
         ":material/model_training: Cascade simulator",
         ":material/menu_book: Methodology & architecture"]
    )
    wave_graphs = build_wave_graphs(visible, params)
    if visible.empty:
        reason = ("No alarms recorded before T" if outcome.frame.empty
                  else "The threat filter hides every alarm before T")
        with tab_overview:
            render_empty_state(cutoff, history, reason=reason, slot="overview")
        with tab_map:
            # The wave map always renders: default topology + what-if simulation.
            render_wave_map_tab(wave_graphs, visible, active, cutoff, params, empty_reason=reason)
        with tab_sim:
            render_empty_state(cutoff, history, reason=reason, slot="simulator")
        with tab_method:
            render_methodology_tab(outcome, cutoff, history, visible, wave_graphs, params)
        return

    graph = build_propagation_graph(visible, max_lag_min=params.max_lag_min)
    regions = [r for r in UKRAINIAN_REGIONS if r in set(history["region"])] or sorted(history["region"].unique())
    with tab_overview:
        render_overview_tab(visible, active, cutoff)
    with tab_map:
        render_wave_map_tab(wave_graphs, visible, active, cutoff, params)
    with tab_sim:
        render_simulator_tab(graph, active, cutoff, params, regions, wave_graphs=wave_graphs)
    with tab_method:
        render_methodology_tab(outcome, cutoff, history, visible, wave_graphs, params)


if __name__ == "__main__":
    main()
