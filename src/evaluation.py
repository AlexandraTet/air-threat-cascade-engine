"""Leakage-safe backtest harness for the cascade predictor.

For each cutoff ``T`` the harness

1. builds the information set available at ``T`` through the loader's
   temporal safeguard (``start_time < T``, running ends masked),
2. learns the propagation graph and predicts the Top-K next regions from the
   alarms active at ``T``,
3. only then opens the **future** window ``[T, T + horizon)`` of the full
   history to score the prediction.

The future is touched in exactly one function, :func:`ground_truth`, and the
model never sees it: :func:`backtest_cutoff` asserts that the frame used for
prediction ends strictly before ``T``. A naive baseline ("the K most active
regions in the last 24 h") is scored on the same labels so the graph model's
lift can be read directly.

Run from the project root::

    python -m src.evaluation            # synthetic history, 12 cutoffs
    python -m src.evaluation --csv data/alarms.csv --cutoffs 20 --top-k 8
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict, dataclass
from typing import Final, Optional, Sequence

import pandas as pd

from src.dashboard import (
    CascadeParams,
    active_alarms,
    build_propagation_graph,
    build_wave_graphs,
    node_activity,
    predict_cascade,
)
from src.loader import AirThreatLoader, generate_synthetic_alarms

__all__ = ["BacktestResult", "backtest_cutoff", "choose_cutoffs", "ground_truth", "run_backtest", "summarise"]

logger = logging.getLogger(__name__)

DEFAULT_HORIZON_MIN: Final[float] = 180.0


@dataclass(frozen=True)
class BacktestResult:
    """Scores for one cutoff.

    Attributes:
        cutoff: Observation instant ``T``.
        n_visible: Alarms available to the model (``start_time < T``).
        n_triggers: Regions active at ``T`` used as triggers.
        n_predicted: Regions returned by the model (``<= top_k``).
        n_actual: Regions with a new onset in ``[T, T + horizon)`` that were
            not already triggers.
        hits: Predicted regions that did alarm in the horizon.
        precision_at_k: ``hits / n_predicted`` (``NaN`` when nothing predicted).
        recall_at_k: ``hits / n_actual`` (``NaN`` when nothing happened).
        mean_abs_eta_error_min: Mean absolute error of the arrival estimate
            over hits (``NaN`` when no hits).
        baseline_precision_at_k: Precision of the most-active-regions baseline.
        baseline_recall_at_k: Recall of the same baseline.
    """

    cutoff: pd.Timestamp
    n_visible: int
    n_triggers: int
    n_predicted: int
    n_actual: int
    hits: int
    precision_at_k: float
    recall_at_k: float
    mean_abs_eta_error_min: float
    baseline_precision_at_k: float
    baseline_recall_at_k: float


def ground_truth(history: pd.DataFrame, cutoff: pd.Timestamp, horizon_min: float) -> pd.Series:
    """First onset per region inside ``[T, T + horizon)``.

    This is the **only** function that reads data at or after ``T``; it is
    used for labels after the prediction has been made.

    Args:
        history: Full alarm history.
        cutoff: Observation instant ``T``.
        horizon_min: Scoring window length in minutes.

    Returns:
        Series ``region -> first onset timestamp`` within the window.
    """
    window = history.loc[
        (history["start_time"] >= cutoff)
        & (history["start_time"] < cutoff + pd.Timedelta(minutes=horizon_min))
    ]
    return window.groupby("region", observed=True)["start_time"].min()


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / denominator if denominator else float("nan")


def backtest_cutoff(
    history: pd.DataFrame,
    cutoff: pd.Timestamp,
    *,
    params: Optional[CascadeParams] = None,
    top_k: int = 8,
    horizon_min: float = DEFAULT_HORIZON_MIN,
    wave: Optional[str] = None,
) -> BacktestResult:
    """Predict at ``T`` from pre-``T`` data only, then score against the future.

    Args:
        history: Full alarm history (canonical schema).
        cutoff: Observation instant ``T``.
        params: Cascade parameters; defaults to :class:`CascadeParams`.
        top_k: Number of regions predicted.
        horizon_min: Scoring window after ``T`` in minutes.
        wave: ``"fast"`` / ``"slow"`` to use that layer, ``None`` for the
            mixed-wave graph.

    Returns:
        A :class:`BacktestResult`.

    Raises:
        AssertionError: If the frame handed to the model contains any row at
            or after ``T`` (leakage tripwire).
    """
    params = params or CascadeParams()
    cutoff = pd.Timestamp(cutoff)
    visible = AirThreatLoader().apply_temporal_cutoff(history, cutoff)
    assert visible.empty or visible["start_time"].max() < cutoff, "model input leaks past T"

    active = active_alarms(visible, cutoff)
    triggers = [(str(r), float(e)) for r, e in
                active.groupby("region", observed=True)["elapsed_min"].min().items()]
    trigger_set = {r for r, _ in triggers}

    if wave in ("fast", "slow"):
        graph = build_wave_graphs(visible, params)[wave]
    else:
        graph = build_propagation_graph(visible, max_lag_min=params.max_lag_min)
    predictions = predict_cascade(graph, triggers, cutoff, top_k=top_k, hops=params.hops,
                                  decay_min=params.decay_min) if triggers else pd.DataFrame()
    predicted = list(predictions["region"]) if not predictions.empty else []

    recent = node_activity(visible, cutoff, window_hours=24.0)
    baseline = [r for r in recent.index if r not in trigger_set][:top_k]

    # ---- the future is opened only here, for labels ---------------------- #
    truth = ground_truth(history, cutoff, horizon_min)
    truth = truth[~truth.index.isin(trigger_set)]
    actual = set(truth.index)

    hit_regions = [r for r in predicted if r in actual]
    eta_errors = []
    for region in hit_regions:
        eta_time = predictions.set_index("region").loc[region, "eta_time"]
        eta_errors.append(abs((truth[region] - eta_time).total_seconds()) / 60.0)
    baseline_hits = sum(1 for r in baseline if r in actual)

    return BacktestResult(
        cutoff=cutoff,
        n_visible=len(visible),
        n_triggers=len(triggers),
        n_predicted=len(predicted),
        n_actual=len(actual),
        hits=len(hit_regions),
        precision_at_k=_safe_ratio(len(hit_regions), len(predicted)),
        recall_at_k=_safe_ratio(len(hit_regions), len(actual)),
        mean_abs_eta_error_min=float(sum(eta_errors) / len(eta_errors)) if eta_errors else float("nan"),
        baseline_precision_at_k=_safe_ratio(baseline_hits, len(baseline)),
        baseline_recall_at_k=_safe_ratio(baseline_hits, len(actual)),
    )


def choose_cutoffs(history: pd.DataFrame, n: int, *, min_history_hours: float = 48.0) -> list[pd.Timestamp]:
    """Evenly spaced cutoffs ten minutes after real onsets, skipping the warm-up.

    Args:
        history: Full alarm history.
        n: Number of cutoffs.
        min_history_hours: Earliest cutoff is this long after the first alarm.

    Returns:
        Sorted list of cutoffs.
    """
    starts = history["start_time"].sort_values().reset_index(drop=True)
    eligible = starts[starts >= starts.iloc[0] + pd.Timedelta(hours=min_history_hours)]
    if eligible.empty:
        eligible = starts
    positions = [int(round(i)) for i in pd.Series(range(n)).mul((len(eligible) - 1) / max(n - 1, 1))]
    return [eligible.iloc[p] + pd.Timedelta(minutes=10) for p in positions]


def run_backtest(
    history: pd.DataFrame,
    cutoffs: Sequence[pd.Timestamp] | int = 12,
    *,
    params: Optional[CascadeParams] = None,
    top_k: int = 8,
    horizon_min: float = DEFAULT_HORIZON_MIN,
    wave: Optional[str] = None,
) -> pd.DataFrame:
    """Backtest many cutoffs.

    Args:
        history: Full alarm history.
        cutoffs: Explicit cutoffs or a count passed to :func:`choose_cutoffs`.
        params: Cascade parameters.
        top_k: Number of regions predicted.
        horizon_min: Scoring window after ``T``.
        wave: Optional wave layer.

    Returns:
        One row per cutoff with the :class:`BacktestResult` fields.
    """
    if isinstance(cutoffs, int):
        cutoffs = choose_cutoffs(history, cutoffs)
    rows = [asdict(backtest_cutoff(history, c, params=params, top_k=top_k, horizon_min=horizon_min, wave=wave))
            for c in cutoffs]
    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame) -> pd.Series:
    """Aggregate a backtest table into headline numbers (means ignore NaN)."""
    scored = results.loc[results["n_predicted"] > 0]
    return pd.Series({
        "cutoffs": len(results),
        "cutoffs_with_triggers": int((results["n_triggers"] > 0).sum()),
        "precision_at_k": float(scored["precision_at_k"].mean()) if len(scored) else float("nan"),
        "recall_at_k": float(scored["recall_at_k"].mean()) if len(scored) else float("nan"),
        "baseline_precision_at_k": float(scored["baseline_precision_at_k"].mean()) if len(scored) else float("nan"),
        "baseline_recall_at_k": float(scored["baseline_recall_at_k"].mean()) if len(scored) else float("nan"),
        "mean_abs_eta_error_min": float(scored["mean_abs_eta_error_min"].mean()) if len(scored) else float("nan"),
        "total_hits": int(results["hits"].sum()),
    })


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Leakage-safe backtest of the cascade predictor.")
    parser.add_argument("--csv", help="Alarm CSV; synthetic history when omitted.")
    parser.add_argument("--records", type=int, default=3000, help="Synthetic rows when no CSV.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cutoffs", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--horizon-min", type=float, default=DEFAULT_HORIZON_MIN)
    parser.add_argument("--wave", choices=["fast", "slow"], default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    if args.csv:
        loader = AirThreatLoader()
        history = loader.clean(loader.load_csv(args.csv))
    else:
        history = generate_synthetic_alarms(args.records, seed=args.seed)
    results = run_backtest(history, args.cutoffs, top_k=args.top_k, horizon_min=args.horizon_min, wave=args.wave)
    table = results.copy()
    numeric = table.select_dtypes("number").columns
    table[numeric] = table[numeric].round(3)
    table["cutoff"] = table["cutoff"].dt.strftime("%Y-%m-%d %H:%M")
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(table.to_string(index=False))
        print("\nsummary:")
        print(summarise(results).round(3).to_string())
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
