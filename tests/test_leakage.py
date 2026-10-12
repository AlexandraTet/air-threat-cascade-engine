"""No-data-leakage invariance tests.

The temporal guarantee is stated as: *every quantity computed at cutoff T is a
function of the alarms that started strictly before T, with end times that
were unknown at T masked.* These tests enforce it end to end rather than by
inspection:

* ``world_as_of(T)`` builds the slice independently of the loader's cutoff
  code (drop rows at/after T, mask running ends) and every analytic output
  must match the loader-produced slice exactly.
* ``perturb_future(T)`` rewrites everything that happens at or after T
  (regions, threat types, end times of alarms still running at T, extra
  future rows). Every analytic output must be *bit-identical* to the
  unperturbed run: if any future information leaked, something would move.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from src.dashboard import (
    CascadeParams,
    active_alarms,
    build_propagation_graph,
    build_wave_graphs,
    cascade_velocity,
    daily_wave_counts,
    hourly_onsets,
    node_activity,
    predict_cascade,
)
from src.loader import THREAT_TYPES, AirThreatLoader, generate_synthetic_alarms, load_and_filter_alarms


def world_as_of(full: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Independent construction of the information set available at ``T``."""
    df = full.loc[full["start_time"] < cutoff].copy()
    running = df["end_time"].notna() & (df["end_time"] >= cutoff)
    df.loc[running, "end_time"] = pd.NaT
    df["duration_min"] = (df["end_time"] - df["start_time"]).dt.total_seconds() / 60.0
    return df.sort_values(["start_time", "region"], kind="mergesort").reset_index(drop=True)


def perturb_future(full: pd.DataFrame, cutoff: pd.Timestamp, seed: int) -> pd.DataFrame:
    """Rewrite the future (>= T) while leaving the past (< T) untouched."""
    rng = np.random.default_rng(seed)
    df = full.copy()
    future = df["start_time"] >= cutoff
    # 1. Shuffle which region / threat each future alarm belongs to.
    idx = df.index[future]
    perm = rng.permutation(len(idx))
    df.loc[idx, "region"] = df.loc[idx, "region"].to_numpy()[perm]
    threats = df.loc[idx, "threat_type"].to_numpy()[perm]
    df.loc[idx, "threat_type"] = threats
    df.loc[idx, "wave_class"] = [THREAT_TYPES[t].value for t in threats]
    # 2. Move future onsets around (still >= T) and their ends.
    shift = pd.to_timedelta(rng.integers(0, 6 * 60, size=len(idx)), unit="m")
    df.loc[idx, "start_time"] = df.loc[idx, "start_time"] + shift
    df.loc[idx, "end_time"] = df.loc[idx, "start_time"] + pd.to_timedelta(rng.integers(5, 300, size=len(idx)), unit="m")
    # 3. Alarms running at T: change how long they keep running (end stays >= T).
    running = (~future) & df["end_time"].notna() & (df["end_time"] >= cutoff)
    extra = pd.to_timedelta(rng.integers(1, 500, size=int(running.sum())), unit="m")
    df.loc[running, "end_time"] = cutoff + extra
    # 4. Append brand-new future alarms.
    regions = df["region"].unique()
    new = pd.DataFrame({
        "alarm_id": np.arange(len(df), len(df) + 40),
        "region": rng.choice(regions, size=40),
        "start_time": cutoff + pd.to_timedelta(rng.integers(0, 24 * 60, size=40), unit="m"),
        "threat_type": rng.choice(list(THREAT_TYPES), size=40),
    })
    new["wave_class"] = [THREAT_TYPES[t].value for t in new["threat_type"]]
    new["end_time"] = new["start_time"] + pd.Timedelta(minutes=30)
    new["duration_min"] = 30.0
    new["source"] = "synthetic"
    out = pd.concat([df, new[df.columns]], ignore_index=True)
    out["duration_min"] = (out["end_time"] - out["start_time"]).dt.total_seconds() / 60.0
    return out.sort_values(["start_time", "region"], kind="mergesort").reset_index(drop=True)


def analytics(df: pd.DataFrame, cutoff: pd.Timestamp, params: CascadeParams) -> dict:
    """Every analytic quantity the dashboard derives from a slice."""
    active = active_alarms(df, cutoff)
    graphs = build_wave_graphs(df, params)
    mixed = build_propagation_graph(df, max_lag_min=params.max_lag_min)
    triggers = [(str(r), float(e)) for r, e in
                active.groupby("region", observed=True)["elapsed_min"].min().items()]
    return {
        "rows": len(df),
        "max_start": df["start_time"].max() if len(df) else None,
        "active": active[["region", "start_time", "elapsed_min"]].reset_index(drop=True),
        "graphs": graphs,
        "mixed": mixed,
        "predictions": predict_cascade(mixed, triggers, cutoff, top_k=10, hops=2),
        "activity": node_activity(df, cutoff, window_hours=24),
        "velocity": cascade_velocity(df, cutoff),
        "onsets": hourly_onsets(df, cutoff),
        "daily": daily_wave_counts(df),
    }


class FutureInvarianceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.full = generate_synthetic_alarms(1500, seed=21)
        self.loader = AirThreatLoader()
        self.params = CascadeParams()
        starts = self.full["start_time"]
        self.cutoffs = [starts.iloc[int(len(starts) * q)] + pd.Timedelta(minutes=7) for q in (0.2, 0.5, 0.8)]
        self.cutoffs.append(starts.iloc[600])  # exactly on an onset: boundary case

    def assert_same(self, a: dict, b: dict) -> None:
        self.assertEqual(a["rows"], b["rows"])
        self.assertEqual(a["max_start"], b["max_start"])
        pd.testing.assert_frame_equal(a["active"], b["active"])
        pd.testing.assert_frame_equal(a["predictions"], b["predictions"])
        pd.testing.assert_series_equal(a["activity"], b["activity"])
        self.assertEqual(a["velocity"], b["velocity"])
        self.assertEqual(a["onsets"], b["onsets"])
        pd.testing.assert_frame_equal(a["daily"], b["daily"])
        self.assertTrue(nx.utils.graphs_equal(a["mixed"], b["mixed"]))
        for wave in ("fast", "slow"):
            self.assertTrue(nx.utils.graphs_equal(a["graphs"][wave], b["graphs"][wave]), wave)

    def test_loader_slice_matches_independent_construction(self) -> None:
        for cutoff in self.cutoffs:
            via_loader = self.loader.apply_temporal_cutoff(self.full, cutoff)
            independent = world_as_of(self.full, cutoff)
            pd.testing.assert_frame_equal(via_loader, independent, check_like=True)
            self.assertLess(via_loader["start_time"].max(), cutoff)
            self.assertTrue(via_loader["end_time"].dropna().lt(cutoff).all())

    def test_outputs_invariant_to_any_change_after_T(self) -> None:
        for cutoff in self.cutoffs:
            baseline = analytics(self.loader.apply_temporal_cutoff(self.full, cutoff), cutoff, self.params)
            self.assertFalse(baseline["predictions"].empty or baseline["mixed"].number_of_edges() == 0,
                             "baseline must be non-trivial for the invariance check to mean anything")
            for seed in (1, 2):
                perturbed = perturb_future(self.full, cutoff, seed)
                self.assertGreater(len(perturbed), len(self.full))  # future really changed
                probe = analytics(self.loader.apply_temporal_cutoff(perturbed, cutoff), cutoff, self.params)
                self.assert_same(baseline, probe)

    def test_future_changes_do_alter_an_unsafe_pipeline(self) -> None:
        """Sanity check of the test itself: without the cutoff, the perturbation is visible."""
        cutoff = self.cutoffs[1]
        unsafe_a = analytics(self.full, cutoff, self.params)
        unsafe_b = analytics(perturb_future(self.full, cutoff, 3), cutoff, self.params)
        self.assertFalse(nx.utils.graphs_equal(unsafe_a["mixed"], unsafe_b["mixed"]))

    def test_csv_path_is_equally_invariant(self) -> None:
        cutoff = self.cutoffs[2]
        with tempfile.TemporaryDirectory() as tmp:
            base, pert = Path(tmp) / "base.csv", Path(tmp) / "pert.csv"
            self.full.to_csv(base, index=False)
            perturb_future(self.full, cutoff, 9).to_csv(pert, index=False)
            a = load_and_filter_alarms(str(base), cutoff.to_pydatetime())
            b = load_and_filter_alarms(str(pert), cutoff.to_pydatetime())
        pd.testing.assert_frame_equal(a.drop(columns=["alarm_id"]), b.drop(columns=["alarm_id"]))
        self.assertLess(a["start_time"].max(), cutoff)

    def test_boundary_semantics(self) -> None:
        onset = self.full["start_time"].iloc[600]
        at_boundary = self.loader.apply_temporal_cutoff(self.full, onset)
        self.assertFalse((at_boundary["start_time"] == onset).any(), "a row starting exactly at T is future")
        running = self.full.loc[(self.full["start_time"] < onset) & (self.full["end_time"] == onset)]
        if len(running):
            masked = at_boundary.merge(running[["region", "start_time"]], on=["region", "start_time"])
            self.assertTrue(masked["end_time"].isna().all(), "an end exactly at T is not yet observed")

    def test_latest_period_is_fully_usable(self) -> None:
        """T after the last alarm: everything is visible and nothing is masked by the cutoff."""
        after = self.full["start_time"].max() + pd.Timedelta(days=1)
        sliced = self.loader.apply_temporal_cutoff(self.full, after)
        self.assertEqual(len(sliced), len(self.full))
        self.assertEqual(int(sliced["end_time"].isna().sum()), int(self.full["end_time"].isna().sum()))


if __name__ == "__main__":
    unittest.main()
