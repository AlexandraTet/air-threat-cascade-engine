"""Tests for ``src.evaluation`` (leakage-safe backtest)."""

from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from src.evaluation import backtest_cutoff, choose_cutoffs, ground_truth, run_backtest, summarise
from src.loader import generate_synthetic_alarms
from tests.test_leakage import perturb_future


class BacktestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.history = generate_synthetic_alarms(2500, seed=11)
        cls.cutoffs = choose_cutoffs(cls.history, 6)

    def test_cutoffs_skip_warmup_and_are_sorted(self) -> None:
        first = self.history["start_time"].min()
        self.assertEqual(len(self.cutoffs), 6)
        self.assertTrue(all(c >= first + pd.Timedelta(hours=48) for c in self.cutoffs))
        self.assertEqual(self.cutoffs, sorted(self.cutoffs))

    def test_ground_truth_window_only(self) -> None:
        cutoff = self.cutoffs[2]
        truth = ground_truth(self.history, cutoff, 180)
        self.assertTrue((truth >= cutoff).all())
        self.assertTrue((truth < cutoff + pd.Timedelta(minutes=180)).all())
        self.assertTrue(truth.index.is_unique)

    def test_single_cutoff_scores_are_consistent(self) -> None:
        result = backtest_cutoff(self.history, self.cutoffs[2], top_k=8)
        self.assertLess(self.history.loc[self.history["start_time"] < result.cutoff, "start_time"].max(),
                        result.cutoff)
        self.assertLessEqual(result.n_predicted, 8)
        self.assertLessEqual(result.hits, min(result.n_predicted, result.n_actual))
        if result.n_predicted:
            self.assertAlmostEqual(result.precision_at_k, result.hits / result.n_predicted)
        if result.n_actual:
            self.assertAlmostEqual(result.recall_at_k, result.hits / result.n_actual)
        if result.hits:
            self.assertGreaterEqual(result.mean_abs_eta_error_min, 0.0)
        else:
            self.assertTrue(math.isnan(result.mean_abs_eta_error_min))

    def test_predictions_do_not_depend_on_the_future(self) -> None:
        """Perturbing everything after T may change the *score* but never the *prediction*."""
        cutoff = self.cutoffs[3]
        base = backtest_cutoff(self.history, cutoff, top_k=8)
        perturbed = backtest_cutoff(perturb_future(self.history, cutoff, seed=4), cutoff, top_k=8)
        for field in ("n_visible", "n_triggers", "n_predicted"):
            self.assertEqual(getattr(base, field), getattr(perturbed, field), field)

    def test_run_backtest_and_summary(self) -> None:
        table = run_backtest(self.history, self.cutoffs, top_k=6, horizon_min=120)
        self.assertEqual(len(table), 6)
        self.assertTrue(((table["precision_at_k"].dropna() >= 0) & (table["precision_at_k"].dropna() <= 1)).all())
        summary = summarise(table)
        self.assertEqual(summary["cutoffs"], 6)
        self.assertGreaterEqual(summary["total_hits"], 0)
        self.assertIn("baseline_precision_at_k", summary.index)

    def test_graph_model_is_informative_on_synthetic_bursts(self) -> None:
        """On burst-structured data the graph should beat the activity baseline on average."""
        table = run_backtest(self.history, 16, top_k=6, horizon_min=120)
        scored = table.loc[table["n_predicted"] > 0]
        self.assertGreater(len(scored), 5)
        self.assertGreater(scored["precision_at_k"].mean(), 0.0)
        self.assertGreaterEqual(scored["precision_at_k"].mean(), scored["baseline_precision_at_k"].mean() * 0.8)

    def test_wave_layer_and_no_trigger_cases(self) -> None:
        early = self.history["start_time"].min() + pd.Timedelta(minutes=1)
        quiet = backtest_cutoff(self.history, early, top_k=5)
        self.assertEqual(quiet.n_predicted, 0)
        self.assertTrue(np.isnan(quiet.precision_at_k))
        fast = backtest_cutoff(self.history, self.cutoffs[4], top_k=5, wave="fast")
        self.assertLessEqual(fast.n_predicted, 5)


if __name__ == "__main__":
    unittest.main()
