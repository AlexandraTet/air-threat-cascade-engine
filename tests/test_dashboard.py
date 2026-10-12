"""Tests for ``src.dashboard``.

Pure analytics are tested directly; the page itself is exercised headlessly
with ``streamlit.testing.v1.AppTest`` (no browser or server required).

Run with ``python -m unittest discover tests``.
"""

from __future__ import annotations

import unittest
from datetime import date, time, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from streamlit.testing.v1 import AppTest

from src.dashboard import (
    PREDICTION_COLUMNS,
    active_alarms,
    apply_threat_filters,
    build_propagation_graph,
    cascade_velocity,
    daily_wave_counts,
    default_cutoff,
    edge_table,
    hourly_onsets,
    node_roles,
    predict_cascade,
    render_graph_figure,
    resolve_source,
)
from src.loader import AirThreatLoader, generate_synthetic_alarms

UTC = timezone.utc
APP_PATH = Path(__file__).resolve().parent.parent / "src" / "dashboard.py"


def _ts(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC")


def _frame(rows: list[tuple[str, str, str | None, str]]) -> pd.DataFrame:
    """Build a canonical-schema frame from (region, start, end, threat) tuples."""
    raw = pd.DataFrame(rows, columns=["region", "start_time", "end_time", "threat_type"])
    return AirThreatLoader().clean(raw)


class ActiveAndVelocityTests(unittest.TestCase):
    def test_active_alarms_respects_masking_and_staleness(self) -> None:
        df = _frame(
            [
                ("Kyiv City", "2026-01-01 10:00", None, "uav"),  # recent, open -> active
                ("Lviv Oblast", "2026-01-01 00:30", None, "uav"),  # open but stale -> feed gap
                ("Sumy Oblast", "2026-01-01 09:00", "2026-01-01 09:20", "ballistic"),  # ended
                ("Odesa Oblast", "2026-01-01 10:30", "2026-01-01 12:00", "uav"),  # end > T (unmasked)
            ]
        )
        cutoff = _ts("2026-01-01 11:00")
        active = active_alarms(df, cutoff)
        self.assertEqual(sorted(active["region"]), ["Kyiv City", "Odesa Oblast"])
        kyiv = active.loc[active["region"] == "Kyiv City", "elapsed_min"].iloc[0]
        self.assertAlmostEqual(float(kyiv), 60.0)

    def test_active_alarms_empty(self) -> None:
        df = _frame([("Kyiv City", "2026-01-01 10:00", None, "uav")]).iloc[0:0]
        out = active_alarms(df, _ts("2026-01-01 11:00"))
        self.assertTrue(out.empty)
        self.assertIn("elapsed_min", out.columns)

    def test_cascade_velocity_windows(self) -> None:
        rows = [("Kyiv City", f"2026-01-01 {h:02d}:00", None, "uav") for h in range(0, 12)]
        df = _frame(rows)
        current, previous = cascade_velocity(df, _ts("2026-01-01 12:00"), window_hours=6)
        self.assertAlmostEqual(current, 1.0)
        self.assertAlmostEqual(previous, 1.0)
        spark = hourly_onsets(df, _ts("2026-01-01 12:00"), hours=6)
        self.assertEqual(spark, [1, 1, 1, 1, 1, 1])

    def test_daily_counts_and_filters(self) -> None:
        df = generate_synthetic_alarms(300, seed=4)
        daily = daily_wave_counts(df)
        self.assertEqual(int(daily.to_numpy().sum()), 300)
        only_fast = apply_threat_filters(df, ["fast"], [])
        self.assertTrue((only_fast["wave_class"] == "fast").all())
        only_shahed = apply_threat_filters(df, ["slow"], ["uav_shahed"])
        self.assertTrue((only_shahed["threat_type"] == "uav_shahed").all())


class GraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.df = _frame(
            [
                ("Sumy Oblast", "2026-01-01 10:00", "2026-01-01 10:20", "ballistic"),
                ("Poltava Oblast", "2026-01-01 10:05", "2026-01-01 10:25", "ballistic"),
                ("Kyiv City", "2026-01-01 10:12", "2026-01-01 10:30", "ballistic"),
                ("Sumy Oblast", "2026-01-02 10:00", "2026-01-02 10:20", "ballistic"),
                ("Poltava Oblast", "2026-01-02 10:07", "2026-01-02 10:25", "ballistic"),
                ("Lviv Oblast", "2026-01-02 10:08", None, "uav"),  # different threat: no edge
                ("Kyiv City", "2026-01-02 14:00", None, "ballistic"),  # outside lag window
            ]
        )

    def test_edges_weights_and_lags(self) -> None:
        graph = build_propagation_graph(self.df, max_lag_min=30)
        self.assertTrue(graph.has_edge("Sumy Oblast", "Poltava Oblast"))
        edge = graph["Sumy Oblast"]["Poltava Oblast"]
        self.assertEqual(edge["weight"], 2)
        self.assertAlmostEqual(edge["median_lag_min"], 6.0)
        self.assertAlmostEqual(edge["probability"], 1.0)  # both Sumy alarms were followed
        self.assertEqual(graph["Sumy Oblast"]["Kyiv City"]["weight"], 1)
        self.assertFalse(graph.has_edge("Poltava Oblast", "Lviv Oblast"))
        self.assertFalse(graph.has_edge("Kyiv City", "Sumy Oblast"))  # direction matters
        self.assertEqual(graph.nodes["Sumy Oblast"]["alarms"], 2)

    def test_empty_graph_and_tables(self) -> None:
        graph = build_propagation_graph(self.df.iloc[0:0])
        self.assertEqual(graph.number_of_nodes(), 0)
        self.assertTrue(edge_table(graph).empty)
        sources, sinks = node_roles(graph)
        self.assertTrue(sources.empty and sinks.empty)
        fig = render_graph_figure(graph, title="empty")
        self.assertIsNotNone(fig)
        plt.close(fig)

    def test_edge_table_and_roles(self) -> None:
        graph = build_propagation_graph(self.df, max_lag_min=30)
        table = edge_table(graph, min_weight=2)
        self.assertEqual(table.iloc[0]["source"], "Sumy Oblast")
        self.assertEqual(table.iloc[0]["target"], "Poltava Oblast")
        sources, _ = node_roles(graph, top_n=1)
        self.assertEqual(sources.iloc[0]["region"], "Sumy Oblast")

    def test_render_figure_with_highlights(self) -> None:
        graph = build_propagation_graph(self.df, max_lag_min=30)
        fig = render_graph_figure(
            graph, highlight={"Sumy Oblast": ("#D1495B", 1.0), "Kyiv City": ("#F2A541", 0.8)},
            min_weight=1, show_labels=True, dark=True,
        )
        self.assertEqual(len(fig.axes), 1)
        plt.close(fig)

    def test_unknown_region_layout_falls_back(self) -> None:
        graph = nx.DiGraph()
        graph.add_edge("Kyiv City", "Atlantis", weight=1, median_lag_min=5.0, mean_lag_min=5.0,
                       probability=0.5)
        fig = render_graph_figure(graph)
        plt.close(fig)


class PredictionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = nx.DiGraph()
        self.graph.add_node("A", alarms=10)
        self.graph.add_node("B", alarms=8)
        self.graph.add_node("C", alarms=5)
        self.graph.add_edge("A", "B", weight=6, median_lag_min=20.0, mean_lag_min=22.0,
                            probability=0.6)
        self.graph.add_edge("A", "C", weight=2, median_lag_min=40.0, mean_lag_min=40.0,
                            probability=0.2)
        self.graph.add_edge("B", "C", weight=4, median_lag_min=15.0, mean_lag_min=15.0,
                            probability=0.5)
        self.cutoff = _ts("2026-01-01 12:00")

    def test_single_hop_ranking_and_eta(self) -> None:
        pred = predict_cascade(self.graph, [("A", 5.0)], self.cutoff, top_k=5, hops=1)
        self.assertEqual(list(pred.columns), list(PREDICTION_COLUMNS))
        self.assertEqual(pred.iloc[0]["region"], "B")
        self.assertAlmostEqual(pred.iloc[0]["probability"], 0.6)
        self.assertAlmostEqual(pred.iloc[0]["eta_min"], 15.0)  # 20 min lag - 5 min elapsed
        self.assertEqual(pred.iloc[0]["eta_time"], self.cutoff + pd.Timedelta(minutes=15))
        self.assertNotIn("A", set(pred["region"]))

    def test_overdue_trigger_decays(self) -> None:
        fresh = predict_cascade(self.graph, [("A", 0.0)], self.cutoff, decay_min=30)
        stale = predict_cascade(self.graph, [("A", 80.0)], self.cutoff, decay_min=30)
        p_fresh = fresh.set_index("region").loc["B", "probability"]
        p_stale = stale.set_index("region").loc["B", "probability"]
        self.assertLess(p_stale, p_fresh)
        self.assertEqual(stale.set_index("region").loc["B", "eta_min"], 0.0)

    def test_two_hops_and_noisy_or(self) -> None:
        one = predict_cascade(self.graph, [("A", 0.0)], self.cutoff, hops=1)
        two = predict_cascade(self.graph, [("A", 0.0)], self.cutoff, hops=2)
        p1 = one.set_index("region").loc["C", "probability"]
        p2 = two.set_index("region").loc["C", "probability"]
        self.assertGreater(p2, p1)
        self.assertLessEqual(p2, 1.0)
        self.assertEqual(two.set_index("region").loc["C", "paths"], 2)

    def test_triggers_without_history_and_empty(self) -> None:
        self.assertTrue(predict_cascade(self.graph, [("Z", 0.0)], self.cutoff).empty)
        self.assertTrue(predict_cascade(self.graph, [], self.cutoff).empty)
        both = predict_cascade(self.graph, [("A", 0.0), ("B", 0.0)], self.cutoff)
        self.assertEqual(list(both["region"]), ["C"])  # A and B are triggers, never predicted


class HelperTests(unittest.TestCase):
    def test_resolve_source(self) -> None:
        self.assertEqual(resolve_source("synthetic", "x.csv"), ("synthetic", None))
        used, reason = resolve_source("csv", "definitely/missing.csv")
        self.assertEqual(used, "synthetic")
        self.assertIn("missing.csv", reason or "")

    def test_default_cutoff_inside_range(self) -> None:
        df = generate_synthetic_alarms(200, seed=1)
        point = pd.Timestamp(default_cutoff(df), tz="UTC")
        self.assertGreater(point, df["start_time"].min())
        self.assertLess(point, df["start_time"].max())


class AppSmokeTests(unittest.TestCase):
    """Headless end-to-end checks of the Streamlit page."""

    def _app(self) -> AppTest:
        at = AppTest.from_file(str(APP_PATH), default_timeout=120)
        at.session_state["source"] = "synthetic"
        at.session_state["synthetic_records"] = 600
        at.session_state["synthetic_seed"] = 7
        return at

    def test_synthetic_mode_renders_all_tabs(self) -> None:
        at = self._app().run()
        self.assertFalse(at.exception, msg=str(at.exception))
        self.assertTrue(any("Synthetic mode" in str(w.value) or "synthetic" in str(w.value).lower()
                            for w in at.warning))
        self.assertGreaterEqual(len(at.metric), 4)
        self.assertTrue(at.dataframe)  # recent alarms / corridors / predictions tables
        self.assertFalse(at.error)

    def test_cutoff_before_data_shows_empty_state(self) -> None:
        # AppTest in this Streamlit build does not apply ``date_input.set_value``
        # (time_input/slider/buttons do), so seed the widget state before the
        # first run and pin the dataset signature so the auto-reset is skipped.
        at = self._app()
        at.session_state["_dataset_signature"] = "synthetic|data/alarms.csv|600|7|2026-01-01|600"
        at.session_state["cutoff_date"] = date(2000, 1, 1)
        at.session_state["cutoff_time"] = time(0, 0)
        at.run()
        self.assertFalse(at.exception, msg=str(at.exception))
        self.assertTrue(any("No alarms recorded before T" in str(i.value) for i in at.info))
        self.assertTrue(any("before the first recorded alarm" in str(w.value) for w in at.warning))
        self.assertEqual(len(at.metric), 0)
        # The one-click recovery button must bring the data back.
        at.button(key="empty_reset_overview").click().run()
        self.assertFalse(at.exception, msg=str(at.exception))
        self.assertGreaterEqual(len(at.metric), 4)

    def test_filter_that_hides_everything_shows_empty_state(self) -> None:
        # Pick the threat type whose first alarm comes latest, and place T one
        # minute before it: other alarms are visible, but none of that type.
        history = generate_synthetic_alarms(600, start_date="2026-01-01", seed=7)
        first_seen = history.groupby("threat_type")["start_time"].min()
        latest_type = str(first_seen.idxmax())
        cutoff = (first_seen.max() - pd.Timedelta(minutes=1)).tz_convert("UTC")
        self.assertGreater(int((history["start_time"] < cutoff).sum()), 0)

        at = self._app()
        at.session_state["_dataset_signature"] = "synthetic|data/alarms.csv|600|7|2026-01-01|600"
        at.session_state["cutoff_date"] = cutoff.date()
        at.session_state["cutoff_time"] = cutoff.to_pydatetime().time().replace(tzinfo=None)
        at.session_state["threat_type_filter"] = [latest_type]
        at.run()
        self.assertFalse(at.exception, msg=str(at.exception))
        self.assertTrue(any("threat filter hides" in str(i.value) for i in at.info))
        self.assertEqual(len(at.metric), 0)

    def test_nudge_buttons_move_cutoff(self) -> None:
        at = self._app().run()
        before = at.session_state["cutoff_date"]
        at.button(key="nudge_p1d").click().run()
        self.assertFalse(at.exception, msg=str(at.exception))
        self.assertEqual(at.session_state["cutoff_date"], before + pd.Timedelta(days=1))
        at.time_input(key="cutoff_time").set_value(time(3, 15)).run()
        self.assertFalse(at.exception, msg=str(at.exception))

    def test_missing_csv_falls_back_with_banner(self) -> None:
        at = AppTest.from_file(str(APP_PATH), default_timeout=120)
        at.session_state["source"] = "csv"
        at.session_state["csv_path"] = "data/this_file_does_not_exist.csv"
        at.session_state["synthetic_records"] = 400
        at.run()
        self.assertFalse(at.exception, msg=str(at.exception))
        self.assertTrue(any("fallback" in str(w.value).lower() or "No CSV found" in str(w.value)
                            for w in at.warning))
        self.assertGreaterEqual(len(at.metric), 4)

    def test_real_csv_is_used_and_filtered(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "alarms.csv"
            generate_synthetic_alarms(300, seed=3).to_csv(path, index=False)
            at = AppTest.from_file(str(APP_PATH), default_timeout=120)
            at.session_state["source"] = "csv"
            at.session_state["csv_path"] = str(path)
            at.run()
            self.assertFalse(at.exception, msg=str(at.exception))
            self.assertFalse(any("No CSV found" in str(w.value) for w in at.warning))
            self.assertGreaterEqual(len(at.metric), 4)


if __name__ == "__main__":
    unittest.main()
