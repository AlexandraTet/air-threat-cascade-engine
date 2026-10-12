"""Tests for ``src.visualizer``. Run with ``python -m unittest discover tests``."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from src.loader import UKRAINIAN_REGIONS
from src.visualizer import (
    CRIMEA_OUTLINE,
    DARK_STYLE,
    LIGHT_STYLE,
    REGION_CENTROIDS,
    UKRAINE_BBOX,
    UKRAINE_OUTLINE,
    WAVE_COLORS,
    _load_base_map_cached,
    layout_positions,
    load_base_map,
    map_style,
    match_region_name,
    polygon_centroid,
    render_graph_figure,
    render_wave_map_figure,
    short_region_name,
)


def _square(x0: float, y0: float, size: float) -> list[list[float]]:
    return [[x0, y0], [x0 + size, y0], [x0 + size, y0 + size], [x0, y0 + size], [x0, y0]]


class GeographyTests(unittest.TestCase):
    def test_centroids_cover_all_regions_inside_bbox(self) -> None:
        self.assertEqual(set(REGION_CENTROIDS), set(UKRAINIAN_REGIONS))
        lon_min, lon_max, lat_min, lat_max = UKRAINE_BBOX
        for region, (lon, lat) in REGION_CENTROIDS.items():
            self.assertTrue(lon_min <= lon <= lon_max, region)
            self.assertTrue(lat_min <= lat <= lat_max, region)
        # Sanity: west/east and north/south ordering of a few well-known oblasts.
        self.assertLess(REGION_CENTROIDS["Lviv Oblast"][0], REGION_CENTROIDS["Kyiv City"][0])
        self.assertLess(REGION_CENTROIDS["Kyiv City"][0], REGION_CENTROIDS["Kharkiv Oblast"][0])
        self.assertGreater(REGION_CENTROIDS["Chernihiv Oblast"][1], REGION_CENTROIDS["Odesa Oblast"][1])

    def test_schematic_outline_is_closed_and_plausible(self) -> None:
        for ring in (UKRAINE_OUTLINE, CRIMEA_OUTLINE):
            self.assertEqual(ring[0], ring[-1])
            arr = np.asarray(ring)
            self.assertTrue((arr[:, 0] >= 21.5).all() and (arr[:, 0] <= 40.5).all())
            self.assertTrue((arr[:, 1] >= 44.0).all() and (arr[:, 1] <= 52.6).all())
        # Every centroid should fall inside the mainland outline's bounding box.
        arr = np.asarray(UKRAINE_OUTLINE)
        for lon, lat in REGION_CENTROIDS.values():
            self.assertTrue(arr[:, 0].min() <= lon <= arr[:, 0].max())
            self.assertTrue(arr[:, 1].min() <= lat <= arr[:, 1].max())

    def test_polygon_centroid(self) -> None:
        cx, cy = polygon_centroid(np.asarray(_square(10.0, 20.0, 2.0)))
        self.assertAlmostEqual(cx, 11.0)
        self.assertAlmostEqual(cy, 21.0)
        cx, cy = polygon_centroid(np.asarray([[1.0, 1.0], [2.0, 2.0]]))  # degenerate
        self.assertAlmostEqual(cx, 1.5)

    def test_match_region_name(self) -> None:
        cases = {
            "Kharkivska": "Kharkiv Oblast", "Kharkov Oblast": "Kharkiv Oblast", "Odessa": "Odesa Oblast",
            "Kyiv": "Kyiv City", "Kyiv City": "Kyiv City", "Kyivska oblast": "Kyiv Oblast",
            "Dnipropetrovska": "Dnipropetrovsk Oblast", "Ivano-Frankivska": "Ivano-Frankivsk Oblast",
            "Transcarpathia": "Zakarpattia Oblast", "Autonomous Republic of Crimea": None,
            "Sevastopol": None, "": None, None: None,
        }
        for raw, expected in cases.items():
            self.assertEqual(match_region_name(raw), expected, raw)

    def test_short_names_and_styles(self) -> None:
        self.assertEqual(short_region_name("Kyiv City"), "Kyiv")
        self.assertEqual(short_region_name("Kyiv Oblast"), "Kyiv reg.")
        self.assertEqual(short_region_name("Lviv Oblast"), "Lviv")
        self.assertIs(map_style(False), LIGHT_STYLE)
        self.assertIs(map_style(True), DARK_STYLE)
        self.assertNotEqual(LIGHT_STYLE.text, LIGHT_STYLE.halo)
        self.assertEqual(set(WAVE_COLORS), {"fast", "slow", "unknown"})


class BaseMapTests(unittest.TestCase):
    def setUp(self) -> None:
        _load_base_map_cached.cache_clear()

    def tearDown(self) -> None:
        _load_base_map_cached.cache_clear()

    def test_schematic_fallback_when_no_file(self) -> None:
        base = load_base_map(search_paths=["definitely/missing.geojson"])
        self.assertTrue(base.is_schematic)
        self.assertEqual(len(base.polygons), 2)
        self.assertEqual(dict(base.centroids), dict(REGION_CENTROIDS))
        self.assertIn("offline fallback", base.description)

    def test_geojson_file_overrides_centroids(self) -> None:
        features = [
            {"type": "Feature", "properties": {"name:en": "Kharkivska"},
             "geometry": {"type": "Polygon", "coordinates": [_square(35.0, 49.0, 2.0)]}},
            {"type": "Feature", "properties": {"NAME_1": "Kyiv"},
             "geometry": {"type": "MultiPolygon",
                          "coordinates": [[_square(30.0, 50.0, 1.0)], [_square(31.5, 50.5, 0.2)]]}},
            {"type": "Feature", "properties": {"name": "Crimea"},
             "geometry": {"type": "Polygon", "coordinates": [_square(33.0, 44.5, 1.0)]}},
            {"type": "Feature", "properties": {"name": "No geometry"}, "geometry": None},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ukraine.geojson"
            path.write_text(json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8")
            base = load_base_map(search_paths=[str(path)])
        self.assertFalse(base.is_schematic)
        self.assertTrue(base.source.endswith("ukraine.geojson"))
        self.assertEqual(base.matched_regions, ("Kharkiv Oblast", "Kyiv City"))
        self.assertEqual(base.centroids["Kharkiv Oblast"], (36.0, 50.0))
        self.assertEqual(base.centroids["Kyiv City"], (30.5, 50.5))  # largest polygon wins
        self.assertEqual(base.centroids["Lviv Oblast"], REGION_CENTROIDS["Lviv Oblast"])  # fallback
        self.assertEqual(len(base.polygons), 4)  # 1 + 2 + 1 rings; empty geometry skipped
        self.assertIn("2/25", base.description)

    def test_corrupt_file_degrades_to_schematic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.geojson"
            path.write_text("{not json", encoding="utf-8")
            with self.assertLogs("src.visualizer", level="WARNING"):
                base = load_base_map(search_paths=[str(path)])
        self.assertTrue(base.is_schematic)

    def test_layout_positions_use_base_map_and_fallback(self) -> None:
        graph = nx.DiGraph()
        graph.add_edge("Kyiv City", "Atlantis")
        pos = layout_positions(graph)
        self.assertEqual(tuple(pos["Kyiv City"]), REGION_CENTROIDS["Kyiv City"])
        self.assertIn("Atlantis", pos)


class RenderTests(unittest.TestCase):
    def _graphs(self) -> dict[str, nx.DiGraph]:
        fast = nx.DiGraph(source="observed", wave="fast")
        fast.add_node("Sumy Oblast", alarms=5)
        fast.add_node("Poltava Oblast", alarms=3)
        fast.add_edge("Sumy Oblast", "Poltava Oblast", weight=4, median_lag_min=6.0, mean_lag_min=6.0,
                      probability=0.8)
        slow = nx.DiGraph(source="prior", wave="slow")
        slow.add_edge("Kherson Oblast", "Mykolaiv Oblast", weight=1, median_lag_min=35.0, mean_lag_min=35.0,
                      probability=0.45, prior=True)
        return {"fast": fast, "slow": slow}

    def test_wave_map_renders_with_and_without_map(self) -> None:
        graphs = self._graphs()
        predictions = pd.DataFrame({"region": ["Poltava Oblast"], "probability": [0.8], "eta_min": [6.0]})
        for dark in (False, True):
            for show_map in (True, False):
                fig = render_wave_map_figure(
                    graphs, layers=["fast", "slow"], activity=pd.Series({"Sumy Oblast": 5}),
                    active_regions=["Sumy Oblast"], trigger="Sumy Oblast", predictions=predictions,
                    show_edge_labels=True, dark=dark, show_map=show_map, title="t",
                )
                self.assertEqual(len(fig.axes), 1)
                ax = fig.axes[0]
                texts = [t.get_text() for t in ax.texts]
                self.assertIn("Sumy", texts)
                self.assertTrue(any(t.startswith("+6 min") for t in texts), texts)
                # Node labels are bold with a halo for contrast.
                label = next(t for t in ax.texts if t.get_text() == "Sumy")
                self.assertEqual(label.get_fontweight(), "bold")
                self.assertTrue(label.get_path_effects())
                if show_map:
                    self.assertGreaterEqual(len([p for p in ax.patches if p.get_zorder() < 1]), 2)
                    self.assertGreater(ax.get_aspect(), 1.0)
                plt.close(fig)

    def test_graph_figure_variants(self) -> None:
        graph = self._graphs()["fast"]
        fig = render_graph_figure(graph, highlight={"Sumy Oblast": ("#D1495B", 1.0),
                                                    "Poltava Oblast": ("#F2A541", 0.9)}, title="g")
        legend_labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        self.assertIn("Active trigger at T", legend_labels)
        self.assertIn("Predicted next target", legend_labels)
        plt.close(fig)
        empty = render_graph_figure(nx.DiGraph(), dark=True, show_map=False)
        self.assertTrue(any("No alarms" in t.get_text() for t in empty.axes[0].texts))
        plt.close(empty)


if __name__ == "__main__":
    unittest.main()
