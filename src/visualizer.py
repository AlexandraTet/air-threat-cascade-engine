"""Map-aware rendering of the regional air-threat propagation graph.

This module owns everything visual about the spatiotemporal graph so that
``src/dashboard.py`` only decides *what* to show:

* **Geography.** Nodes are bound to the geographic centroids of Ukrainian
  oblasts. Real centroids and boundaries are read from a local GeoJSON or
  shapefile when one is available (``data/geo/`` or the
  ``UKRAINE_GEOJSON_PATH`` environment variable), through ``geopandas`` if
  it is installed and through the standard-library JSON parser otherwise.
  Without any file the module falls back to :data:`REGION_CENTROIDS` and a
  bundled schematic outline of Ukraine, so the map always renders offline.
* **Readability.** Bold region labels with a contrasting halo, saturated
  fast-wave (red) and slow-wave (blue) edges with thicker strokes, boxed ETA
  tags, framed legends and dark node outlines; a dark variant of the style is
  used when the Streamlit theme is dark.
* **Compatibility.** Pure ``networkx`` + ``matplotlib``; no geospatial
  dependency is required.

Public entry points:
:func:`render_wave_map_figure` (layered fast/slow map with trigger highlight),
:func:`render_graph_figure` (single mixed graph), :func:`load_base_map`,
:func:`layout_positions` and :func:`short_region_name`.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # Headless backend; must precede pyplot import.

import matplotlib.patheffects as patheffects  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Polygon as MplPolygon  # noqa: E402

from src.loader import UKRAINIAN_REGIONS  # noqa: E402

__all__ = [
    "BaseMap",
    "CRIMEA_OUTLINE",
    "DARK_STYLE",
    "GEO_SEARCH_PATHS",
    "LIGHT_STYLE",
    "MapStyle",
    "REGION_CENTROIDS",
    "UKRAINE_BBOX",
    "UKRAINE_OUTLINE",
    "WAVE_COLORS",
    "draw_base_map",
    "layout_positions",
    "load_base_map",
    "map_style",
    "match_region_name",
    "polygon_centroid",
    "render_graph_figure",
    "render_wave_map_figure",
    "short_region_name",
]

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #
WAVE_COLORS: Final[Mapping[str, str]] = {
    "fast": "#C8102E",   # saturated red: ballistic / aviation
    "slow": "#0B5FA5",   # saturated blue: UAV / Shahed
    "unknown": "#6C757D",
}
"""Wave-class colours shared by the map and the dashboard charts."""


@dataclass(frozen=True)
class MapStyle:
    """Colours and strokes for one theme variant.

    Attributes:
        name: ``"light"`` or ``"dark"``.
        text: Label colour.
        halo: Outline colour drawn behind labels for contrast.
        land: Fill of the Ukraine land polygons.
        land_edge: Outline of the land polygons.
        oblast_edge: Oblast boundary colour when boundaries are available.
        node_fill: Neutral node colour.
        node_edge: Node outline colour.
        active: Fill for regions active at ``T``.
        trigger: Fill for the selected trigger star.
        target: Fill for predicted targets and ETA tags.
        route: Colour of trigger → target arrows.
        prior_edge: Colour of prior-topology (dashed) edges.
        legend_face: Legend background.
        eta_text: Text colour inside ETA tags.
    """

    name: str
    text: str
    halo: str
    land: str
    land_edge: str
    oblast_edge: str
    node_fill: str
    node_edge: str
    active: str = "#D7263D"
    trigger: str = "#FFC43D"
    target: str = "#F28C28"
    route: str = "#E07B00"
    prior_edge: str = "#5B6B7C"
    legend_face: str = "#FFFFFF"
    eta_text: str = "#1A1A1A"


LIGHT_STYLE: Final[MapStyle] = MapStyle(
    name="light", text="#111111", halo="#FFFFFF", land="#E9EDF1", land_edge="#8C98A4",
    oblast_edge="#B4BEC8", node_fill="#C9D1D9", node_edge="#2B2B2B", legend_face="#FFFFFF",
)
DARK_STYLE: Final[MapStyle] = MapStyle(
    name="dark", text="#F5F5F5", halo="#101418", land="#2B3139", land_edge="#8A95A1",
    oblast_edge="#4A535E", node_fill="#7A8794", node_edge="#F0F0F0", legend_face="#1E232A",
    eta_text="#1A1A1A",
)


def map_style(dark: bool) -> MapStyle:
    """Return the style for the active theme."""
    return DARK_STYLE if dark else LIGHT_STYLE


# --------------------------------------------------------------------------- #
# Geography: centroids, bounding box, schematic outline
# --------------------------------------------------------------------------- #
REGION_CENTROIDS: Final[Mapping[str, tuple[float, float]]] = {
    "Volyn Oblast": (25.00, 51.10),
    "Rivne Oblast": (26.45, 51.00),
    "Zhytomyr Oblast": (28.45, 50.60),
    "Kyiv Oblast": (30.25, 50.05),
    "Kyiv City": (30.52, 50.45),
    "Chernihiv Oblast": (31.90, 51.40),
    "Sumy Oblast": (34.30, 51.00),
    "Kharkiv Oblast": (36.50, 49.60),
    "Luhansk Oblast": (39.10, 48.90),
    "Donetsk Oblast": (37.85, 48.00),
    "Zaporizhzhia Oblast": (35.60, 47.40),
    "Kherson Oblast": (33.30, 46.60),
    "Mykolaiv Oblast": (31.80, 47.20),
    "Odesa Oblast": (30.00, 46.60),
    "Dnipropetrovsk Oblast": (34.80, 48.40),
    "Kirovohrad Oblast": (31.90, 48.60),
    "Poltava Oblast": (34.00, 49.60),
    "Cherkasy Oblast": (31.60, 49.20),
    "Vinnytsia Oblast": (28.50, 49.00),
    "Khmelnytskyi Oblast": (26.90, 49.40),
    "Ternopil Oblast": (25.60, 49.40),
    "Lviv Oblast": (24.00, 49.60),
    "Ivano-Frankivsk Oblast": (24.60, 48.70),
    "Zakarpattia Oblast": (23.20, 48.40),
    "Chernivtsi Oblast": (25.90, 48.20),
}
"""Approximate geographic centroids (longitude, latitude) of each oblast.

Used as the node layout whenever no boundary file is available, and as the
fallback for any region a boundary file does not name."""

UKRAINE_BBOX: Final[tuple[float, float, float, float]] = (22.0, 40.3, 44.3, 52.5)
"""``(lon_min, lon_max, lat_min, lat_max)`` used for fixed map extents."""

UKRAINE_OUTLINE: Final[Sequence[tuple[float, float]]] = (
    (22.15, 48.40), (22.65, 49.10), (23.10, 49.60), (23.55, 50.10), (23.95, 50.45),
    (24.10, 50.85), (23.65, 51.35), (23.60, 51.60), (24.40, 51.90), (25.50, 51.95),
    (26.50, 51.85), (27.50, 51.65), (28.60, 51.55), (29.30, 51.45), (30.30, 51.35),
    (30.55, 51.55), (31.00, 52.10), (31.80, 52.12), (32.50, 52.30), (33.50, 52.35),
    (34.10, 51.90), (34.40, 51.30), (35.30, 51.05), (35.55, 50.50), (36.30, 50.35),
    (37.40, 50.40), (38.10, 50.00), (38.30, 49.65), (39.60, 49.70), (40.10, 49.40),
    (40.20, 49.00), (39.80, 48.60), (40.00, 48.30), (39.90, 47.95), (39.30, 47.85),
    (38.70, 47.65), (38.30, 47.10), (37.50, 47.10), (36.80, 46.75), (35.90, 46.60),
    (35.20, 46.45), (34.80, 46.15), (33.65, 46.15), (33.00, 46.00), (32.60, 46.40),
    (32.00, 46.55), (31.60, 46.60), (31.00, 46.60), (30.75, 46.45), (30.45, 46.10),
    (30.10, 45.85), (29.70, 45.55), (29.60, 45.25), (28.90, 45.30), (28.20, 45.45),
    (28.50, 45.95), (29.00, 46.40), (29.90, 46.40), (29.60, 46.90), (29.30, 47.30),
    (29.00, 47.80), (28.40, 48.15), (27.80, 48.45), (27.30, 48.40), (26.90, 48.35),
    (26.60, 48.25), (26.30, 48.20), (25.90, 47.95), (25.20, 47.95), (24.90, 47.75),
    (24.30, 47.85), (23.60, 48.00), (23.10, 47.95), (22.60, 48.05), (22.15, 48.40),
)
"""Schematic mainland outline (longitude, latitude), used offline."""

CRIMEA_OUTLINE: Final[Sequence[tuple[float, float]]] = (
    (33.60, 46.10), (34.50, 45.95), (35.00, 45.65), (35.40, 45.30), (35.80, 45.40),
    (36.60, 45.45), (36.40, 45.05), (35.40, 44.95), (34.80, 44.80), (34.20, 44.45),
    (33.60, 44.45), (33.40, 44.65), (33.50, 45.15), (32.50, 45.40), (33.00, 45.80),
    (33.60, 46.10),
)
"""Schematic outline of the Crimean peninsula (temporarily occupied)."""

GEO_SEARCH_PATHS: Final[Sequence[str]] = (
    "data/geo/ukraine_oblasts.geojson",
    "data/geo/ukraine_adm1.geojson",
    "data/geo/ukraine.geojson",
    "data/geo/ukraine_oblasts.shp",
    "assets/geo/ukraine_oblasts.geojson",
)
"""Relative paths probed for real administrative boundaries."""

_NAME_KEYS: Final[Sequence[str]] = (
    "name:en", "name_en", "NAME_EN", "NAME_1", "ADM1_EN", "shapeName", "NAME", "name", "region",
    "oblast", "Name",
)
_NAME_ALIASES: Final[Mapping[str, str]] = {
    "odessa": "Odesa Oblast", "dnipro": "Dnipropetrovsk Oblast", "zaporozh": "Zaporizhzhia Oblast",
    "nikolaev": "Mykolaiv Oblast", "transcarpath": "Zakarpattia Oblast", "khmelnytsk": "Khmelnytskyi Oblast",
    "kropyvnytsk": "Kirovohrad Oblast", "kirovograd": "Kirovohrad Oblast", "lugansk": "Luhansk Oblast",
    "chernigov": "Chernihiv Oblast", "kharkov": "Kharkiv Oblast", "lvov": "Lviv Oblast",
    "rovno": "Rivne Oblast", "vinnitsa": "Vinnytsia Oblast", "zhitomir": "Zhytomyr Oblast",
    "ternopol": "Ternopil Oblast", "cherkassy": "Cherkasy Oblast", "chernovtsy": "Chernivtsi Oblast",
}


def match_region_name(name: object) -> Optional[str]:
    """Map a boundary-file feature name to a canonical region.

    Handles English and transliterated spellings (``"Kharkivska"``,
    ``"Odessa"``), and distinguishes Kyiv city from Kyiv oblast.

    Args:
        name: Feature property value.

    Returns:
        A member of :data:`src.loader.UKRAINIAN_REGIONS` or ``None`` (for
        example Crimea or Sevastopol, which are not alarm regions).
    """
    if name is None:
        return None
    text = str(name).strip().lower()
    if not text:
        return None
    if any(k in text for k in ("kyiv", "kiev", "київ")):
        is_city = any(k in text for k in ("city", "misto", "місто", "m.")) or text in ("kyiv", "kiev", "київ")
        return "Kyiv City" if is_city else "Kyiv Oblast"
    for region in UKRAINIAN_REGIONS:
        stem = region.split()[0].lower()
        if stem in text:
            return region
    for alias, region in _NAME_ALIASES.items():
        if alias in text:
            return region
    return None


def polygon_centroid(ring: np.ndarray) -> tuple[float, float]:
    """Area-weighted centroid of a closed or open ring via the shoelace formula.

    Args:
        ring: ``(N, 2)`` array of ``(lon, lat)`` vertices.

    Returns:
        ``(lon, lat)`` centroid; the vertex mean for degenerate rings.
    """
    pts = np.asarray(ring, dtype=float)[:, :2]
    if len(pts) < 3:
        return float(pts[:, 0].mean()), float(pts[:, 1].mean())
    x, y = pts[:, 0], pts[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    area = cross.sum() / 2.0
    if abs(area) < 1e-12:
        return float(x.mean()), float(y.mean())
    cx = float(((x + x1) * cross).sum() / (6.0 * area))
    cy = float(((y + y1) * cross).sum() / (6.0 * area))
    return cx, cy


def _ring_area(ring: np.ndarray) -> float:
    pts = np.asarray(ring, dtype=float)
    x, y = pts[:, 0], pts[:, 1]
    return abs(float((x * np.roll(y, -1) - np.roll(x, -1) * y).sum()) / 2.0)


def _rings_from_geometry(geometry: Mapping[str, Any]) -> list[np.ndarray]:
    """Exterior rings of a GeoJSON geometry as ``(N, 2)`` arrays."""
    gtype = geometry.get("type")
    coords = geometry.get("coordinates")
    if gtype == "Polygon" and coords:
        return [np.asarray(coords[0], dtype=float)[:, :2]]
    if gtype == "MultiPolygon" and coords:
        return [np.asarray(poly[0], dtype=float)[:, :2] for poly in coords if poly]
    if gtype == "GeometryCollection":
        rings: list[np.ndarray] = []
        for sub in geometry.get("geometries", []):
            rings.extend(_rings_from_geometry(sub))
        return rings
    return []


@dataclass(frozen=True)
class BaseMap:
    """Boundary polygons and node anchors used to draw the underlay.

    Attributes:
        polygons: Land outlines (``(N, 2)`` lon/lat arrays).
        region_polygons: Oblast polygons keyed by canonical region when the
            source file provides them; empty for the schematic fallback.
        centroids: Node anchor per region (``lon, lat``). Always complete:
            regions missing from the file use :data:`REGION_CENTROIDS`.
        source: ``"schematic"``, ``"geojson:<path>"`` or ``"geopandas:<path>"``.
        matched_regions: Regions whose polygon came from the file.
    """

    polygons: tuple[np.ndarray, ...]
    region_polygons: Mapping[str, tuple[np.ndarray, ...]] = field(default_factory=dict)
    centroids: Mapping[str, tuple[float, float]] = field(default_factory=lambda: dict(REGION_CENTROIDS))
    source: str = "schematic"
    matched_regions: tuple[str, ...] = ()

    @property
    def is_schematic(self) -> bool:
        """``True`` when the bundled outline is in use (no boundary file)."""
        return self.source == "schematic"

    @property
    def description(self) -> str:
        """Human-readable provenance line for captions."""
        if self.is_schematic:
            return ("schematic outline with hard-coded oblast centroids (offline fallback) · drop a "
                    "GeoJSON into data/geo/ for real boundaries")
        return (f"{self.source} · {len(self.matched_regions)}/{len(UKRAINIAN_REGIONS)} oblast "
                "boundaries matched; nodes at polygon centroids")


def _schematic_base_map() -> BaseMap:
    return BaseMap(
        polygons=(np.asarray(UKRAINE_OUTLINE, dtype=float), np.asarray(CRIMEA_OUTLINE, dtype=float)),
        region_polygons={},
        centroids=dict(REGION_CENTROIDS),
        source="schematic",
        matched_regions=(),
    )


def _features_from_file(path: Path) -> tuple[list[dict[str, Any]], str]:
    """Load GeoJSON-like features from ``path`` (geopandas if available)."""
    if path.suffix.lower() in {".shp", ".gpkg", ".zip"}:
        import geopandas as gpd  # type: ignore[import-not-found]

        frame = gpd.read_file(path)
        if frame.crs is not None and frame.crs.to_epsg() != 4326:
            frame = frame.to_crs(4326)
        return list(frame.__geo_interface__["features"]), f"geopandas:{path.as_posix()}"
    try:
        import geopandas as gpd  # type: ignore[import-not-found]

        frame = gpd.read_file(path)
        if frame.crs is not None and frame.crs.to_epsg() != 4326:
            frame = frame.to_crs(4326)
        return list(frame.__geo_interface__["features"]), f"geopandas:{path.as_posix()}"
    except ImportError:
        pass
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if data.get("type") == "FeatureCollection":
        return list(data.get("features", [])), f"geojson:{path.as_posix()}"
    if data.get("type") == "Feature":
        return [data], f"geojson:{path.as_posix()}"
    raise ValueError(f"{path} is not a GeoJSON Feature or FeatureCollection")


def _base_map_from_features(features: Sequence[Mapping[str, Any]], source: str) -> BaseMap:
    polygons: list[np.ndarray] = []
    region_polygons: dict[str, list[np.ndarray]] = {}
    centroids: dict[str, tuple[float, float]] = dict(REGION_CENTROIDS)
    for feature in features:
        geometry = feature.get("geometry") or {}
        rings = _rings_from_geometry(geometry)
        if not rings:
            continue
        polygons.extend(rings)
        props = feature.get("properties") or {}
        name = next((props[k] for k in _NAME_KEYS if props.get(k)), None)
        region = match_region_name(name)
        if region is None:
            continue
        region_polygons.setdefault(region, []).extend(rings)
    for region, rings in region_polygons.items():
        largest = max(rings, key=_ring_area)
        centroids[region] = polygon_centroid(largest)
    if not polygons:
        raise ValueError("boundary file contains no polygon geometries")
    return BaseMap(
        polygons=tuple(polygons),
        region_polygons={k: tuple(v) for k, v in region_polygons.items()},
        centroids=centroids,
        source=source,
        matched_regions=tuple(sorted(region_polygons)),
    )


@lru_cache(maxsize=8)
def _load_base_map_cached(paths: tuple[str, ...]) -> BaseMap:
    for candidate in paths:
        path = Path(candidate)
        if not path.is_file():
            continue
        try:
            features, source = _features_from_file(path)
            base = _base_map_from_features(features, source)
            logger.info("Base map loaded from %s (%d regions matched)", path, len(base.matched_regions))
            return base
        except Exception as exc:  # noqa: BLE001 - any failure must degrade to the schematic map
            logger.warning("Could not use boundary file %s (%s); trying next source", path, exc)
    logger.info("No boundary file found; using schematic Ukraine outline")
    return _schematic_base_map()


def load_base_map(search_paths: Optional[Sequence[str]] = None) -> BaseMap:
    """Load real boundaries if a file is available, else the schematic outline.

    Resolution order: ``UKRAINE_GEOJSON_PATH`` environment variable, then
    ``search_paths`` (default :data:`GEO_SEARCH_PATHS`, relative to the
    current working directory and to the project root). Results are cached.

    Args:
        search_paths: Candidate boundary files to probe in order.

    Returns:
        A :class:`BaseMap`; never raises.
    """
    candidates: list[str] = []
    env_path = os.environ.get("UKRAINE_GEOJSON_PATH")
    if env_path:
        candidates.append(env_path)
    project_root = Path(__file__).resolve().parent.parent
    for rel in (search_paths if search_paths is not None else GEO_SEARCH_PATHS):
        candidates.append(rel)
        candidates.append(str(project_root / rel))
    return _load_base_map_cached(tuple(dict.fromkeys(candidates)))


# --------------------------------------------------------------------------- #
# Layout helpers
# --------------------------------------------------------------------------- #
def short_region_name(region: str) -> str:
    """Compact label for graph nodes (``"Kyiv Oblast"`` -> ``"Kyiv reg."``)."""
    if region == "Kyiv City":
        return "Kyiv"
    if region == "Kyiv Oblast":
        return "Kyiv reg."
    return region.replace(" Oblast", "")


def layout_positions(graph: nx.Graph, base_map: Optional[BaseMap] = None) -> dict[str, tuple[float, float]]:
    """Geographic node layout with a spring-layout fallback for unknown nodes.

    Args:
        graph: Any graph whose nodes are region names.
        base_map: Source of centroids; defaults to :data:`REGION_CENTROIDS`.

    Returns:
        Mapping of node to ``(lon, lat)`` (unknown nodes get synthetic
        coordinates anchored to the known ones).
    """
    anchors = base_map.centroids if base_map is not None else REGION_CENTROIDS
    known = {n: tuple(anchors[n]) for n in graph.nodes if n in anchors}
    if len(known) == graph.number_of_nodes():
        return known
    if known:
        return nx.spring_layout(graph, pos=known, fixed=list(known), seed=42)
    return nx.spring_layout(graph, seed=42)


def _halo(style: MapStyle, width: float = 2.8) -> list[patheffects.AbstractPathEffect]:
    return [patheffects.withStroke(linewidth=width, foreground=style.halo)]


def _new_axes(style: MapStyle, title: str, figsize: tuple[float, float]) -> tuple[Figure, Any]:
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    ax.set_axis_off()
    if title:
        ax.set_title(title, color=style.text, fontsize=12.5, fontweight="bold", loc="left")
    return fig, ax


def draw_base_map(
    ax: Any,
    base_map: BaseMap,
    style: MapStyle,
    *,
    shade: Optional[Mapping[str, str]] = None,
    extent: tuple[float, float, float, float] = UKRAINE_BBOX,
) -> None:
    """Draw the land underlay and fix a geographic aspect ratio and extent.

    Args:
        ax: Matplotlib axes.
        base_map: Polygons to draw.
        style: Theme style.
        shade: Optional ``region -> colour`` tints applied to oblast polygons
            (only when the base map has per-region polygons).
        extent: ``(lon_min, lon_max, lat_min, lat_max)``.
    """
    for ring in base_map.polygons:
        ax.add_patch(MplPolygon(ring, closed=True, facecolor=style.land, edgecolor=style.land_edge,
                                linewidth=1.1, zorder=0.5, joinstyle="round"))
    if base_map.region_polygons:
        for region, rings in base_map.region_polygons.items():
            colour = (shade or {}).get(region)
            for ring in rings:
                ax.add_patch(MplPolygon(
                    ring, closed=True, facecolor=colour if colour else "none",
                    alpha=0.28 if colour else 1.0, edgecolor=style.oblast_edge, linewidth=0.6, zorder=0.6,
                ))
    elif base_map.is_schematic and len(base_map.polygons) > 1:
        # Crimea is drawn hatched to mark it as occupied and outside the alarm regions.
        ax.add_patch(MplPolygon(base_map.polygons[1], closed=True, facecolor="none",
                                edgecolor=style.land_edge, hatch="///", linewidth=0.8, zorder=0.7))
    lon_min, lon_max, lat_min, lat_max = extent
    ax.set_xlim(lon_min - 0.3, lon_max + 0.3)
    ax.set_ylim(lat_min - 0.3, lat_max + 0.3)
    mean_lat = math.radians((lat_min + lat_max) / 2.0)
    ax.set_aspect(1.0 / math.cos(mean_lat))


def _edge_rgba(colour: str, alphas: Sequence[float]) -> list[tuple[float, float, float, float]]:
    r, g, b = to_rgb(colour)
    return [(r, g, b, float(a)) for a in alphas]


# Label nudges (degrees lon, lat) for centroids that sit close together, so
# neighbouring names do not overprint each other on the map.
_LABEL_OFFSETS: Final[Mapping[str, tuple[float, float]]] = {
    "Ternopil Oblast": (-0.15, -0.30),
    "Khmelnytskyi Oblast": (0.25, 0.30),
    "Kyiv City": (0.0, 0.30),
    "Kyiv Oblast": (0.0, -0.32),
    "Ivano-Frankivsk Oblast": (0.0, -0.30),
    "Chernivtsi Oblast": (0.0, -0.28),
    "Lviv Oblast": (0.0, 0.28),
    "Zakarpattia Oblast": (0.0, -0.28),
}


def _draw_labels(ax: Any, pos: Mapping[str, tuple[float, float]], nodes: Sequence[str],
                 style: MapStyle, *, font_size: float = 8.8) -> None:
    for node in nodes:
        if node not in pos:
            continue
        x, y = pos[node]
        dx, dy = _LABEL_OFFSETS.get(node, (0.0, 0.0))
        ax.text(x + dx, y + dy, short_region_name(node), ha="center", va="center", fontsize=font_size,
                fontweight="bold", color=style.text, zorder=6, path_effects=_halo(style))


def _draw_eta_tags(ax: Any, pos: Mapping[str, tuple[float, float]],
                   predicted: Mapping[str, tuple[float, float]], style: MapStyle) -> None:
    for target, (_, eta) in predicted.items():
        if target not in pos:
            continue
        x, y = pos[target]
        ax.annotate(
            f"+{eta:.0f} min", (x, y), xytext=(0, 15), textcoords="offset points", ha="center",
            va="bottom", fontsize=8, fontweight="bold", color=style.eta_text, zorder=7,
            bbox={"boxstyle": "round,pad=0.25,rounding_size=0.4", "facecolor": style.target,
                  "edgecolor": style.eta_text, "linewidth": 0.6, "alpha": 0.97},
        )


def _legend(ax: Any, handles: Sequence[Line2D], style: MapStyle, *, ncol: int = 1) -> None:
    legend = ax.legend(handles=handles, loc="lower left", fontsize=8, ncol=ncol, frameon=True,
                       fancybox=True, framealpha=0.92, facecolor=style.legend_face,
                       edgecolor=style.land_edge, labelcolor=style.text, borderpad=0.6)
    legend.set_zorder(8)


# --------------------------------------------------------------------------- #
# Renderers
# --------------------------------------------------------------------------- #
def render_graph_figure(
    graph: nx.DiGraph,
    *,
    highlight: Optional[Mapping[str, tuple[str, float]]] = None,
    min_weight: int = 1,
    max_edges: int = 80,
    show_labels: bool = True,
    title: str = "",
    dark: bool = False,
    show_map: bool = True,
    base_map: Optional[BaseMap] = None,
) -> Figure:
    """Draw a single (mixed-wave) propagation graph over the Ukraine map.

    Args:
        graph: Propagation graph.
        highlight: Optional ``region -> (colour, alpha)`` overrides for nodes
            (used to mark triggers and predicted targets).
        min_weight: Hide edges observed fewer times than this.
        max_edges: Keep only the strongest ``max_edges`` edges.
        show_labels: Draw region labels.
        title: Figure title.
        dark: Use the dark style variant.
        show_map: Draw the geographic underlay.
        base_map: Boundaries / centroids; loaded via :func:`load_base_map`
            when omitted.

    Returns:
        A transparent-background matplotlib figure owned by the caller.
    """
    style = map_style(dark)
    base = base_map or load_base_map()
    fig, ax = _new_axes(style, title, (10.5, 6.8))
    if show_map:
        draw_base_map(ax, base, style)

    if graph.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No alarms before T", ha="center", va="center", transform=ax.transAxes,
                color=style.text, fontsize=12, fontweight="bold", path_effects=_halo(style))
        return fig

    pos = layout_positions(graph, base)
    nodes = list(graph.nodes)
    alarms = np.asarray([graph.nodes[n].get("alarms", 1) for n in nodes], dtype=float)
    sizes = 150 + 950 * (alarms / alarms.max() if alarms.max() > 0 else alarms)
    highlight = highlight or {}
    colours = [highlight.get(n, (style.node_fill, 0.95))[0] for n in nodes]
    alphas = [highlight.get(n, (style.node_fill, 0.95))[1] for n in nodes]

    edges = sorted(
        (e for e in graph.edges(data=True) if e[2].get("weight", 1) >= min_weight),
        key=lambda e: (e[2].get("weight", 1), e[2].get("probability", 0.0)),
        reverse=True,
    )[:max_edges]
    if edges:
        weights = np.asarray([e[2].get("weight", 1) for e in edges], dtype=float)
        probs = np.asarray([e[2].get("probability", 0.0) for e in edges], dtype=float)
        prior = graph.graph.get("source") == "prior"
        widths = 1.2 + 4.0 * (probs if probs.max() > 0 else weights / weights.max())
        alphas_e = 0.55 + 0.4 * weights / weights.max()
        nx.draw_networkx_edges(
            graph, pos, edgelist=[(u, v) for u, v, _ in edges], width=list(widths),
            edge_color=_edge_rgba(style.prior_edge if prior else "#3D4F60", alphas_e),
            style="dashed" if prior else "solid", arrows=True, arrowstyle="-|>", arrowsize=14,
            connectionstyle="arc3,rad=0.15", node_size=list(sizes), ax=ax,
        )
        for artist in ax.patches[-len(edges):]:
            artist.set_zorder(2)
    nx.draw_networkx_nodes(graph, pos, nodelist=nodes, node_size=list(sizes), node_color=colours,
                           alpha=alphas, linewidths=1.2, edgecolors=style.node_edge, ax=ax).set_zorder(4)
    if show_labels:
        _draw_labels(ax, pos, nodes, style)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=style.node_fill,
               markeredgecolor=style.node_edge, markersize=10, label="Region (size = alarms before T)"),
        Line2D([0], [0], color="#3D4F60", lw=3.5, label="Propagation edge (width = probability)"),
    ]
    used = {c for c, _ in highlight.values()}
    if "#D1495B" in used or style.active in used:
        handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor=style.active,
                              markeredgecolor=style.node_edge, markersize=10, label="Active trigger at T"))
    if "#F2A541" in used or style.target in used:
        handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor=style.target,
                              markeredgecolor=style.node_edge, markersize=10, label="Predicted next target"))
    _legend(ax, handles, style)
    return fig


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
    show_map: bool = True,
    base_map: Optional[BaseMap] = None,
) -> Figure:
    """Draw fast- and slow-wave propagation layers over the Ukraine map.

    Nodes sit on oblast centroids and are sized by ``activity`` (alarm
    frequency before ``T``). Each layer draws its strongest edges in a
    saturated wave colour, width scaled by transition probability and opacity
    by observation count; prior-topology edges are dashed. Edge labels show
    ``probability · lag``. Regions active at ``T`` are filled red (and their
    oblast polygon tinted when boundaries are available), the selected
    trigger gets a star, and predicted targets are orange with boxed ETA tags.

    Args:
        graphs: Per-wave graphs (``"fast"``, ``"slow"``).
        layers: Wave classes to draw.
        activity: Region -> alarm count used for node sizing.
        active_regions: Regions with an alarm active at ``T``.
        trigger: Region selected as the simulation trigger.
        predictions: Prediction frame with ``region``, ``probability``,
            ``eta_min`` columns.
        min_probability: Hide edges below this transition probability.
        max_edges_per_layer: Strongest edges kept per layer.
        show_edge_labels: Annotate the strongest edges.
        max_labels: Cap on annotated edges across layers.
        show_node_labels: Draw region names.
        title: Figure title.
        dark: Use the dark style variant.
        show_map: Draw the geographic underlay.
        base_map: Boundaries / centroids; loaded via :func:`load_base_map`
            when omitted.

    Returns:
        A transparent-background matplotlib figure owned by the caller.
    """
    style = map_style(dark)
    base = base_map or load_base_map()
    fig, ax = _new_axes(style, title, (11.5, 7.4))

    union = nx.DiGraph()
    for graph in graphs.values():
        union.add_nodes_from(graph.nodes)
    for region in UKRAINIAN_REGIONS:
        union.add_node(region)
    pos = layout_positions(union, base)
    nodes = list(union.nodes)
    active_set = set(active_regions)

    if show_map:
        draw_base_map(ax, base, style, shade={r: style.active for r in active_set})

    counts = np.asarray([float(activity.get(n, 0.0)) if len(activity) else 0.0 for n in nodes])
    peak = counts.max() if counts.size and counts.max() > 0 else 1.0
    sizes = 130 + 1150 * (counts / peak)

    predicted: dict[str, tuple[float, float]] = {}
    if predictions is not None and not predictions.empty:
        predicted = {str(r.region): (float(r.probability), float(r.eta_min))
                     for r in predictions.itertuples(index=False)}
    fill, alpha = [], []
    for node in nodes:
        if node == trigger:
            fill.append(style.trigger); alpha.append(1.0)
        elif node in predicted:
            fill.append(style.target); alpha.append(0.6 + 0.4 * predicted[node][0])
        elif node in active_set:
            fill.append(style.active); alpha.append(0.98)
        else:
            fill.append(style.node_fill); alpha.append(0.92)

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
        colour = WAVE_COLORS.get(wave, WAVE_COLORS["unknown"])
        rad = 0.08 if wave == "fast" else 0.22
        widths = 1.4 + 4.6 * probs
        alphas_e = np.full(len(edges), 0.5) if prior else 0.55 + 0.4 * weights / weights.max()
        nx.draw_networkx_edges(
            graph, pos, edgelist=[(u, v) for u, v, _ in edges], width=list(widths),
            edge_color=_edge_rgba(colour, alphas_e), style="dashed" if prior else "solid",
            arrows=True, arrowstyle="-|>", arrowsize=13, connectionstyle=f"arc3,rad={rad}",
            node_size=list(sizes), ax=ax,
        )
        for artist in ax.patches[-len(edges):]:
            artist.set_zorder(2)
        if show_edge_labels and label_budget > 0:
            labelled = edges[:label_budget]
            label_budget -= len(labelled)
            texts = nx.draw_networkx_edge_labels(
                graph, pos,
                edge_labels={(u, v): f"{d['probability']:.0%} · {d['median_lag_min']:.0f} min"
                             for u, v, d in labelled},
                font_size=7, font_color=colour, font_weight="bold", label_pos=0.5,
                bbox={"boxstyle": "round,pad=0.15", "facecolor": style.legend_face,
                      "edgecolor": "none", "alpha": 0.85},
                connectionstyle=f"arc3,rad={rad}", ax=ax,
            )
            for text in texts.values():
                text.set_zorder(5)

    if trigger and predicted:
        route = nx.DiGraph()
        route.add_edges_from((trigger, target) for target in predicted if trigger in pos)
        nx.draw_networkx_edges(
            route, pos, width=[2.0 + 3.5 * predicted[v][0] for _, v in route.edges],
            edge_color=style.route, alpha=0.95, arrows=True, arrowstyle="-|>", arrowsize=16,
            connectionstyle="arc3,rad=0.12", node_size=list(sizes), ax=ax,
        )
        for artist in ax.patches[-route.number_of_edges():]:
            artist.set_zorder(3)
        _draw_eta_tags(ax, pos, predicted, style)

    nx.draw_networkx_nodes(union, pos, nodelist=nodes, node_size=list(sizes), node_color=fill,
                           alpha=alpha, linewidths=1.2, edgecolors=style.node_edge, ax=ax).set_zorder(4)
    if trigger and trigger in pos:
        ax.scatter([pos[trigger][0]], [pos[trigger][1]], marker="*", s=520, color=style.trigger,
                   edgecolors="#5A3E00", linewidths=1.1, zorder=5)
    if show_node_labels:
        _draw_labels(ax, pos, nodes, style)

    handles = [
        Line2D([0], [0], color=WAVE_COLORS["fast"], lw=4, label="Fast-wave edge (prob · lag)"),
        Line2D([0], [0], color=WAVE_COLORS["slow"], lw=4, label="Slow-wave edge (prob · lag)"),
        Line2D([0], [0], color=style.prior_edge, lw=2.5, linestyle="dashed", label="Prior topology (no data)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=style.node_fill,
               markeredgecolor=style.node_edge, markersize=10, label="Region (size = alarm frequency)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=style.active,
               markeredgecolor=style.node_edge, markersize=10, label="Active at T"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor=style.trigger,
               markeredgecolor="#5A3E00", markersize=15, label="Selected trigger"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=style.target,
               markeredgecolor=style.node_edge, markersize=10, label="Predicted target (+ETA)"),
    ]
    _legend(ax, handles, style, ncol=2)
    return fig
