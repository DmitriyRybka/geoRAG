"""Геоинструмент: поиск мест по координатам. Все расстояния в метрах (UTM 36N, EPSG:32636).

Для парков (полигонов) расстояние считается до ближайшей точки полигона,
внутри парка оно равно 0. Для кафе и ресторанов (точек) это обычное расстояние.
"""
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
from pyproj import Transformer
from shapely.geometry import Point

PLACES_PATH = Path(__file__).resolve().parent / "data" / "places.geojson"
METRIC_CRS = "EPSG:32636"
CATEGORIES = {"cafe", "restaurant", "park"}
COLUMNS = ["place_id", "name", "category", "cuisine", "distance_m"]

_to_metric = Transformer.from_crs("EPSG:4326", METRIC_CRS, always_xy=True)  # вход: (lon, lat)


@lru_cache(maxsize=1)
def load_places() -> gpd.GeoDataFrame:
    """Читает data/places.geojson (WGS84) и переводит в метры."""
    return gpd.read_file(PLACES_PATH).to_crs(METRIC_CRS)


def _distances(places: gpd.GeoDataFrame, lat: float, lon: float):
    x, y = _to_metric.transform(lon, lat)
    return places.geometry.distance(Point(x, y))


def _candidates(category: str, cuisine: str | None, places: gpd.GeoDataFrame | None):
    if category not in CATEGORIES:
        raise ValueError(f"category должна быть одной из {sorted(CATEGORIES)}, получено {category!r}")
    places = load_places() if places is None else places
    places = places[places["category"] == category]
    if cuisine:
        # в OSM кухня может быть списком через ";", например "italian;pizza"
        places = places[places["cuisine"].fillna("").str.split(";").apply(lambda cs: cuisine in cs)]
    return places


def places_by_attribute(category: str, cuisine: str | None = None, places: gpd.GeoDataFrame | None = None):
    """Все места категории category (и кухни cuisine) без привязки к точке; distance_m не определено."""
    found = _candidates(category, cuisine, places)
    return found.assign(distance_m=float("nan")).sort_values("place_id")[COLUMNS].reset_index(drop=True)


def places_within_radius(category: str, lat: float, lon: float, radius_m: float,
                         cuisine: str | None = None, places: gpd.GeoDataFrame | None = None):
    """Места категории category не дальше radius_m метров от точки, от ближних к дальним."""
    if radius_m < 0:
        raise ValueError("radius_m не может быть отрицательным")
    found = _candidates(category, cuisine, places)
    found = found.assign(distance_m=_distances(found, lat, lon).round(1))
    found = found[found["distance_m"] <= radius_m]
    return found.sort_values(["distance_m", "place_id"])[COLUMNS].reset_index(drop=True)


def nearest_places(category: str, lat: float, lon: float, k: int,
                   cuisine: str | None = None, places: gpd.GeoDataFrame | None = None):
    """k ближайших мест категории category, от ближних к дальним."""
    if k < 1:
        raise ValueError("k должно быть не меньше 1")
    found = _candidates(category, cuisine, places)
    found = found.assign(distance_m=_distances(found, lat, lon).round(1))
    return found.sort_values(["distance_m", "place_id"])[COLUMNS].head(k).reset_index(drop=True)
