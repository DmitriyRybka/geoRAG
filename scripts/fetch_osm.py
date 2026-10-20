"""Загрузка кафе, ресторанов и парков Центрального и Адмиралтейского районов из OSM.

Запуск из корня проекта: python scripts/fetch_osm.py
Результат: data/places.geojson (WGS84). Сырые ответы Overpass кэшируются в cache/.
"""
import time
from pathlib import Path

import geopandas as gpd
import osmnx as ox

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_PATH = DATA_DIR / "places.geojson"

PLACES = [
    "Центральный район, Санкт-Петербург, Россия",
    "Адмиралтейский район, Санкт-Петербург, Россия",
]
TAGS = {"amenity": ["cafe", "restaurant"], "leisure": "park"}

OVERPASS_URLS = [
    "https://overpass-api.de/api",
    "https://overpass.kumi.systems/api",
    "https://overpass.private.coffee/api",
]
RETRIES = 3

# теги OSM, которые пригодятся для описаний (этап 4) и вопросов по атрибуту (этап 5)
KEEP = ["name", "cuisine", "addr:street", "addr:housenumber", "opening_hours"]


def fetch() -> gpd.GeoDataFrame:
    """Публичные серверы Overpass бывают перегружены: пробуем каждый с нарастающей паузой."""
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(ROOT / "cache")
    for url in OVERPASS_URLS:
        ox.settings.overpass_url = url
        for attempt in range(RETRIES):
            try:
                return ox.features_from_place(PLACES, TAGS)
            except Exception as e:
                wait = 10 * 2**attempt
                print(f"{url}, попытка {attempt + 1}/{RETRIES}: {type(e).__name__}; жду {wait} с")
                time.sleep(wait)
    raise RuntimeError("Ни один сервер Overpass не ответил, попробуйте позже")


def clean(raw: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = raw.reset_index()  # уровни индекса osmnx: element (node/way/relation) и id

    gdf["place_id"] = gdf["element"] + "/" + gdf["id"].astype(str)
    # category берём из amenity, а для парков из leisure
    gdf["category"] = gdf["amenity"].where(gdf["amenity"].isin(["cafe", "restaurant"]), None)
    gdf.loc[gdf["leisure"] == "park", "category"] = "park"
    gdf = gdf[gdf["category"].notna()]

    for col in KEEP:
        if col not in gdf.columns:
            gdf[col] = None

    total = len(gdf)
    gdf = gdf[gdf["name"].notna()]
    print(f"Без названия отброшено: {total - len(gdf)} из {total}")

    return gdf[["place_id", "category"] + KEEP + ["geometry"]].reset_index(drop=True)


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    gdf = clean(fetch())
    gdf.to_file(OUT_PATH, driver="GeoJSON")
    print(f"Сохранено {len(gdf)} мест в {OUT_PATH}")
    print(gdf.groupby(["category", gdf.geometry.geom_type]).size())


if __name__ == "__main__":
    main()
