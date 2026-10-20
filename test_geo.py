from math import asin, cos, radians, sin, sqrt

import geopandas as gpd
from pyproj import Transformer
from shapely.geometry import Point, box

import geo

HERMITAGE = (59.9398, 30.3146)  # (lat, lon), примерные координаты
ISAAC = (59.9343, 30.3061)


def haversine_m(a, b):
    """Независимая от проекции формула: расстояние по сфере радиуса 6371 км."""
    lat1, lon1, lat2, lon2 = map(radians, (*a, *b))
    h = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371000 * asin(sqrt(h))


def one_place(geometry_wgs84, category="cafe"):
    gdf = gpd.GeoDataFrame(
        {"place_id": ["test/1"], "name": ["тест"], "category": [category], "cuisine": [None]},
        geometry=[geometry_wgs84], crs="EPSG:4326",
    )
    return gdf.to_crs(geo.METRIC_CRS)


# 1. UTM даёт то же расстояние, что гаверсинус (допуск 0,5 %)
d_utm = geo._distances(one_place(Point(ISAAC[1], ISAAC[0])), *HERMITAGE).iloc[0]
d_hav = haversine_m(HERMITAGE, ISAAC)
print(f"Эрмитаж - Исаакий: UTM {d_utm:.1f} м, гаверсинус {d_hav:.1f} м")
assert abs(d_utm - d_hav) / d_hav < 0.005

# 2. Парк-квадрат 100x100 м: точка в 30 м от стороны, внутри и в углу (ответы известны точно)
x0, y0 = 360000, 6650000
square = gpd.GeoDataFrame({"place_id": ["test/2"], "name": ["парк"], "category": ["park"], "cuisine": [None]},
                          geometry=[box(x0, y0, x0 + 100, y0 + 100)], crs=geo.METRIC_CRS)
to_wgs = Transformer.from_crs(geo.METRIC_CRS, "EPSG:4326", always_xy=True)


def dist_to_square(x, y):
    lon, lat = to_wgs.transform(x, y)
    return geo._distances(square, lat, lon).iloc[0]


print("Парк-квадрат: сбоку 30 м ->", round(dist_to_square(x0 + 130, y0 + 50), 1),
      "| внутри ->", round(dist_to_square(x0 + 50, y0 + 50), 1),
      "| от угла (30, 40) ->", round(dist_to_square(x0 + 130, y0 + 140), 1))
assert abs(dist_to_square(x0 + 130, y0 + 50) - 30) < 0.1
assert dist_to_square(x0 + 50, y0 + 50) == 0
assert abs(dist_to_square(x0 + 130, y0 + 140) - 50) < 0.1  # 3-4-5

# 3. within_radius совпадает с перебором по гаверсинусу (кафе в базе - точки)
places = geo.load_places()
to_wgs_all = places.to_crs("EPSG:4326")
cafes = to_wgs_all[(to_wgs_all["category"] == "cafe") & (to_wgs_all.geometry.geom_type == "Point")]
for radius in (300, 800):
    got = set(geo.places_within_radius("cafe", *HERMITAGE, radius)["place_id"]) & set(cafes["place_id"])
    expected = {
        row.place_id for row in cafes.itertuples()
        if haversine_m(HERMITAGE, (row.geometry.y, row.geometry.x)) <= radius
    }
    diff = got ^ expected
    # допускаем расхождения только у самой границы круга (там сферическая модель чуть врёт)
    for pid in diff:
        row = cafes[cafes["place_id"] == pid].iloc[0]
        assert abs(haversine_m(HERMITAGE, (row.geometry.y, row.geometry.x)) - radius) < 3, pid
    print(f"Кафе в {radius} м от Эрмитажа: {len(got)} (перебор: {len(expected)}, на границе спорных: {len(diff)})")

# 4. nearest_places: k штук, по возрастанию, и место само для себя ближайшее с расстоянием 0
near = geo.nearest_places("restaurant", *HERMITAGE, k=5)
print(near.to_string())
assert len(near) == 5 and near["distance_m"].is_monotonic_increasing
one = to_wgs_all[to_wgs_all["category"] == "restaurant"].iloc[0]
pt = one.geometry.representative_point()
assert geo.nearest_places("restaurant", pt.y, pt.x, k=1).loc[0, "place_id"] == one["place_id"]

# 5. парк: точка внутри полигона даёт 0 м, nearest сам находит этот парк
park = to_wgs_all[to_wgs_all["category"] == "park"].iloc[0]
pt = park.geometry.representative_point()
inside = geo.nearest_places("park", pt.y, pt.x, k=1)
print("Точка внутри парка:", inside.loc[0, "name"], inside.loc[0, "distance_m"], "м")
assert inside.loc[0, "place_id"] == park["place_id"] and inside.loc[0, "distance_m"] == 0

# 6. фильтр по кухне и проверка аргументов
georgian = geo.places_within_radius("restaurant", *HERMITAGE, 2000, cuisine="georgian")
print("Грузинская кухня в 2 км от Эрмитажа:", len(georgian))
assert len(georgian) > 0 and all("georgian" in c.split(";") for c in georgian["cuisine"])
for bad in (lambda: geo.nearest_places("bar", *HERMITAGE, k=1),
            lambda: geo.nearest_places("cafe", *HERMITAGE, k=0),
            lambda: geo.places_within_radius("cafe", *HERMITAGE, -5)):
    try:
        bad()
        raise AssertionError("ожидалась ValueError")
    except ValueError:
        pass

print("OK: все проверки пройдены")
