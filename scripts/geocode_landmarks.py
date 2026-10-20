"""Координаты ориентиров из landmarks.py через геокодинг OSMnx (Nominatim).

Запуск из корня проекта: python scripts/geocode_landmarks.py
Результат: data/landmarks.json {имя: {"lat": ..., "lon": ..., "query": ...}}. Ответы кэшируются в cache/.
"""
import json
import sys
from pathlib import Path

import osmnx as ox

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from landmarks import LANDMARKS  # noqa: E402

OUT_PATH = ROOT / "data" / "landmarks.json"

# Запрос уточняет объект: по одному имени Nominatim может найти что-то в другом городе.
QUERIES = {
    "Летний сад": "Летний сад, Санкт-Петербург, Россия",
    "Эрмитаж": "Государственный Эрмитаж, Санкт-Петербург, Россия",
    "Исаакиевский собор": "Исаакиевский собор, Санкт-Петербург, Россия",
    "Казанский собор": "Казанский собор, Санкт-Петербург, Россия",
    "Дворцовая площадь": "Дворцовая площадь, Санкт-Петербург, Россия",
    "Медный всадник": "Медный всадник, Санкт-Петербург, Россия",
    "Невский проспект": "Невский проспект, Санкт-Петербург, Россия",
}


def main() -> None:
    assert set(QUERIES) == set(LANDMARKS), "QUERIES должны покрывать все ориентиры из landmarks.py"
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(ROOT / "cache")

    result = {}
    for name, query in QUERIES.items():
        lat, lon = ox.geocode(query)
        result[name] = {"lat": lat, "lon": lon, "query": query}
        print(f"{name}: {lat:.5f}, {lon:.5f}")

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Сохранено в {OUT_PATH}")


if __name__ == "__main__":
    main()
