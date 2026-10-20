"""Базовая линия «простой RAG»: поиск мест по смыслу текста, без координат.

Описание места из тегов OSM -> эмбеддинг (multilingual-e5-small) -> top-k по сходству с вопросом.
Запуск: python rag.py "Найди кафе в 500 метрах от Летнего сада"
Эмбеддинги кэшируются в cache/ и пересчитываются, если изменились описания или модель.
"""
import hashlib
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

import geo

MODEL_NAME = "intfloat/multilingual-e5-small"
CACHE_DIR = Path(__file__).resolve().parent / "cache"

TYPE_RU = {"cafe": "Кафе", "restaurant": "Ресторан", "park": "Парк"}
CUISINE_RU = {
    "georgian": "грузинская", "italian": "итальянская", "russian": "русская",
    "japanese": "японская", "chinese": "китайская", "french": "французская",
    "indian": "индийская", "uzbek": "узбекская", "vietnamese": "вьетнамская",
}


def _text(value) -> str | None:
    """Пустые значения в таблице бывают None или NaN: приводим к None."""
    return value.strip() if isinstance(value, str) and value.strip() else None


def describe(place: dict) -> str:
    """Текстовое описание места из тегов: тип, название, кухня, адрес."""
    parts = [f"{TYPE_RU[place['category']]} «{place['name']}»"]
    if cuisine := _text(place.get("cuisine")):
        names = [CUISINE_RU.get(c, c) for c in cuisine.split(";")]
        parts.append(f"Кухня: {', '.join(names)} ({cuisine})")
    if street := _text(place.get("addr:street")):
        house = _text(place.get("addr:housenumber"))
        parts.append(f"Адрес: {street}, {house}" if house else f"Адрес: {street}")
    return ". ".join(parts)


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME, device="cpu")


@lru_cache(maxsize=1)
def load_index():
    """Возвращает (таблица мест, тексты описаний, матрица эмбеддингов), строки согласованы."""
    places = geo.load_places().drop(columns="geometry").reset_index(drop=True)
    texts = [describe(p) for p in places.to_dict("records")]
    key = hashlib.sha1((MODEL_NAME + "\n".join(texts)).encode("utf-8")).hexdigest()[:12]
    path = CACHE_DIR / f"embeddings_{key}.npy"
    if path.exists():
        embeddings = np.load(path)
    else:
        # префикс "passage: " обязателен для e5; нормировка делает скалярное произведение косинусом
        embeddings = _model().encode(["passage: " + t for t in texts], batch_size=64,
                                     normalize_embeddings=True, show_progress_bar=True)
        CACHE_DIR.mkdir(exist_ok=True)
        np.save(path, embeddings)
    return places[["place_id", "name", "category"]], texts, embeddings


def retrieve(question: str, k: int = 5) -> pd.DataFrame:
    """k мест, описания которых ближе всего к вопросу по смыслу. Расстояния не учитываются."""
    if k < 1:
        raise ValueError("k должно быть не меньше 1")
    places, texts, embeddings = load_index()
    query = _model().encode(["query: " + question], normalize_embeddings=True)[0]
    scores = embeddings @ query
    top = np.argsort(-scores, kind="stable")[:k]
    result = places.iloc[top].copy()
    result["score"] = scores[top].astype(float).round(4)
    result["text"] = [texts[i] for i in top]
    return result.reset_index(drop=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) < 2:
        sys.exit('Использование: python rag.py "вопрос" [k]')
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    for i, row in enumerate(retrieve(sys.argv[1], k).itertuples(), 1):
        print(f"{i:>2}. {row.text}  [{row.score}] ({row.place_id})")
