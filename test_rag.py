import json
from collections import Counter

import numpy as np

import geo
import rag

# 1. описание места собирается из тегов предсказуемо
sample = {"category": "restaurant", "name": "Сулико", "cuisine": "georgian;russian",
          "addr:street": "Казанская улица", "addr:housenumber": "5"}
expected = "Ресторан «Сулико». Кухня: грузинская, русская (georgian;russian). Адрес: Казанская улица, 5"
assert rag.describe(sample) == expected, rag.describe(sample)
assert rag.describe({"category": "park", "name": "Таврический сад", "cuisine": None,
                     "addr:street": float("nan"), "addr:housenumber": None}) == "Парк «Таврический сад»"

# 2. индекс согласован с данными, векторы нормированы
places, texts, emb = rag.load_index()
assert len(places) == len(texts) == len(emb) == len(geo.load_places())
assert np.allclose(np.linalg.norm(emb, axis=1), 1, atol=1e-3)
print(f"Индекс: {len(places)} мест, вектор {emb.shape[1]} чисел")

# 3. место находится по собственному описанию (берём места с уникальным текстом)
counts = Counter(texts)
rng = np.random.default_rng(0)
candidates = [i for i, t in enumerate(texts) if counts[t] == 1]
picked = rng.choice(candidates, size=100, replace=False)
hits = sum(rag.retrieve(texts[i], k=1).loc[0, "place_id"] == places.loc[i, "place_id"] for i in picked)
print(f"Поиск по собственному описанию: top-1 верно у {hits} из 100")
assert hits >= 80

# 4. смысловой запрос по кухне: доля грузинских ресторанов в top-10
top = rag.retrieve("грузинский ресторан", k=10)
full = geo.load_places().set_index("place_id")
georgian = sum(
    top.loc[i, "category"] == "restaurant" and "georgian" in str(full.loc[top.loc[i, "place_id"], "cuisine"]).split(";")
    for i in range(len(top))
)
print(f"«грузинский ресторан»: грузинских ресторанов в top-10: {georgian}")
assert georgian >= 7

# 5. кэш эмбеддингов лежит в cache/ и повторная загрузка даёт те же числа
cached = list(rag.CACHE_DIR.glob("embeddings_*.npy"))
assert cached and any(np.array_equal(np.load(p), emb) for p in cached)

# 6. демонстрация гипотезы (только печать): где на самом деле места, которые вернул RAG
landmark = json.loads((rag.Path(__file__).parent / "data" / "landmarks.json").read_text(encoding="utf-8"))["Летний сад"]
question = "Найди кафе в 500 метрах от Летнего сада"
answer = rag.retrieve(question, k=10)
truth = geo.places_within_radius("cafe", landmark["lat"], landmark["lon"], 500)
distances = geo._distances(full.loc[answer["place_id"]].reset_index(), landmark["lat"], landmark["lon"])
print(f"RAG top-10 для «{question}»: расстояния до Летнего сада, м:", sorted(int(d) for d in distances))
print(f"Совпало с настоящим ответом: {len(set(answer['place_id']) & set(truth['place_id']))} из {len(truth)} кафе в радиусе")

print("OK: все проверки пройдены")
