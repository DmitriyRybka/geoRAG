"""Вопрос -> планировщик -> ориентир с координатами -> геоинструмент -> ответ.

Запуск: python ask.py "Найди кафе в 500 метрах от Летнего сада"
Координаты ориентиров: python scripts/geocode_landmarks.py (создаёт data/landmarks.json).
"""
import json
import sys
from pathlib import Path

import geo
from landmarks import LANDMARKS, resolve_place
from planner import plan_question

LANDMARKS_PATH = Path(__file__).resolve().parent / "data" / "landmarks.json"
DEFAULT_K = 5  # если в вопросе нет ни радиуса, ни числа мест
MAX_SHOWN = 15


def load_landmark_coords() -> dict:
    if not LANDMARKS_PATH.exists():
        raise FileNotFoundError(f"Нет {LANDMARKS_PATH}: запустите python scripts/geocode_landmarks.py")
    return json.loads(LANDMARKS_PATH.read_text(encoding="utf-8"))


def ask(question: str) -> dict:
    """Возвращает словарь: plan_result, place, call, table (DataFrame или None), message (если не вышло)."""
    out = {"question": question, "plan_result": None, "place": None, "call": None, "table": None, "message": None}

    out["plan_result"] = result = plan_question(question)
    if result.plan is None:
        out["message"] = "Не удалось разобрать вопрос (модель вернула неверную структуру)."
        return out
    plan = result.plan

    if plan.category is None:
        out["message"] = "Не понял, что искать: кафе, ресторан или парк."
        return out

    out["place"] = place = resolve_place(plan.landmark, question)
    if place is None:
        if plan.cuisine:  # вопрос про атрибут без ориентира: все места такой категории и кухни
            out["table"] = geo.places_by_attribute(plan.category.value, plan.cuisine.value)
            out["call"] = f"places_by_attribute({plan.category.value}, cuisine={plan.cuisine.value})"
            return out
        out["message"] = f"Не нашёл ориентир в вопросе. Известные ориентиры: {', '.join(LANDMARKS)}."
        return out
    point = load_landmark_coords()[place[0]]
    where = (plan.category.value, point["lat"], point["lon"])
    cuisine = plan.cuisine.value if plan.cuisine else None

    if plan.radius_m is not None:
        table = geo.places_within_radius(*where, plan.radius_m, cuisine=cuisine)
        out["call"] = f"places_within_radius({plan.category.value}, {place[0]}, radius_m={plan.radius_m}, cuisine={cuisine})"
        if plan.k:
            table = table.head(plan.k)
            out["call"] += f" + первые {plan.k}"
    else:
        k = plan.k or DEFAULT_K
        table = geo.nearest_places(*where, k, cuisine=cuisine)
        out["call"] = f"nearest_places({plan.category.value}, {place[0]}, k={k}, cuisine={cuisine})"
    out["table"] = table
    return out


def print_answer(out: dict) -> None:
    result = out["plan_result"]
    print(f"Вопрос: {out['question']}")
    print(f"Ответ модели: {result.raw.strip()}")
    if out["place"]:
        print(f"Ориентир: {out['place'][0]} (оценка {out['place'][1]}, {out['place'][2]})")
    if out["message"]:
        print(out["message"])
        return
    table = out["table"]
    print(f"Вызов: {out['call']}")
    print(f"Найдено мест: {len(table)}")
    for i, row in enumerate(table.head(MAX_SHOWN).itertuples(), 1):
        distance = f"{row.distance_m:.0f} м" if row.distance_m == row.distance_m else "без расстояния"
        print(f"{i:>2}. {row.name} - {distance} ({row.place_id})")
    if len(table) > MAX_SHOWN:
        print(f"... и ещё {len(table) - MAX_SHOWN}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) < 2:
        sys.exit('Использование: python ask.py "вопрос"')
    print_answer(ask(sys.argv[1]))
