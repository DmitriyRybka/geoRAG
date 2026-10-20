"""Проверка planner.py: старое поведение (4 вопроса из test_llm.py) и новые поля k и cuisine."""
from landmarks import resolve_place
from planner import plan_question

# (вопрос, категория, радиус, k, кухня, ориентир после resolve_place)
cases = [
    ("Найди кафе в 500 метрах от Летнего сада", "cafe", 500, None, None, "Летний сад"),
    ("Где поесть рядом с Исаакиевским собором", "restaurant", None, None, None, "Исаакиевский собор"),
    ("Парк для прогулки в центре", "park", None, None, None, None),
    ("Ресторан в километре от Эрмитажа", "restaurant", 1000, None, None, "Эрмитаж"),
    ("Три ближайших ресторана к Медному всаднику", "restaurant", None, 3, None, "Медный всадник"),
    ("Грузинский ресторан в 700 метрах от Казанского собора", "restaurant", 700, None, "georgian", "Казанский собор"),
    ("Сколько кафе в радиусе 300 м от Казанского собора", "cafe", 300, None, None, "Казанский собор"),
]

failed = 0
for question, category, radius, k, cuisine, landmark in cases:
    result = plan_question(question)
    plan = result.plan
    got = None
    if plan:
        place = resolve_place(plan.landmark, question)
        got = (plan.category and plan.category.value, plan.radius_m, plan.k,
               plan.cuisine and plan.cuisine.value, place and place[0])
    want = (category, radius, k, cuisine, landmark)
    ok = got == want
    failed += not ok
    print("OK  " if ok else "FAIL", question)
    print("     сырой ответ:", result.raw.strip())
    if not ok:
        print("     получено:", got, "| ожидалось:", want, "| ошибка:", result.error)

print(f"Не совпало: {failed} из {len(cases)}")
