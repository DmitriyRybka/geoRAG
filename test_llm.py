from enum import Enum
from ollama import chat
from pydantic import BaseModel

class Category(str, Enum):
    cafe = "cafe"
    restaurant = "restaurant"
    park = "park"

class Plan(BaseModel):
    category: Category | None
    radius_m: int | None
    landmark: str | None

SYSTEM = """Ты разбираешь вопрос пользователя на параметры поиска мест.
Правила:
- category: cafe, restaurant или park. Если не подходит ни одна, null.
- radius_m: только если расстояние явно названо в вопросе. Иначе null.
- landmark: название ориентира из вопроса. Если нет, null.
Ничего не выдумывай.

Примеры:
Вопрос: Найди кафе в 500 метрах от Летнего сада
Ответ: {"category": "cafe", "radius_m": 500, "landmark": "Летний сад"}
Вопрос: Где поесть рядом с Исаакиевским собором
Ответ: {"category": "restaurant", "radius_m": null, "landmark": "Исаакиевский собор"}
Вопрос: Парк для прогулки
Ответ: {"category": "park", "radius_m": null, "landmark": null}"""

questions = [
    "Найди кафе в 500 метрах от Летнего сада",
    "Где поесть рядом с Исаакиевским собором",
    "Парк для прогулки в центре",
    "Ресторан в километре от Эрмитажа",
]

for q in questions:
    resp = chat(
        model="qwen2.5:3b",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Вопрос: {q}"},
        ],
        format=Plan.model_json_schema(),
        options={"temperature": 0},
    )
    print(q, "->", Plan.model_validate_json(resp.message.content))