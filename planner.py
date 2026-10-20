from dataclasses import dataclass
from enum import Enum

from ollama import chat
from pydantic import BaseModel, ValidationError

MODEL = "qwen2.5:3b"


class Category(str, Enum):
    cafe = "cafe"
    restaurant = "restaurant"
    park = "park"


class Cuisine(str, Enum):
    georgian = "georgian"
    italian = "italian"
    russian = "russian"
    japanese = "japanese"
    chinese = "chinese"
    french = "french"
    indian = "indian"
    uzbek = "uzbek"
    vietnamese = "vietnamese"


class Plan(BaseModel):
    category: Category | None
    cuisine: Cuisine | None
    radius_m: int | None
    k: int | None
    landmark: str | None


@dataclass
class PlanResult:
    """Разбор вопроса: сырой ответ модели хранится рядом с проверенным планом для разбора ошибок."""
    question: str
    raw: str
    plan: Plan | None
    error: str | None = None


SYSTEM = """Ты разбираешь вопрос пользователя на параметры поиска мест.
Правила:
- category: cafe, restaurant или park. Если не подходит ни одна, null.
- cuisine: кухня, только если она явно названа в вопросе (georgian, italian, russian, japanese, chinese, french, indian, uzbek, vietnamese). Иначе null.
- radius_m: только если расстояние явно названо в вопросе, в метрах. Иначе null.
- k: только если в вопросе явно назван, сколько ближайших мест нужно (например, "три ближайших" -> 3). Иначе null.
- landmark: название ориентира из вопроса. Если нет, null.
Ничего не выдумывай.

Примеры:
Вопрос: Найди кафе в 500 метрах от Летнего сада
Ответ: {"category": "cafe", "cuisine": null, "radius_m": 500, "k": null, "landmark": "Летний сад"}
Вопрос: Где поесть рядом с Исаакиевским собором
Ответ: {"category": "restaurant", "cuisine": null, "radius_m": null, "k": null, "landmark": "Исаакиевский собор"}
Вопрос: Парк для прогулки
Ответ: {"category": "park", "cuisine": null, "radius_m": null, "k": null, "landmark": null}
Вопрос: Два ближайших кафе к Дворцовой площади
Ответ: {"category": "cafe", "cuisine": null, "radius_m": null, "k": 2, "landmark": "Дворцовая площадь"}
Вопрос: Итальянский ресторан в 800 метрах от Эрмитажа
Ответ: {"category": "restaurant", "cuisine": "italian", "radius_m": 800, "k": null, "landmark": "Эрмитаж"}"""


def plan_question(question: str) -> PlanResult:
    resp = chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Вопрос: {question}"},
        ],
        format=Plan.model_json_schema(),
        options={"temperature": 0},
    )
    raw = resp.message.content
    try:
        return PlanResult(question, raw, Plan.model_validate_json(raw))
    except ValidationError as e:
        return PlanResult(question, raw, None, error=str(e))
