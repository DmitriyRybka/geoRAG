from ollama import chat
from pydantic import BaseModel

class Plan(BaseModel):
    category: str
    radius_m: int | None
    landmark: str | None

question = "Где поесть рядом с Исаакиевским собором"

resp = chat(
    model="qwen2.5:3b",
    messages=[{
        "role": "user",
        "content": f"Разбери вопрос на параметры поиска. Вопрос: {question}",
    }],
    format=Plan.model_json_schema(),
)

plan = Plan.model_validate_json(resp.message.content)
print(plan)