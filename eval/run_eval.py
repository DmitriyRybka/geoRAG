"""Детерминированная оценка: эталоны и метрики считает код, LLM в оценке не участвует.

Запуск из корня проекта: python eval/run_eval.py
Результат: eval/results.json (сырые данные) и eval/results.md (таблицы и примеры ошибок).

Системы:
  geo      - ask.ask(): планировщик -> ориентир -> геоинструмент (весь путь, как для пользователя)
  rag      - rag.retrieve(): только текст, без фильтров
  rag_cat  - то же, но результаты оставлены только нужной категории (категория берётся из
             параметров вопроса: оракул, верхняя граница для RAG)
Размер ответа RAG равен размеру эталона (k = |эталон|): это подсказка в пользу RAG.
"""
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import ask as ask_module  # noqa: E402
import geo  # noqa: E402
import rag  # noqa: E402

QUESTIONS_PATH = ROOT / "eval" / "questions.yaml"
RESULTS_JSON = ROOT / "eval" / "results.json"
RESULTS_MD = ROOT / "eval" / "results.md"

TYPES = ["radius", "nearest", "attribute", "mixed"]
TYPE_RU = {"radius": "в радиусе", "nearest": "ближайшие", "attribute": "по атрибуту", "mixed": "смешанные"}
SYSTEMS = ["geo", "rag_cat", "rag"]
SYSTEM_RU = {"geo": "Геосистема", "rag": "RAG", "rag_cat": "RAG + категория"}


def compute_truth(q: dict, coords: dict) -> list[str]:
    """ID мест правильного ответа, от ближних к дальним (для атрибута: по ID)."""
    cuisine = q.get("cuisine")
    if "landmark" in q:
        point = coords[q["landmark"]]
        where = (q["category"], point["lat"], point["lon"])
        if "radius_m" in q:
            return list(geo.places_within_radius(*where, q["radius_m"], cuisine=cuisine)["place_id"])
        return list(geo.nearest_places(*where, q["k"], cuisine=cuisine)["place_id"])
    return list(geo.places_by_attribute(q["category"], cuisine)["place_id"])


def score(pred: list[str], truth: list[str], nearest: bool) -> dict:
    hits = len(set(pred) & set(truth))
    return {
        "precision": hits / len(pred) if pred else 0.0,
        "recall": hits / len(truth),
        # для «ближайших»: есть ли самое близкое место в ответе
        "top1": (truth[0] in pred) if nearest else None,
    }


def plan_diff(out: dict, q: dict) -> list[str]:
    """Какие поля разбора вопроса не совпали с параметрами вопроса (диагностика ошибок)."""
    plan = out["plan_result"].plan
    if plan is None:
        return ["разбор не удался"]
    place = out["place"]
    got = {
        "category": plan.category and plan.category.value,
        "cuisine": plan.cuisine and plan.cuisine.value,
        "radius_m": plan.radius_m,
        "k": plan.k,
        "landmark": place and place[0],
    }
    return [f"{f}: {got[f]!r} вместо {q.get(f)!r}" for f in got if got[f] != q.get(f)]


def run_question(q: dict, coords: dict) -> dict:
    truth = compute_truth(q, coords)
    nearest = "landmark" in q and "k" in q and "radius_m" not in q
    n = len(truth)

    out = ask_module.ask(q["text"])
    geo_pred = list(out["table"]["place_id"]) if out["table"] is not None else []

    all_ranked = rag.retrieve(q["text"], k=len(geo.load_places()))
    rag_pred = list(all_ranked["place_id"].head(n))
    rag_cat_pred = list(all_ranked[all_ranked["category"] == q["category"]]["place_id"].head(n))

    preds = {"geo": geo_pred, "rag": rag_pred, "rag_cat": rag_cat_pred}
    pool = int((geo.load_places()["category"] == q["category"]).sum())  # мест этой категории всего
    return {
        "pool": pool,
        "id": q["id"], "type": q["type"], "text": q["text"], "params": {k: v for k, v in q.items() if k not in ("id", "type", "text")},
        "nearest": nearest, "truth": truth, "preds": preds,
        "scores": {s: score(preds[s], truth, nearest) for s in SYSTEMS},
        "plan_raw": out["plan_result"].raw.strip(), "plan_diff": plan_diff(out, q), "message": out["message"],
    }


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def chance(r: dict) -> float:
    """Ожидание для случайного ответа того же размера из мест нужной категории (P, R и шанс top1 равны)."""
    return len(r["truth"]) / r["pool"]


def pr_cells(group: list[dict]) -> list[str]:
    cells = [f"{mean(r['scores'][s]['precision'] for r in group):.2f} / {mean(r['scores'][s]['recall'] for r in group):.2f}"
             for s in SYSTEMS]
    return cells + [f"{mean(chance(r) for r in group):.2f} / {mean(chance(r) for r in group):.2f}"]


def summary_tables(results: list[dict]) -> str:
    systems_ru = [SYSTEM_RU[s] for s in SYSTEMS] + ["Случайный выбор (ожидание)"]
    rows = []
    for t in TYPES + ["all"]:
        group = [r for r in results if t == "all" or r["type"] == t]
        if group:
            rows.append([("**Все вопросы**" if t == "all" else TYPE_RU[t]), str(len(group))] + pr_cells(group))
    out = table(["Тип вопроса", "N"] + [f"{s} P / R" for s in systems_ru], rows)

    nearest = [r for r in results if r["nearest"]]
    top1 = ["Самое близкое место есть в ответе (вопросы «k ближайших»), N=%d" % len(nearest)]
    top1 += [f"{sum(r['scores'][s]['top1'] for r in nearest)} из {len(nearest)}" for s in SYSTEMS]
    top1 += [f"{sum(chance(r) for r in nearest):.1f} из {len(nearest)}"]
    out += "\n\n" + table(["Показатель"] + systems_ru, [top1])

    dist = [r for r in results if "landmark" in r["params"]]
    row = ["Только вопросы с расстоянием, N=%d (P / R)" % len(dist)] + pr_cells(dist)
    out += "\n\n" + table(["Показатель"] + systems_ru, [row])
    return out


def per_question_table(results: list[dict]) -> str:
    header = ["ID", "Тип", "Эталон, шт."] + [f"{SYSTEM_RU[s]} P / R" for s in SYSTEMS] + ["Разбор вопроса"]
    rows = []
    for r in results:
        cells = [f"{r['scores'][s]['precision']:.2f} / {r['scores'][s]['recall']:.2f}" for s in SYSTEMS]
        rows.append([r["id"], TYPE_RU[r["type"]], str(len(r["truth"]))] + cells + ["верно" if not r["plan_diff"] else "ошибка"])
    return table(header, rows)


def names_with_distance(ids: list[str], point: dict | None, places_idx, limit: int = 5) -> str:
    ids = ids[:limit]
    if not ids:
        return "пусто"
    sub = places_idx.loc[ids]
    if point is None:
        return "; ".join(sub["name"])
    dist = geo._distances(sub, point["lat"], point["lon"]).values
    return "; ".join(f"{n} ({d:.0f} м)" for n, d in zip(sub["name"], dist))


def error_examples(results: list[dict], coords: dict) -> str:
    places_idx = geo.load_places().set_index("place_id")
    blocks = []

    with_distance = [r for r in results if "landmark" in r["params"]]
    worst = sorted(with_distance, key=lambda r: (r["scores"]["rag"]["recall"], r["id"]))[:3]
    blocks.append("### Где ошибается RAG (вопросы с расстоянием, худшие по recall)\n")
    for r in worst:
        point = coords[r["params"]["landmark"]]
        blocks.append(
            f"**{r['id']}. {r['text']}** (эталон {len(r['truth'])} мест, RAG recall {r['scores']['rag']['recall']:.2f})\n\n"
            f"- Эталон (первые 5, расстояние до ориентира): {names_with_distance(r['truth'], point, places_idx)}\n"
            f"- RAG вернул (первые 5, настоящее расстояние до ориентира): {names_with_distance(r['preds']['rag'], point, places_idx)}\n"
        )

    bad_geo = [r for r in results if r["scores"]["geo"]["recall"] < 1 or r["scores"]["geo"]["precision"] < 1]
    blocks.append("### Где ошибается геосистема\n")
    if not bad_geo:
        blocks.append("Ошибок нет: на всех вопросах ответ совпал с эталоном.\n")
    for r in bad_geo:
        point = coords.get(r["params"].get("landmark"))
        blocks.append(
            f"**{r['id']}. {r['text']}** (P {r['scores']['geo']['precision']:.2f} / R {r['scores']['geo']['recall']:.2f})\n\n"
            f"- Ответ модели: `{r['plan_raw']}`\n"
            f"- Расхождения разбора: {'; '.join(r['plan_diff']) or 'нет (параметры разобраны верно)'}\n"
            f"- Сообщение системы: {r['message'] or '-'}\n"
            f"- Эталон (первые 5): {names_with_distance(r['truth'], point, places_idx)}\n"
            f"- Геосистема вернула (первые 5): {names_with_distance(r['preds']['geo'], point, places_idx)}\n"
        )
    return "\n".join(blocks)


def write_report(results: list[dict], coords: dict, questions_path: Path, out_path: Path, supplementary: bool) -> None:
    n_plan_ok = sum(not r["plan_diff"] for r in results)
    if supplementary:
        title = f"""# Дополнительная проверка: сложные формулировки

**Эти вопросы добавлены после просмотра основных результатов** (`eval/results.md`), чтобы проверить планировщик на формулировках,
непохожих на few-shot-примеры. Они не входят в основную таблицу и не подгонялись под результат.
Вопросов: {len(results)}. Параметры: `{questions_path.relative_to(ROOT).as_posix()}`.
Воспроизвести: `python eval/run_eval.py {questions_path.relative_to(ROOT).as_posix()} stress`."""
    else:
        title = f"""# Результаты оценки

Вопросов: {len(results)} (в радиусе, ближайшие, по атрибуту, смешанные). Параметры вопросов: `{questions_path.relative_to(ROOT).as_posix()}`.
Воспроизвести: `python eval/run_eval.py`."""
    md = f"""{title}

## Как считали

- Эталон вычисляет код по координатам (UTM 36N, EPSG:32636) из параметров вопроса. Вручную ответы не размечали, LLM в оценке не участвует.
- Метрики по ID мест: precision (P) = доля верных среди возвращённых, recall (R) = доля найденных среди эталона. Средние по вопросам.
- **Геосистема**: весь путь для пользователя (`ask.ask`): планировщик qwen2.5:3b -> ориентир -> геоинструмент.
- **RAG**: `multilingual-e5-small`, только текст описаний, без координат. Размер его ответа равен размеру эталона (подсказка в пользу RAG, разрыв с геосистемой поэтому консервативен).
- **RAG + категория**: то же, но оставлены места нужной категории; категория взята из параметров вопроса (оракул, верхняя граница для RAG).
- Планировщик разобрал параметры верно в {n_plan_ok} из {len(results)} вопросов.

## Итог по типам вопросов

{summary_tables(results)}

## По каждому вопросу

{per_question_table(results)}

## Примеры ошибок

{error_examples(results, coords)}
## Ограничения оценки

- Выборка маленькая ({len(results)} вопросов), доверительных интервалов нет; различия в десятых долях нужно читать осторожно.
- Эталон считается тем же `geo.py`, что и использует геосистема. Сам `geo.py` проверен отдельно на независимых расчётах (`test_geo.py`), но в этой оценке геосистема может ошибиться только на разборе вопроса, а не на расчёте расстояний.
- Вопросы «ближайшие» заданы с явным числом: планировщик не понимает единственное число («ближайший парк» даёт k=null). Известный провал заранее исключён из основной выборки; он проверяется в дополнительном наборе (`eval/results_stress.md`).
- Few-shot-примеры планировщика близки по форме к вопросам основной выборки («Два ближайших…», «Итальянский ресторан в 800 метрах от Эрмитажа»), поэтому точность разбора в основной выборке показывает работу на знакомых формулировках, а не обобщение.
- Эталон «по атрибуту» строится по тегу `cuisine`. RAG находит и места, у которых тег не проставлен, но название говорит о кухне (например, японские по названию), и в этой оценке они считаются ошибками. На тип «по атрибуту» оценка смещена в пользу структурированного тега; результат RAG там сильно зависит от формулировки («японский ресторан» и «Покажи японские рестораны» дают разные ответы).
- Ориентиров 7, «Невский проспект» (линия) в вопросы не взят.
"""
    out_path.write_text(md, encoding="utf-8")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    # python eval/run_eval.py [файл_вопросов метка]: с меткой результаты пишутся в results_<метка>.*
    questions_path = (ROOT / sys.argv[1]) if len(sys.argv) > 1 else QUESTIONS_PATH
    tag = sys.argv[2] if len(sys.argv) > 2 else ""
    out_json = RESULTS_JSON.with_name(f"results_{tag}.json") if tag else RESULTS_JSON
    out_md = RESULTS_MD.with_name(f"results_{tag}.md") if tag else RESULTS_MD
    questions = yaml.safe_load(questions_path.read_text(encoding="utf-8"))
    coords = ask_module.load_landmark_coords()
    results = []
    for q in questions:
        r = run_question(q, coords)
        results.append(r)
        s = r["scores"]
        print(f"{r['id']:>3} эталон={len(r['truth']):>2} | " + " | ".join(
            f"{name} P={s[name]['precision']:.2f} R={s[name]['recall']:.2f}" for name in SYSTEMS)
            + (" | РАЗБОР: " + "; ".join(r["plan_diff"]) if r["plan_diff"] else ""))
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    write_report(results, coords, questions_path, out_md, supplementary=bool(tag))
    print(f"\nОтчёт: {out_md}")


if __name__ == "__main__":
    main()
