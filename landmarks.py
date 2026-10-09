from rapidfuzz import fuzz, process

LANDMARKS = {
    "Летний сад": ["летний сад", "летнего сада", "летнем саду"],
    "Эрмитаж": ["эрмитаж", "эрмитажа", "эрмитаже"],
    "Исаакиевский собор": ["исаакиевский собор", "исаакиевского собора", "исаакий"],
    "Казанский собор": ["казанский собор", "казанского собора"],
    "Дворцовая площадь": ["дворцовая площадь", "дворцовой площади"],
    "Медный всадник": ["медный всадник", "медного всадника"],
    "Невский проспект": ["невский проспект", "невского проспекта", "невском проспекте"],
}

LAT2CYR = str.maketrans({
    "a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х", "y": "у",
    "m": "м", "i": "и", "t": "т", "j": "й", "k": "к", "h": "х", "n": "н",
    "b": "в", "u": "и", "s": "с", "d": "д", "l": "л", "r": "р", "z": "з",
})


def normalize(text: str) -> str:
    return text.lower().replace("ё", "е").translate(LAT2CYR).strip()


ALIASES = []
ALIAS_TO_CANON = []
for canon, forms in LANDMARKS.items():
    for form in forms:
        ALIASES.append(normalize(form))
        ALIAS_TO_CANON.append(canon)


def resolve_place(candidate, question, cutoff=85):
    """Возвращает (каноническое_имя, оценка, способ) или None."""
    if candidate:
        match = process.extractOne(
            normalize(candidate), ALIASES, scorer=fuzz.ratio, score_cutoff=cutoff
        )
        if match:
            _, score, idx = match
            return ALIAS_TO_CANON[idx], round(score), "по ответу модели"

    q = normalize(question)
    best = None
    for alias, canon in zip(ALIASES, ALIAS_TO_CANON):
        if len(alias) < 5:
            continue
        score = fuzz.partial_ratio(alias, q)
        if score >= 90 and (best is None or score > best[1]):
            best = (canon, round(score), "по тексту вопроса")
    return best