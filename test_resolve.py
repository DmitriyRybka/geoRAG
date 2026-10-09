from landmarks import resolve_place

cases = [
    ("Эрmitaj", "Ресторан в километре от Эрмитажа"),
    ("Летний сад", "Найди кафе в 500 метрах от Летнего сада"),
    ("Летнем саду", "Парк в Летнем саду"),
    (None, "Где поесть рядом с Исаакиевским собором"),
    ("Пушкин", "Кафе рядом с Пушкиным"),
    (None, "Парк для прогулки в центре"),
]

for candidate, question in cases:
    print(candidate, "|", question, "->", resolve_place(candidate, question))