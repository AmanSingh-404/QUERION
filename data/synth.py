import json
import random
import sqlite3
import sys

NAMES = (
    "Alice Bob Carlos Dana Emma Farid Grace Hiro Ivan Julia Kofi Lena Mateo "
    "Nina Omar Priya Quinn Rosa Sam Tara"
).split()
CITIES = (
    "Paris Berlin Madrid Rome Lisbon Oslo Vienna Dublin Prague Warsaw "
    "Athens Cairo Lagos Nairobi Lima Quito Seoul Osaka"
).split()
COUNTRIES = (
    "France Germany Spain Italy Portugal Norway Austria Ireland Poland "
    "Greece Egypt Nigeria Kenya Peru Japan Brazil"
).split()
DEPTS = "Sales Marketing Finance Research Support Legal Design Operations".split()
STATUS = "active pending closed cancelled completed".split()
GENRES = "drama comedy thriller romance fantasy history science horror".split()
MAJORS = "Physics History Biology Chemistry Economics Law Music Math".split()
SUBJECTS = "Algebra Biology Chemistry Geography Literature Physics Art".split()
SPECIALTIES = "cardiology dermatology neurology pediatrics oncology surgery".split()
POSITIONS = "goalkeeper defender midfielder forward".split()
CUISINES = "italian thai mexican indian japanese french greek".split()
PLANS = "basic premium family student".split()
CROPS = "wheat corn rice barley soy cotton oats".split()
SEASONS = "spring summer autumn winter".split()
VENUES = "NeurIPS ICML ACL CVPR SIGMOD VLDB KDD".split()
DAYS = "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split()
MEMBERSHIPS = "bronze silver gold platinum".split()
TITLES = (
    "Blue River|Silent Echo|Golden Gate|Hidden Path|Red Horizon|Iron Bridge|"
    "Paper Moon|Last Harbor|Wild Garden|Broken Mirror|Quiet Storm|Bright Star|"
    "Distant Shore|Open Window|Frozen Lake|Falling Leaves|Rising Tide|Lost Key"
).split("|")
STREETS = "Oak Street|Maple Avenue|Pine Road|Cedar Lane|Elm Court|River Drive".split("|")


def TXT(pool):
    return ("t", pool)


def INT(lo, hi):
    return ("i", lo, hi)


def REAL(lo, hi):
    return ("r", lo, hi)


FK = ("fk",)

DOMAINS = [
    [
        (
            "employees",
            {
                "name": TXT(NAMES),
                "department": TXT(DEPTS),
                "salary": INT(30000, 150000),
                "hire_year": INT(2005, 2024),
                "city": TXT(CITIES),
            },
        ),
        (
            "projects",
            {
                "title": TXT(TITLES),
                "budget": INT(10000, 500000),
                "status": TXT(STATUS),
                "employee_id": FK,
            },
        ),
    ],
    [
        (
            "students",
            {
                "name": TXT(NAMES),
                "major": TXT(MAJORS),
                "gpa": REAL(1.5, 4.0),
                "enrollment_year": INT(2015, 2024),
                "city": TXT(CITIES),
            },
        ),
        (
            "grades",
            {
                "subject": TXT(SUBJECTS),
                "score": INT(40, 100),
                "semester": INT(1, 8),
                "student_id": FK,
            },
        ),
    ],
    [
        (
            "customers",
            {
                "name": TXT(NAMES),
                "city": TXT(CITIES),
                "age": INT(18, 80),
                "membership": TXT(MEMBERSHIPS),
            },
        ),
        (
            "orders",
            {
                "total": INT(5, 900),
                "status": TXT(STATUS),
                "order_year": INT(2018, 2025),
                "customer_id": FK,
            },
        ),
    ],
    [
        ("authors", {"name": TXT(NAMES), "country": TXT(COUNTRIES), "birth_year": INT(1930, 1995)}),
        (
            "books",
            {
                "title": TXT(TITLES),
                "genre": TXT(GENRES),
                "pages": INT(80, 900),
                "price": INT(5, 60),
                "author_id": FK,
            },
        ),
    ],
    [
        (
            "directors",
            {"name": TXT(NAMES), "country": TXT(COUNTRIES), "birth_year": INT(1940, 1990)},
        ),
        (
            "films",
            {
                "title": TXT(TITLES),
                "genre": TXT(GENRES),
                "year": INT(1990, 2025),
                "rating": REAL(3.0, 9.8),
                "budget": INT(1, 300),
                "director_id": FK,
            },
        ),
    ],
    [
        (
            "doctors",
            {
                "name": TXT(NAMES),
                "specialty": TXT(SPECIALTIES),
                "years_experience": INT(1, 40),
                "city": TXT(CITIES),
            },
        ),
        (
            "appointments",
            {
                "patient_name": TXT(NAMES),
                "fee": INT(20, 400),
                "status": TXT(STATUS),
                "month": INT(1, 12),
                "doctor_id": FK,
            },
        ),
    ],
    [
        ("airlines", {"name": TXT(TITLES), "country": TXT(COUNTRIES), "fleet_size": INT(5, 400)}),
        (
            "flights",
            {
                "origin": TXT(CITIES),
                "destination": TXT(CITIES),
                "price": INT(50, 1500),
                "duration_minutes": INT(45, 900),
                "airline_id": FK,
            },
        ),
    ],
    [
        (
            "teams",
            {
                "name": TXT(TITLES),
                "city": TXT(CITIES),
                "founded_year": INT(1880, 2010),
                "stadium_capacity": INT(5000, 90000),
            },
        ),
        (
            "players",
            {
                "name": TXT(NAMES),
                "position": TXT(POSITIONS),
                "goals": INT(0, 40),
                "age": INT(17, 40),
                "team_id": FK,
            },
        ),
    ],
    [
        (
            "artists",
            {
                "name": TXT(NAMES),
                "country": TXT(COUNTRIES),
                "genre": TXT(GENRES),
                "debut_year": INT(1970, 2022),
            },
        ),
        (
            "albums",
            {
                "title": TXT(TITLES),
                "year": INT(1975, 2025),
                "sales": INT(1, 5000),
                "rating": REAL(2.0, 5.0),
                "artist_id": FK,
            },
        ),
    ],
    [
        (
            "restaurants",
            {
                "name": TXT(TITLES),
                "cuisine": TXT(CUISINES),
                "city": TXT(CITIES),
                "rating": REAL(2.0, 5.0),
                "seats": INT(10, 200),
            },
        ),
        (
            "reviews",
            {
                "reviewer": TXT(NAMES),
                "stars": INT(1, 5),
                "visit_year": INT(2015, 2025),
                "restaurant_id": FK,
            },
        ),
    ],
    [
        ("agents", {"name": TXT(NAMES), "city": TXT(CITIES), "experience_years": INT(1, 35)}),
        (
            "listings",
            {
                "street": TXT(STREETS),
                "price": INT(50, 2000),
                "bedrooms": INT(1, 6),
                "area_sqft": INT(300, 4000),
                "status": TXT(STATUS),
                "agent_id": FK,
            },
        ),
    ],
    [
        (
            "members",
            {
                "name": TXT(NAMES),
                "age": INT(16, 75),
                "plan": TXT(PLANS),
                "join_year": INT(2015, 2025),
            },
        ),
        (
            "sessions",
            {
                "trainer": TXT(NAMES),
                "minutes": INT(15, 120),
                "calories": INT(50, 900),
                "day": TXT(DAYS),
                "member_id": FK,
            },
        ),
    ],
    [
        ("farmers", {"name": TXT(NAMES), "region": TXT(CITIES), "farm_size": INT(2, 500)}),
        (
            "crops",
            {
                "crop": TXT(CROPS),
                "yield_tons": INT(1, 300),
                "price_per_ton": INT(80, 900),
                "season": TXT(SEASONS),
                "farmer_id": FK,
            },
        ),
    ],
    [
        (
            "professors",
            {
                "name": TXT(NAMES),
                "department": TXT(MAJORS),
                "salary": INT(40000, 180000),
                "years": INT(1, 40),
            },
        ),
        (
            "papers",
            {
                "title": TXT(TITLES),
                "year": INT(1995, 2025),
                "citations": INT(0, 2000),
                "venue": TXT(VENUES),
                "professor_id": FK,
            },
        ),
    ],
]

SQL_TYPE = {"t": "TEXT", "i": "INTEGER", "r": "REAL", "fk": "INTEGER"}
AGG_WORD = {"AVG": "average", "MAX": "maximum", "MIN": "minimum", "SUM": "total"}


def agg_funcs(col):
    if "year" in col:
        return ["MAX", "MIN"]
    return list(AGG_WORD)


def human(col):
    return col.replace("_", " ")


def singular(name):
    return name[:-1] if name.endswith("s") else name


def gen_value(rng, spec):
    if spec[0] == "t":
        return rng.choice(spec[1])
    if spec[0] == "i":
        return rng.randint(spec[1], spec[2])
    return round(rng.uniform(spec[1], spec[2]), 1)


def make_instance(rng):
    dom = rng.choice(DOMAINS)
    two = rng.random() < 0.5
    chosen = dom if two else [rng.choice(dom)]
    conn = sqlite3.connect(":memory:")
    tables = {}
    ddl = []
    for idx, (name, cols) in enumerate(chosen):
        base = [c for c, s in cols.items() if s[0] != "fk"]
        keep = set(rng.sample(base, rng.randint(min(3, len(base)), len(base))))
        use = {c: s for c, s in cols.items() if c in keep or (s[0] == "fk" and two)}
        parent = chosen[0][0] if (two and idx == 1) else None
        fk = next((c for c, s in use.items() if s[0] == "fk"), None)
        n = rng.randint(12, 25)
        parts = ["id INTEGER PRIMARY KEY"] + [f"{c} {SQL_TYPE[s[0]]}" for c, s in use.items()]
        if fk:
            parts.append(f"FOREIGN KEY ({fk}) REFERENCES {parent}(id)")
        stmt = f"CREATE TABLE {name} ({', '.join(parts)})"
        conn.execute(stmt)
        ddl.append(stmt)
        parent_n = tables[parent]["n"] if parent else 0
        for i in range(1, n + 1):
            row = [i]
            for _c, s in use.items():
                row.append(rng.randint(1, parent_n) if s[0] == "fk" else gen_value(rng, s))
            marks = ",".join("?" * len(row))
            conn.execute(f"INSERT INTO {name} VALUES ({marks})", row)
        tables[name] = {"cols": use, "n": n, "parent": parent, "fk": fk}
    return conn, tables, "\n".join(ddl)


def text_cols(t):
    return [c for c, s in t["cols"].items() if s[0] == "t"]


def num_cols(t):
    return [c for c, s in t["cols"].items() if s[0] in ("i", "r")]


def plain_cols(t):
    return [c for c, s in t["cols"].items() if s[0] != "fk"]


def column_values(conn, name, col):
    return [r[0] for r in conn.execute(f"SELECT {col} FROM {name}")]


def lit(v):
    return f"'{v}'" if isinstance(v, str) else str(v)


def say(rng, v):
    return f"'{v}'" if isinstance(v, str) and rng.random() < 0.5 else str(v)


def q_count_all(rng, conn, tables):
    name = rng.choice(list(tables))
    q = rng.choice(
        [
            f"How many {name} are there?",
            f"What is the total number of {name}?",
            f"Count the {name}.",
        ]
    )
    return "count_all", q, f"SELECT COUNT(*) FROM {name}"


def q_count_num(rng, conn, tables):
    name = rng.choice(list(tables))
    cols = num_cols(tables[name])
    if not cols:
        return None
    col = rng.choice(cols)
    v = rng.choice(column_values(conn, name, col))
    if rng.random() < 0.5:
        op, word = ">", rng.choice(["greater than", "above", "more than"])
    else:
        op, word = "<", rng.choice(["less than", "below", "under"])
    q = rng.choice(
        [
            f"How many {name} have {human(col)} {word} {v}?",
            f"Count the {name} with {human(col)} {word} {v}.",
        ]
    )
    return "count_num", q, f"SELECT COUNT(*) FROM {name} WHERE {col} {op} {v}"


def q_count_text(rng, conn, tables):
    name = rng.choice(list(tables))
    cols = text_cols(tables[name])
    if not cols:
        return None
    col = rng.choice(cols)
    v = rng.choice(column_values(conn, name, col))
    sv = say(rng, v)
    q = rng.choice(
        [
            f"How many {name} have {human(col)} equal to {sv}?",
            f"How many {name} are there with {human(col)} {sv}?",
        ]
    )
    return "count_text", q, f"SELECT COUNT(*) FROM {name} WHERE {col} = {lit(v)}"


def q_select_text(rng, conn, tables):
    name = rng.choice(list(tables))
    t = tables[name]
    cols = text_cols(t)
    if not cols:
        return None
    c1 = rng.choice(cols)
    others = [c for c in plain_cols(t) if c != c1]
    if not others:
        return None
    c2 = rng.choice(others)
    v = rng.choice(column_values(conn, name, c1))
    sv = say(rng, v)
    q = rng.choice(
        [
            f"List the {human(c2)} of {name} whose {human(c1)} is {sv}.",
            f"What is the {human(c2)} of every record in {name} with {human(c1)} {sv}?",
        ]
    )
    return "select_text", q, f"SELECT {c2} FROM {name} WHERE {c1} = {lit(v)}"


def q_select_num(rng, conn, tables):
    name = rng.choice(list(tables))
    t = tables[name]
    cols = num_cols(t)
    if not cols:
        return None
    c1 = rng.choice(cols)
    others = [c for c in plain_cols(t) if c != c1]
    if not others:
        return None
    c2 = rng.choice(others)
    v = rng.choice(column_values(conn, name, c1))
    q = rng.choice(
        [
            f"Show the {human(c2)} of {name} with {human(c1)} above {v}.",
            f"Which {name} have {human(c1)} greater than {v}? Give their {human(c2)}.",
        ]
    )
    return "select_num", q, f"SELECT {c2} FROM {name} WHERE {c1} > {v}"


def q_agg(rng, conn, tables):
    name = rng.choice(list(tables))
    cols = num_cols(tables[name])
    if not cols:
        return None
    col = rng.choice(cols)
    fn = rng.choice(agg_funcs(col))
    q = rng.choice(
        [
            f"What is the {AGG_WORD[fn]} {human(col)} of all {name}?",
            f"Find the {AGG_WORD[fn]} {human(col)} across {name}.",
        ]
    )
    return "agg", q, f"SELECT {fn}({col}) FROM {name}"


def q_order_limit(rng, conn, tables):
    name = rng.choice(list(tables))
    t = tables[name]
    cols = num_cols(t)
    if not cols:
        return None
    col = rng.choice(cols)
    others = [c for c in plain_cols(t) if c != col]
    if not others:
        return None
    c2 = rng.choice(others)
    k = rng.randint(2, 5)
    if rng.random() < 0.5:
        order, word = "DESC", "highest"
    else:
        order, word = "ASC", "lowest"
    q = rng.choice(
        [
            f"Show the {human(c2)} of the {k} {name} with the {word} {human(col)}.",
            f"What are the {human(c2)} values of the top {k} {name} by {word} {human(col)}?",
        ]
    )
    sql = f"SELECT {c2} FROM {name} ORDER BY {col} {order} LIMIT {k}"
    return f"order_limit:{k}", q, sql


def q_group_count(rng, conn, tables):
    name = rng.choice(list(tables))
    cols = text_cols(tables[name])
    if not cols:
        return None
    col = rng.choice(cols)
    q = rng.choice(
        [
            f"How many {name} are there for each {human(col)}?",
            f"Count the {name} grouped by {human(col)}.",
        ]
    )
    return "group_count", q, f"SELECT {col}, COUNT(*) FROM {name} GROUP BY {col}"


def q_group_agg(rng, conn, tables):
    name = rng.choice(list(tables))
    t = tables[name]
    cats, nums = text_cols(t), num_cols(t)
    if not cats or not nums:
        return None
    cat, num = rng.choice(cats), rng.choice(nums)
    fn = rng.choice(agg_funcs(num))
    q = rng.choice(
        [
            f"What is the {AGG_WORD[fn]} {human(num)} for each {human(cat)} in {name}?",
            f"For every {human(cat)}, find the {AGG_WORD[fn]} {human(num)} of {name}.",
        ]
    )
    return "group_agg", q, f"SELECT {cat}, {fn}({num}) FROM {name} GROUP BY {cat}"


def q_distinct(rng, conn, tables):
    name = rng.choice(list(tables))
    cols = text_cols(tables[name])
    if not cols:
        return None
    col = rng.choice(cols)
    q = rng.choice(
        [
            f"List all distinct {human(col)} values in {name}.",
            f"What different {human(col)} values appear in {name}?",
        ]
    )
    return "distinct", q, f"SELECT DISTINCT {col} FROM {name}"


def q_and_cond(rng, conn, tables):
    name = rng.choice(list(tables))
    t = tables[name]
    cats, nums = text_cols(t), num_cols(t)
    if not cats or not nums:
        return None
    c1, c2 = rng.choice(cats), rng.choice(nums)
    rows = conn.execute(f"SELECT {c1}, {c2} FROM {name}").fetchall()
    v1, x = rng.choice(rows)
    below = [r[1] for r in rows if r[1] < x]
    v2 = rng.choice(below) if below else x - 1
    s1 = say(rng, v1)
    q = rng.choice(
        [
            f"How many {name} have {human(c1)} equal to {s1} and {human(c2)} greater than {v2}?",
            f"Count the {name} with {human(c1)} {s1} and {human(c2)} above {v2}.",
        ]
    )
    sql = f"SELECT COUNT(*) FROM {name} WHERE {c1} = {lit(v1)} AND {c2} > {v2}"
    return "and_cond", q, sql


def q_between(rng, conn, tables):
    name = rng.choice(list(tables))
    t = tables[name]
    cols = num_cols(t)
    if not cols:
        return None
    c1 = rng.choice(cols)
    others = [c for c in plain_cols(t) if c != c1]
    if not others:
        return None
    c2 = rng.choice(others)
    vals = column_values(conn, name, c1)
    a, b = sorted([rng.choice(vals), rng.choice(vals)])
    if a == b:
        return None
    q = rng.choice(
        [
            f"Show the {human(c2)} of {name} with {human(c1)} between {a} and {b}.",
            f"Which {name} have {human(c1)} from {a} to {b}? List their {human(c2)}.",
        ]
    )
    return "between", q, f"SELECT {c2} FROM {name} WHERE {c1} BETWEEN {a} AND {b}"


def child_parent(tables):
    for name, t in tables.items():
        if t["parent"]:
            return name, t["parent"], t["fk"]
    return None


def q_join_list(rng, conn, tables):
    cp = child_parent(tables)
    if not cp:
        return None
    child, parent, fk = cp
    c = rng.choice(plain_cols(tables[child]))
    p = rng.choice(plain_cols(tables[parent]))
    q = rng.choice(
        [
            f"List the {human(c)} of each {singular(child)} along with the {human(p)} of its "
            f"{singular(parent)}.",
            f"For every {singular(child)}, show its {human(c)} and the {human(p)} of the "
            f"{singular(parent)} it belongs to.",
        ]
    )
    sql = f"SELECT T1.{c}, T2.{p} FROM {child} AS T1 JOIN {parent} AS T2 ON T1.{fk} = T2.id"
    return "join_list", q, sql


def q_join_where(rng, conn, tables):
    cp = child_parent(tables)
    if not cp:
        return None
    child, parent, fk = cp
    pcols = text_cols(tables[parent])
    if not pcols:
        return None
    pc = rng.choice(pcols)
    c = rng.choice(plain_cols(tables[child]))
    v = rng.choice(column_values(conn, parent, pc))
    sv = say(rng, v)
    q = rng.choice(
        [
            f"What are the {human(c)} of {child} whose {singular(parent)} has {human(pc)} {sv}?",
            f"List the {human(c)} of all {child} belonging to a {singular(parent)} with "
            f"{human(pc)} equal to {sv}.",
        ]
    )
    sql = (
        f"SELECT T1.{c} FROM {child} AS T1 JOIN {parent} AS T2 ON T1.{fk} = T2.id "
        f"WHERE T2.{pc} = {lit(v)}"
    )
    return "join_where", q, sql


def q_join_count(rng, conn, tables):
    cp = child_parent(tables)
    if not cp:
        return None
    child, parent, fk = cp
    pcols = text_cols(tables[parent])
    if not pcols:
        return None
    pc = rng.choice(pcols)
    q = rng.choice(
        [
            f"How many {child} does each {singular(parent)} have? Show the {human(pc)}.",
            f"For each {singular(parent)}, count their {child} and show the {human(pc)}.",
        ]
    )
    sql = (
        f"SELECT T2.{pc}, COUNT(*) FROM {child} AS T1 JOIN {parent} AS T2 "
        f"ON T1.{fk} = T2.id GROUP BY T2.id"
    )
    return "join_count", q, sql


KINDS = [
    q_count_all,
    q_count_num,
    q_count_text,
    q_select_text,
    q_select_num,
    q_agg,
    q_order_limit,
    q_group_count,
    q_group_agg,
    q_distinct,
    q_and_cond,
    q_between,
    q_join_list,
    q_join_where,
    q_join_count,
    q_join_list,
    q_join_where,
    q_join_count,
]


def verify(kind, rows, tables, name_hint=None):
    n = len(rows)
    if kind == "count_all":
        return rows[0][0] > 0
    if kind in ("count_num", "count_text", "and_cond"):
        return rows[0][0] > 0
    if kind == "agg":
        return rows[0][0] is not None
    if kind.startswith("order_limit"):
        return n == int(kind.split(":")[1])
    if kind in ("group_count", "group_agg", "join_count"):
        return n >= 2
    if kind == "distinct":
        return n >= 2
    if kind in ("select_text", "select_num", "between", "join_where"):
        return n > 0
    if kind == "join_list":
        return n > 0
    return False


def strict_subset_ok(kind, sql, rows, conn, tables):
    # filters must neither match nothing nor everything
    if kind in ("select_text", "select_num", "between"):
        name = sql.split(" FROM ")[1].split(" ")[0]
        return len(rows) < tables[name]["n"]
    if kind in ("count_num", "count_text", "and_cond"):
        name = sql.split(" FROM ")[1].split(" ")[0]
        return rows[0][0] < tables[name]["n"]
    if kind == "join_where":
        child = sql.split(" FROM ")[1].split(" ")[0]
        return len(rows) < tables[child]["n"]
    return True


def generate(n, seed=0, per_instance=8):
    rng = random.Random(seed)
    seen = set()
    out = []
    attempts = 0
    while len(out) < n and attempts < n * 50:
        attempts += 1
        conn, tables, ddl = make_instance(rng)
        made = 0
        for _ in range(per_instance * 3):
            if made >= per_instance or len(out) >= n:
                break
            res = rng.choice(KINDS)(rng, conn, tables)
            if res is None:
                continue
            kind, q, sql = res
            try:
                rows = conn.execute(sql).fetchall()
            except sqlite3.Error:
                continue
            if not rows or not verify(kind, rows, tables):
                continue
            if not strict_subset_ok(kind, sql, rows, conn, tables):
                continue
            key = (ddl, q, sql)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "id": len(out),
                    "schema": ddl,
                    "question": q,
                    "sql": sql,
                    "kind": kind.split(":")[0],
                }
            )
            made += 1
        conn.close()
    return out


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
    path = sys.argv[2] if len(sys.argv) > 2 else "/content/drive/MyDrive/querion/data/synth.jsonl"
    rows = generate(n)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    kinds = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print("pairs:", len(rows))
    print(json.dumps(kinds, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
