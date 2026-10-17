from tokenizer.normalize import normalize

BOS = "<|bos|>"
EOS = "<|eos|>"
SCHEMA = "<|schema|>"
QUESTION = "<|question|>"
SQL = "<|sql|>"


def statements_from_gretel(context):
    """Keep only CREATE TABLE / CREATE VIEW statements (drop INSERTs and the rest)."""
    out = []
    for part in context.split(";"):
        s = normalize(part)
        if s.lower().startswith(("create table", "create view")):
            out.append(s)
    return out


def statements_from_lines(text):
    """One statement per line, no trailing semicolons (the synthetic generator's format)."""
    out = []
    for line in text.split("\n"):
        s = normalize(line).rstrip(";").rstrip()
        if s:
            out.append(s)
    return out


def clean_sql(sql):
    s = normalize(sql)
    while s.endswith(";"):
        s = s[:-1].rstrip()
    return s


def format_prompt(statements, question):
    schema = "\n".join(statements)
    return f"{BOS}{SCHEMA}{schema}{QUESTION}{normalize(question)}{SQL}"


def format_example(statements, question, sql):
    return format_prompt(statements, question) + clean_sql(sql) + EOS
