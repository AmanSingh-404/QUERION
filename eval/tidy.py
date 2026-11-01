import re

QUOTED = re.compile(r"(['\"])(.*?)\1")


def tidy_literals(sql):
    """Trim stray spaces just inside quotes: " Ward" becomes "Ward"."""
    return QUOTED.sub(lambda m: m.group(1) + m.group(2).strip() + m.group(1), sql)
