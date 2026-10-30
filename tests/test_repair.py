import sqlite3

import pytest

from eval.repair import repair


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "t.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE singer (singer_id INTEGER, name TEXT, country TEXT, age INTEGER);
        CREATE TABLE concert (concert_id INTEGER, singer_id INTEGER, year INTEGER);
        INSERT INTO singer VALUES (1, 'Ann', 'France', 30);
        """
    )
    conn.commit()
    conn.close()
    return path


def test_glued_keywords_are_split(db):
    assert repair(db, "SELECT count(*) FROMsinger") == "SELECT count(*) FROM singer"


def test_near_miss_table_is_fixed(db):
    assert repair(db, "SELECT count(*) FROM singers") == "SELECT count(*) FROM singer"


def test_near_miss_column_is_fixed(db):
    assert repair(db, "SELECT nam FROM singer") == "SELECT name FROM singer"


def test_alias_prefixed_column_is_fixed(db):
    sql = "SELECT T1.nam FROM singer AS T1"
    assert repair(db, sql) == "SELECT T1.name FROM singer AS T1"


def test_hopeless_names_and_valid_sql_are_left_alone(db):
    assert repair(db, "SELECT foo FROM singer") == "SELECT foo FROM singer"
    assert repair(db, "SELECT name FROM singer") == "SELECT name FROM singer"
