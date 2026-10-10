"""Small read-only PostgreSQL access layer used by billing tools."""

import os

import psycopg
from psycopg.rows import dict_row


def connect_db():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set. Copy .env.example to .env and start PostgreSQL.")
    return psycopg.connect(database_url, row_factory=dict_row)


def query_one(sql: str, params: tuple = ()) -> dict | None:
    with connect_db() as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql, params)
            row = cursor.fetchone()
            return dict(row) if row else None


def query_all(sql: str, params: tuple = ()) -> list[dict]:
    with connect_db() as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql, params)
            return [dict(row) for row in cursor.fetchall()]
