"""Optional PostgreSQL integration fixtures, always isolated in generated schemas."""

import os
from uuid import uuid4

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo
from psycopg.rows import dict_row
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver


@pytest.fixture
def postgres_url():
    base = os.getenv("SUPPORT_TEST_DATABASE_URL")
    if not base:
        pytest.skip("Set SUPPORT_TEST_DATABASE_URL to run disposable PostgreSQL integration checks.")
    schema = "workflow_test_" + uuid4().hex
    with psycopg.connect(base, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    try:
        yield make_conninfo(base, options=f"-c search_path={schema}")
    finally:
        with psycopg.connect(base, autocommit=True) as connection:
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


@pytest.fixture
def postgres_saver(postgres_url):
    with psycopg.connect(postgres_url, autocommit=True, prepare_threshold=0, row_factory=dict_row) as connection:
        saver = PostgresSaver(connection)
        saver.setup()
        yield saver


@pytest.fixture(params=["memory", "postgres"])
def checkpoint_backend(request):
    if request.param == "memory":
        return InMemorySaver()
    return request.getfixturevalue("postgres_saver")
