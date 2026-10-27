from uuid import uuid4

import pytest

from vectordb import (
    IndexStats,
    SearchResult,
    VectorDBClient,
    VectorRecord,
)


def unique_id():
    return uuid4().int & ((1 << 62) - 1)


def test_insert_and_filtered_search():
    token = uuid4().hex
    vector_id = unique_id()

    with VectorDBClient() as db:
        db.insert(
            vector_id,
            [1.0, 0.0, 0.0],
            metadata={"test_run": token, "category": "fruit"},
        )

        results = db.search(
            [1.0, 0.0, 0.0],
            k=5,
            metadata_filter={"test_run": token},
        )

    assert len(results) == 1
    assert isinstance(results[0], SearchResult)
    assert results[0].id == vector_id
    assert results[0].distance == pytest.approx(0.0, abs=1e-5)


def test_batch_insert():
    token = uuid4().hex
    first_id = unique_id()

    records = [
        VectorRecord(
            id=first_id,
            vector=[1.0, 0.0, 0.0],
            metadata={"test_run": token},
        ),
        VectorRecord(
            id=first_id + 1,
            vector=[0.0, 1.0, 0.0],
            metadata={"test_run": token},
        ),
    ]

    with VectorDBClient() as db:
        inserted = db.insert_batch(records)
        results = db.search(
            [1.0, 0.0, 0.0],
            k=5,
            metadata_filter={"test_run": token},
        )

    assert inserted == 2
    assert {result.id for result in results} == {
        first_id,
        first_id + 1,
    }


def test_stats():
    with VectorDBClient() as db:
        stats = db.get_stats()

    assert isinstance(stats, IndexStats)
    assert stats.dimension == 3
    assert stats.metric == "cosine"
    assert stats.vector_count >= 1


def test_snapshot():
    with VectorDBClient() as db:
        db.create_snapshot()


def test_search_rejects_invalid_k():
    with VectorDBClient() as db:
        with pytest.raises(ValueError, match="at least 1"):
            db.search([1.0, 0.0, 0.0], k=0)


def test_closed_client_rejects_requests():
    db = VectorDBClient()
    db.close()

    with pytest.raises(RuntimeError, match="closed"):
        db.get_stats()
