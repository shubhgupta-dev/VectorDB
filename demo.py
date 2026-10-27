"""Repeatable end-to-end demonstration of VectorDB."""

# import shutil
# from pathlib import Path

from vectordb import VectorDBClient, VectorRecord

HOST = "127.0.0.1:50053"
DIMENSION = 3
STORE_DIR = "data/interview_demo"


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def main() -> None:
    # # This is disposable demo data, separate from the normal server store.
    # if STORE_DIR.exists():
    #     shutil.rmtree(STORE_DIR)

    print("VectorDB — Interview Demo")
    print(f"Expected server: {HOST}")
    print(f"Demo data directory: {STORE_DIR}")

    with VectorDBClient(HOST) as db:
        section("1. Initial index stats")
        print(db.get_stats())

        section("2. Insert a single vector")
        db.insert(
            1001,
            [1.0, 0.0, 0.0],
            metadata={"category": "fruit", "color": "red", "name": "apple"},
        )
        print("Inserted ID 1001: apple")

        section("3. Batch ingestion")
        count = db.insert_batch([
            VectorRecord(
                id=1002,
                vector=[0.95, 0.05, 0.0],
                metadata={"category": "fruit", "color": "green", "name": "pear"},
            ),
            VectorRecord(
                id=1003,
                vector=[0.0, 1.0, 0.0],
                metadata={"category": "vegetable", "name": "cucumber"},
            ),
            VectorRecord(
                id=1004,
                vector=[0.0, 0.0, 1.0],
                metadata={"category": "fruit", "color": "yellow", "name": "banana"},
            ),
        ])
        print(f"Batch inserted: {count} vectors")

        section("4. Nearest-neighbor search")
        results = db.search([1.0, 0.0, 0.0], k=4)
        for result in results:
            print(f"ID={result.id:<5} cosine_distance={result.distance:.6f}")

        section("5. Metadata-filtered search")
        filtered = db.search(
            [1.0, 0.0, 0.0],
            k=4,
            metadata_filter={"category": "fruit"},
        )
        print("Filter: category=fruit")
        for result in filtered:
            print(f"ID={result.id:<5} cosine_distance={result.distance:.6f}")

        returned_ids = {result.id for result in filtered}
        assert returned_ids == {1001, 1002, 1004}, returned_ids
        print("Metadata filter verification: PASS")

        section("6. Index stats")
        stats = db.get_stats()
        print(stats)
        assert stats.vector_count == 4
        assert stats.dimension == DIMENSION
        print("Stats verification: PASS")

        section("7. Snapshot")
        db.create_snapshot()
        print("Snapshot RPC: PASS")

    section("Demo complete")
    print("All demo checks passed.")


if __name__ == "__main__":
    main()
