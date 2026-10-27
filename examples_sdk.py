from uuid import uuid4

from vectordb import VectorDBClient, VectorRecord


def main():
    # Generate fresh IDs on every run to avoid persistent-store collisions.
    base_id = uuid4().int & ((1 << 62) - 1)

    with VectorDBClient() as db:
        print("Connected to VectorDB")
        print(f"Initial stats: {db.get_stats()}")

        db.insert(
            base_id,
            [1.0, 0.0, 0.0],
            metadata={"category": "fruit", "color": "red"},
        )

        inserted = db.insert_batch([
            VectorRecord(
                id=base_id + 1,
                vector=[0.9, 0.1, 0.0],
                metadata={"category": "fruit", "color": "green"},
            ),
            VectorRecord(
                id=base_id + 2,
                vector=[0.0, 1.0, 0.0],
                metadata={"category": "vegetable"},
            ),
        ])
        print(f"Batch insert count: {inserted}")

        results = db.search(
            [1.0, 0.0, 0.0],
            k=5,
            metadata_filter={"category": "fruit"},
        )

        print("\nSearch results:")
        for result in results:
            print(f"  ID={result.id}, distance={result.distance:.6f}")

        print(f"\nFinal stats: {db.get_stats()}")

        db.create_snapshot()
        print("Snapshot created.")


if __name__ == "__main__":
    main()
