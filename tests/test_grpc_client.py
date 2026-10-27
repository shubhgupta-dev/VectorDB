import time
import grpc

import vectordb_pb2 as pb
import vectordb_pb2_grpc as rpc


def main():
    channel = grpc.insecure_channel("127.0.0.1:50051")
    grpc.channel_ready_future(channel).result(timeout=5)
    client = rpc.VectorDBStub(channel)

    base_id = time.time_ns() % (2**60)

    # 1. Insert one vector with metadata.
    response = client.Insert(
        pb.InsertRequest(
            record=pb.VectorRecord(
                id=base_id,
                values=[1.0, 0.0, 0.0],
                metadata={"category": "fruit", "color": "red"},
            )
        )
    )
    assert response.success, response.message
    print("Single insert: PASS")

    # 2. Batch insert two more vectors.
    batch = pb.BatchInsertRequest(
        records=[
            pb.VectorRecord(
                id=base_id + 1,
                values=[0.9, 0.1, 0.0],
                metadata={"category": "fruit", "color": "green"},
            ),
            pb.VectorRecord(
                id=base_id + 2,
                values=[0.0, 1.0, 0.0],
                metadata={"category": "vegetable", "color": "green"},
            ),
        ]
    )
    batch_response = client.InsertBatch(batch)
    assert batch_response.inserted_count == 2
    print("Batch insert: PASS")

    # 3. Search with a metadata filter.
    results = client.Search(
        pb.SearchRequest(
            query=[1.0, 0.0, 0.0],
            k=10,
            metadata_filter={"category": "fruit"},
        )
    )

    result_ids = {item.id for item in results.results}
    assert base_id in result_ids
    assert base_id + 1 in result_ids
    assert base_id + 2 not in result_ids
    print("Nearest-neighbor search + metadata filtering: PASS")

    for item in results.results:
        print(f"  ID: {item.id}, distance: {item.distance:.6f}")

    # 4. Retrieve index statistics.
    stats = client.GetStats(pb.StatsRequest())
    assert stats.dimension == 3
    assert stats.vector_count >= 3
    print(
        f"Stats: PASS (vectors={stats.vector_count}, "
        f"dimension={stats.dimension}, metric={stats.metric})"
    )

    # 5. Request a snapshot through gRPC.
    snapshot = client.CreateSnapshot(pb.SnapshotRequest())
    assert snapshot.success, snapshot.message
    print("Snapshot via gRPC: PASS")

    channel.close()
    print("\nALL GRPC TESTS PASSED")


if __name__ == "__main__":
    main()
