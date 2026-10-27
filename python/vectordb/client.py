"""High-level Python client for the VectorDB gRPC service."""

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import grpc

from .generated import vectordb_pb2 as pb
from .generated import vectordb_pb2_grpc as rpc


@dataclass(frozen=True)
class VectorRecord:
    """A vector and its optional string metadata."""

    id: int
    vector: Sequence[float]
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    """A nearest-neighbor result. Lower distance means closer."""

    id: int
    distance: float


@dataclass(frozen=True)
class IndexStats:
    """Basic information about the running index."""

    vector_count: int
    dimension: int
    metric: str


class VectorDBClient:
    """Convenient Python interface to a running VectorDB server."""

    def __init__(
        self,
        target: str = "127.0.0.1:50051",
        timeout: float = 5.0,
    ) -> None:
        if not target:
            raise ValueError("target must not be empty")
        if timeout <= 0:
            raise ValueError("timeout must be positive")

        self._channel = grpc.insecure_channel(target)
        self._stub = rpc.VectorDBStub(self._channel)
        self._timeout = timeout
        self._closed = False

    def connect(self) -> None:
        """Wait until the server is reachable."""
        self._ensure_open()
        grpc.channel_ready_future(self._channel).result(
            timeout=self._timeout
        )

    def insert(
        self,
        id: int,
        vector: Sequence[float],
        metadata: Mapping[str, str] | None = None,
    ) -> None:
        """Insert one vector with optional metadata."""
        self._ensure_open()
        record = self._to_proto(
            VectorRecord(id=id, vector=vector, metadata=metadata or {})
        )
        response = self._stub.Insert(
            pb.InsertRequest(record=record),
            timeout=self._timeout,
        )
        if not response.success:
            raise RuntimeError(response.message or "Vector insertion failed")

    def insert_batch(self, records: Iterable[VectorRecord]) -> int:
        """Insert multiple vectors and return the reported inserted count."""
        self._ensure_open()
        records = list(records)

        request = pb.BatchInsertRequest(
            records=[self._to_proto(record) for record in records]
        )
        response = self._stub.InsertBatch(
            request,
            timeout=self._timeout,
        )
        return response.inserted_count

    def search(
        self,
        query: Sequence[float],
        k: int = 5,
        metadata_filter: Mapping[str, str] | None = None,
    ) -> list[SearchResult]:
        """Search for the k nearest vectors, optionally filtering metadata."""
        self._ensure_open()
        if k < 1:
            raise ValueError("k must be at least 1")

        response = self._stub.Search(
            pb.SearchRequest(
                query=list(query),
                k=k,
                metadata_filter=dict(metadata_filter or {}),
            ),
            timeout=self._timeout,
        )
        return [
            SearchResult(id=result.id, distance=result.distance)
            for result in response.results
        ]

    def get_stats(self) -> IndexStats:
        """Return vector count, dimension, and configured metric."""
        self._ensure_open()
        response = self._stub.GetStats(
            pb.StatsRequest(),
            timeout=self._timeout,
        )
        return IndexStats(
            vector_count=response.vector_count,
            dimension=response.dimension,
            metric=response.metric,
        )

    def create_snapshot(self) -> None:
        """Ask the server to create a persistence snapshot."""
        self._ensure_open()
        response = self._stub.CreateSnapshot(
            pb.SnapshotRequest(),
            timeout=self._timeout,
        )
        if not response.success:
            raise RuntimeError(response.message or "Snapshot creation failed")

    @staticmethod
    def _to_proto(record: VectorRecord) -> pb.VectorRecord:
        return pb.VectorRecord(
            id=record.id,
            values=list(record.vector),
            metadata=dict(record.metadata),
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("VectorDBClient is closed")

    def close(self) -> None:
        """Close the underlying gRPC channel."""
        if not self._closed:
            self._channel.close()
            self._closed = True

    def __enter__(self) -> "VectorDBClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
