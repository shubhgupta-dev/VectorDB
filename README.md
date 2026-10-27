# VectorDB

A lightweight vector search engine built in C++ with HNSW approximate nearest-neighbor (ANN) indexing, a write-ahead log (WAL), snapshots, a gRPC API, and a Python SDK.

This interview-focused systems project demonstrates vector indexing, persistence and recovery, API design, testing, and benchmark methodology. It is a learning/demo implementation, not a production-ready database.

## Features

- HNSW approximate nearest-neighbor search using [hnswlib](https://github.com/nmslib/hnswlib).
- Cosine distance, squared L2 distance, and inner product distance.
- Persistent records with WAL replay and snapshot loading.
- String key-value metadata and exact-match metadata filters.
- Single-vector and batch insertion.
- gRPC API for insertion, search, statistics, and snapshot creation.
- Python SDK with context-manager support.
- Automated SDK tests and an HNSW-versus-exact-search benchmark.

## Architecture

```text
Python application / demo
          |
          v
      Python SDK
          |
          v
  gRPC + Protocol Buffers
          |
          v
      C++ service
          |
          v
   PersistentStore -------> WAL + snapshot files
          |
          v
   VectorIndex (hnswlib)
          |
          v
  HNSW nearest-neighbor search
```

The persistent store owns vectors and metadata, while the vector index provides nearest-neighbor candidates. Metadata filtering is applied by the store after candidate retrieval.

## Repository layout

```text
include/vectordb/       C++ public headers
src/                    C++ index, persistence, and gRPC server
proto/                  Protocol Buffers API definition
python/vectordb/        Python SDK and generated bindings
tests/                  SDK tests and gRPC test helpers
benchmarks/             Benchmark source and recorded results
third_party/hnswlib/    HNSW index dependency
```

## Requirements

Developed and tested with WSL2/Debian, CMake, GCC, gRPC, Protocol Buffers, and Python.

Install system dependencies on Debian/Ubuntu:

```bash
sudo apt update
sudo apt install -y build-essential cmake git \
    libgrpc++-dev libprotobuf-dev protobuf-compiler protobuf-compiler-grpc \
    python3 python3-venv python3-pip
```

Ensure `third_party/hnswlib/` is present. Create the Python environment and install the SDK:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Build the C++ targets:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
```

## Run the server and tests

Start the default server in one terminal:

```bash
./build/vectordb_server
```

Defaults: `127.0.0.1:50051`, dimension `3`, cosine distance, and storage directory `data/server_store`.

In a second terminal, activate the virtual environment and run the SDK tests:

```bash
source .venv/bin/activate
python -m pytest tests/test_sdk.py -v
```

The server must be running on `127.0.0.1:50051` for these integration tests.

The standalone gRPC smoke test can be run with:

```bash
PYTHONPATH="$PWD/python/vectordb/generated" python tests/test_grpc_client.py
```

### Server configuration

The server accepts these optional positional arguments:

```text
./build/vectordb_server [address] [dimension] [metric] [storage_directory]
```

Supported metrics: `cosine`, `l2`, and `inner_product`.

Example: run a 64-dimensional cosine index on a separate port and storage directory:

```bash
./build/vectordb_server 127.0.0.1:50052 64 cosine data/server_store_64
```

Keep the dimension and metric consistent with a storage directory across restarts. Use a new directory when experimenting with another configuration. The server currently uses insecure gRPC credentials and is intended for local development.

## End-to-end demo

The demo inserts a vector, batch-inserts three more, performs nearest-neighbor and metadata-filtered searches, checks statistics, and requests a snapshot.

1. Stop any previous server using port `50053`.
2. Prepare the dedicated demo directory **before** starting the server:

   ```bash
   rm -rf data/interview_demo
   mkdir -p data/interview_demo
   ```

3. Start the server in one terminal:

   ```bash
   ./build/vectordb_server 127.0.0.1:50053 3 cosine data/interview_demo
   ```

4. In a second terminal, run:

   ```bash
   source .venv/bin/activate
   python demo.py
   ```

The demo should finish with `All demo checks passed.` It uses fixed record IDs and a dedicated, disposable storage directory. Never delete or replace that directory while the server is running. To reset the demo, stop the server first, recreate the directory, and restart the server.

## ANN benchmark

The benchmark compares HNSW approximate search with brute-force exact search on synthetic vectors. The following results are medians across three runs on the same development machine.

### Setup

| Parameter | Value |
|---|---:|
| Dataset size | 5,000 vectors |
| Vector dimensions | 64 |
| Queries per run | 500 |
| Neighbors requested (`k`) | 10 |
| Random seed | 42 |
| HNSW `M` | 16 |
| HNSW `ef_construction` | 100 |
| `ef_search` values | 25, 50, 100, 200 |
| Runs per configuration | 3 |

### Results

Latency values are in milliseconds.

| `ef_search` | Recall@10 | p50 latency | p95 latency | p99 latency | Effective serial QPS |
|---:|---:|---:|---:|---:|---:|
| 25 | 67.58% | 0.0204 | 0.0427 | 0.0564 | 40,021.5 |
| 50 | 85.16% | 0.0533 | 0.1431 | 0.2161 | 15,412.3 |
| 100 | 95.82% | 0.0551 | 0.0869 | 0.1292 | 16,631.8 |
| 200 | 99.60% | 0.1003 | 0.1673 | 0.4146 | 8,495.2 |

### Methodology and limitations

- **Recall@10** is the average fraction of exact top-10 neighbors also returned by HNSW.
- Latency percentiles measure individual HNSW searches and exclude exact-search work used to calculate recall.
- **Effective serial QPS** is the number of queries divided by the sum of measured HNSW search durations. It is not concurrent server throughput and excludes gRPC and client overhead.
- The dataset is synthetic and contains only 5,000 vectors. These results do not establish performance on larger or real-world datasets.
- Results depend on the machine and workload. Latency percentiles can vary between runs and are not production guarantees.

Rebuild and run one configuration:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2 --target vectordb_benchmark
./build/vectordb_benchmark 5000 500 64 100
```

Append results from three runs per configuration to a CSV:

```bash
rm -f benchmarks/repeated_results.csv

for run in 1 2 3; do
    for ef in 25 50 100 200; do
        ./build/vectordb_benchmark 5000 500 64 "$ef" \
            benchmarks/repeated_results.csv
    done
done
```

## Current limitations

- Metadata filtering happens after retrieving indexed candidates; there is no dedicated metadata index, so filtering can be expensive as the dataset grows.
- Batch insertion delegates to individual inserts and is **not atomic**. A failure partway through can leave earlier records committed.
- WAL writes are flushed through the C++ stream but do not explicitly call `fsync`; this does not guarantee recovery from every power-loss scenario.
- Snapshot creation does not compact or rotate the WAL.
- The server uses insecure gRPC transport and is not intended for exposure to untrusted networks.
- The benchmark uses synthetic data and does not measure concurrent gRPC throughput.

These limitations are documented intentionally so the project can be discussed honestly in interviews and improved incrementally.
