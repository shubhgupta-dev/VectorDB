import json
import math
import re
import secrets
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from vectordb import VectorDBClient, VectorRecord


ROOT = Path(__file__).resolve().parent
BENCHMARK = ROOT / "build" / "vectordb_benchmark"
BENCHMARK_CSV = ROOT / "data" / "ui_benchmarks.csv"

st.set_page_config(
    page_title="VectorDB Explorer",
    # page_icon
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.8rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {
        background: rgba(128,128,128,0.08);
        padding: 14px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("VectorDB Explorer")
st.caption(
    "An interactive showcase of approximate nearest-neighbor search, "
    "powered by a C++ HNSW index and a Python gRPC client."
)

with st.sidebar:
    st.header("Connection")
    target = st.text_input("gRPC server address", "127.0.0.1:50051")
    st.caption("Start vectordb_server in a separate terminal.")
    check = st.button("Check connection", use_container_width=True)

    if check:
        try:
            with VectorDBClient(target, timeout=3.0) as client:
                stats = client.get_stats()
            st.success("Server connected")
            st.session_state["server_stats"] = {
                "count": stats.vector_count,
                "dimension": stats.dimension,
                "metric": stats.metric,
            }
        except Exception as exc:
            st.error(f"Connection failed: {exc}")

    st.divider()
    st.markdown("**Architecture**")
    st.code("Streamlit UI\n      ↓\nPython SDK\n      ↓ gRPC\nC++ VectorDB server\n      ↓\nHNSW index + storage")


def get_stats():
    with VectorDBClient(target, timeout=3.0) as client:
        return client.get_stats()


def make_id():
    # Positive, unsigned 63-bit ID. Still check for duplicates in each upload.
    return secrets.randbits(63) + 1


def parse_metadata(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return {}
    if isinstance(value, dict):
        obj = value
    elif isinstance(value, str) and value.strip():
        obj = json.loads(value)
    else:
        return {}
    return {str(k): str(v) for k, v in obj.items()}


def parse_upload(uploaded, expected_dim):
    name = uploaded.name.lower()

    if name.endswith(".json"):
        payload = json.load(uploaded)
        if isinstance(payload, dict):
            payload = payload.get("records", [payload])
        if not isinstance(payload, list):
            raise ValueError("JSON must contain a list of vector records.")

        records = []
        for i, item in enumerate(payload):
            if not isinstance(item, dict) or "vector" not in item:
                raise ValueError(f"JSON record {i} needs a 'vector' array.")
            vector = [float(x) for x in item["vector"]]
            record_id = int(item.get("id", make_id()))
            metadata = parse_metadata(item.get("metadata", {}))
            records.append(VectorRecord(record_id, vector, metadata))

    elif name.endswith(".csv"):
        df = pd.read_csv(uploaded)
        if df.empty:
            raise ValueError("The CSV file has no rows.")

        vector_cols = sorted(
            [c for c in df.columns if re.fullmatch(r"x\d+", str(c))],
            key=lambda c: int(str(c)[1:]),
        )

        records = []
        for _, row in df.iterrows():
            if "vector" in df.columns:
                raw = row["vector"]
                if isinstance(raw, str):
                    try:
                        vector = json.loads(raw)
                    except json.JSONDecodeError:
                        vector = [x.strip() for x in raw.split(",")]
                else:
                    vector = raw
                if not isinstance(vector, (list, tuple)):
                    raise ValueError(
                        "CSV 'vector' values must be JSON arrays or comma-separated numbers."
                    )
                vector = [float(x) for x in vector]
            elif vector_cols:
                vector = [float(row[c]) for c in vector_cols]
            else:
                raise ValueError(
                    "CSV needs x0, x1, ... columns or a 'vector' column."
                )

            record_id = int(row["id"]) if "id" in df.columns and pd.notna(row["id"]) else make_id()

            if "metadata" in df.columns:
                metadata = parse_metadata(row["metadata"])
            else:
                excluded = {"id", "vector"}
                metadata_cols = [
                    c for c in df.columns
                    if c not in excluded and c not in vector_cols
                ]
                metadata = {
                    str(c): str(row[c])
                    for c in metadata_cols
                    if pd.notna(row[c])
                }

            records.append(VectorRecord(record_id, vector, metadata))
    else:
        raise ValueError("Upload a .csv or .json file.")

    if not records:
        raise ValueError("No vector records were found.")

    ids = [r.id for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("The upload contains duplicate IDs.")

    for i, record in enumerate(records):
        if len(record.vector) != expected_dim:
            raise ValueError(
                f"Record {record.id} has dimension {len(record.vector)}; "
                f"the server expects {expected_dim}."
            )
        if not all(math.isfinite(float(v)) for v in record.vector):
            raise ValueError(f"Record {record.id} contains a non-finite value.")
        if record.id < 0 or record.id >= 2**64:
            raise ValueError(f"Record {record.id} is outside the uint64 ID range.")

    return records


def metric_card_stats(stats):
    a, b, c = st.columns(3)
    a.metric("Stored vectors", f"{stats.vector_count:,}")
    b.metric("Vector dimensions", stats.dimension)
    c.metric("Distance metric", stats.metric)


def parse_benchmark_output(output):
    patterns = {
        "build_seconds": r"Build time:\s+([\d.]+)\s+s",
        "p50_ms": r"Latency p50:\s+([\d.]+)\s+ms",
        "p95_ms": r"Latency p95:\s+([\d.]+)\s+ms",
        "p99_ms": r"Latency p99:\s+([\d.]+)\s+ms",
        "hnsw_qps": r"HNSW-only QPS:\s+([\d.]+)",
        "recall_at_10": r"Recall@10:\s+([\d.]+)",
    }
    result = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if not match:
            raise ValueError(f"Could not parse '{key}' from benchmark output.")
        result[key] = float(match.group(1))
    return result


def run_benchmark(count, queries, dimension, ef):
    if not BENCHMARK.exists():
        raise FileNotFoundError(
            f"Benchmark executable not found: {BENCHMARK}"
        )

    BENCHMARK_CSV.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(BENCHMARK),
        str(count),
        str(queries),
        str(dimension),
        str(ef),
        str(BENCHMARK_CSV),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr or completed.stdout)

    result = parse_benchmark_output(completed.stdout)
    result.update({
        "vectors": count,
        "queries": queries,
        "dimension": dimension,
        "ef_search": ef,
    })
    return result, completed.stdout


def make_graph(seed, count, neighbors):
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(count, 2))
    # Create a clustered layout that is easier to understand visually.
    centers = np.array([[-1.4, -0.7], [1.3, 0.8], [-0.4, 1.5]])
    labels = rng.integers(0, len(centers), size=count)
    points += centers[labels] * 0.55

    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    adjacency = []
    edges = set()

    for i in range(count):
        nearest = np.argsort(distances[i])[1:neighbors + 1].tolist()
        adjacency.append(nearest)
        for j in nearest:
            edges.add(tuple(sorted((i, j))))

    query = np.array([0.25, -0.05])
    start = int(np.argmax(points[:, 0]))
    visited = [start]
    current = start

    # Greedy walk over this illustrative graph.
    for _ in range(count):
        candidates = adjacency[current]
        if not candidates:
            break
        best = min(candidates, key=lambda j: np.linalg.norm(points[j] - query))
        if np.linalg.norm(points[best] - query) >= np.linalg.norm(points[current] - query):
            break
        current = best
        if current in visited:
            break
        visited.append(current)

    return points, edges, query, visited


tabs = st.tabs(["Playground", "HNSW Visualizer", "Benchmark Lab"])

# ---------------------------------------------------------------------------
# Playground
# ---------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Vector database playground")
    st.write("Upload vectors, insert them into the running server, and query nearest neighbors.")

    try:
        stats = get_stats()
        metric_card_stats(stats)
        expected_dim = stats.dimension
    except Exception as exc:
        expected_dim = None
        st.warning(
            f"Database is not reachable at `{target}`. Start the C++ server to use "
            f"insert/search/stats features. Details: {exc}"
        )

    if expected_dim is not None:
        st.markdown("#### 1. Insert vectors")
        uploaded = st.file_uploader(
            "Choose a CSV or JSON file",
            type=["csv", "json"],
            key="vector_upload",
            help="CSV: use x0, x1, ... columns; optional id and metadata columns. "
                 "JSON: a list of objects with vector and optional id/metadata.",
        )

        if uploaded is not None:
            try:
                records = parse_upload(uploaded, expected_dim)
                st.success(f"Validated {len(records):,} records.")
                preview = pd.DataFrame([
                    {
                        "id": r.id,
                        "vector": r.vector,
                        "metadata": r.metadata,
                    }
                    for r in records[:10]
                ])
                st.dataframe(preview, use_container_width=True, hide_index=True)

                if st.button("Insert uploaded vectors", type="primary"):
                    with st.spinner("Inserting vectors into the C++ server..."):
                        with VectorDBClient(target, timeout=10.0) as client:
                            inserted = client.insert_batch(records)
                            new_stats = client.get_stats()
                    st.session_state["uploaded_metadata"] = {
                        **st.session_state.get("uploaded_metadata", {}),
                        **{r.id: dict(r.metadata) for r in records},
                    }
                    st.success(f"Server reported {inserted:,} inserted records.")
                    metric_card_stats(new_stats)
            except Exception as exc:
                st.error(f"Upload validation/insertion failed: {exc}")

        with st.expander("Supported file formats"):
            st.markdown("**CSV example**")
            st.code(
                "id,x0,x1,x2,category\n"
                "101,0.1,0.2,0.3,fruit\n"
                "102,0.8,0.2,0.1,vehicle",
                language="csv",
            )
            st.markdown("**JSON example**")
            st.code(
                '[\n'
                '  {"id": 101, "vector": [0.1, 0.2, 0.3], '
                '"metadata": {"category": "fruit"}},\n'
                '  {"id": 102, "vector": [0.8, 0.2, 0.1], '
                '"metadata": {"category": "vehicle"}}\n'
                ']',
                language="json",
            )
            st.caption(
                "The number of vector dimensions must match the server configuration. "
                "Use unique IDs; this UI does not overwrite existing records intentionally."
            )

        st.divider()
        st.markdown("#### 2. Similarity search")

        with st.form("search_form"):
            query_text = st.text_input(
                f"Query vector ({expected_dim} comma-separated values)",
                value=",".join(["0.1"] * expected_dim),
            )
            k = st.slider("Number of neighbors (k)", 1, 20, 5)
            filter_text = st.text_input(
                "Metadata filter (JSON object, optional)",
                value="{}",
                help='Example: {"category": "fruit"}',
            )
            submitted = st.form_submit_button("Search", type="primary")

        if submitted:
            try:
                query = [float(x.strip()) for x in query_text.split(",")]
                if len(query) != expected_dim:
                    raise ValueError(f"Expected {expected_dim} query values.")
                if not all(math.isfinite(x) for x in query):
                    raise ValueError("Query values must be finite.")
                metadata_filter = json.loads(filter_text or "{}")
                if not isinstance(metadata_filter, dict):
                    raise ValueError("Metadata filter must be a JSON object.")

                start = time.perf_counter()
                with VectorDBClient(target, timeout=10.0) as client:
                    results = client.search(
                        query,
                        k=k,
                        metadata_filter={
                            str(key): str(value)
                            for key, value in metadata_filter.items()
                        },
                    )
                elapsed_ms = (time.perf_counter() - start) * 1000

                st.metric("Client-observed search time", f"{elapsed_ms:.3f} ms")
                metadata_cache = st.session_state.get("uploaded_metadata", {})
                result_rows = [
                    {
                        "Rank": rank,
                        "Vector ID": result.id,
                        "Distance (lower is closer)": result.distance,
                        "Cached metadata": json.dumps(
                            metadata_cache.get(result.id, {}),
                            ensure_ascii=False,
                        ),
                    }
                    for rank, result in enumerate(results, start=1)
                ]
                if result_rows:
                    st.dataframe(
                        pd.DataFrame(result_rows),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("No results returned. Try another query or filter.")
                st.caption(
                    "Client-observed time includes request overhead. Cached metadata "
                    "is only shown for records uploaded through this UI in this session."
                )
            except Exception as exc:
                st.error(f"Search failed: {exc}")

        st.divider()
        st.markdown("#### 3. Index operations")
        left, right = st.columns(2)
        with left:
            if st.button("Refresh index statistics"):
                try:
                    metric_card_stats(get_stats())
                except Exception as exc:
                    st.error(str(exc))
        with right:
            if st.button("Create persistence snapshot"):
                try:
                    with VectorDBClient(target, timeout=10.0) as client:
                        client.create_snapshot()
                    st.success("Snapshot request completed.")
                except Exception as exc:
                    st.error(f"Snapshot failed: {exc}")


# ---------------------------------------------------------------------------
# HNSW visualizer
# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("How graph-based nearest-neighbor search works")
    st.warning(
        "Teaching visualization only: this is a generated 2D neighbor graph, "
        "not the internal graph extracted from your live hnswlib index. "
        "The current SDK does not expose HNSW links or visited nodes."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        graph_nodes = st.slider("Illustrative nodes", 20, 100, 45, 5)
    with c2:
        graph_neighbors = st.slider("Edges per node", 2, 8, 4)
    with c3:
        graph_seed = st.number_input("Layout seed", min_value=0, max_value=9999, value=42)

    points, edges, query_point, path = make_graph(
        int(graph_seed), graph_nodes, graph_neighbors
    )

    edge_x, edge_y = [], []
    for a, b in edges:
        edge_x.extend([points[a, 0], points[b, 0], None])
        edge_y.extend([points[a, 1], points[b, 1], None])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=0.7, color="rgba(130,140,160,0.45)"),
        hoverinfo="skip",
        name="Graph links",
    ))

    path_edges_x, path_edges_y = [], []
    for a, b in zip(path[:-1], path[1:]):
        path_edges_x.extend([points[a, 0], points[b, 0], None])
        path_edges_y.extend([points[a, 1], points[b, 1], None])

    fig.add_trace(go.Scatter(
        x=path_edges_x, y=path_edges_y,
        mode="lines",
        line=dict(width=4, color="#F59E0B"),
        hoverinfo="skip",
        name="Illustrative greedy path",
    ))
    fig.add_trace(go.Scatter(
        x=points[:, 0], y=points[:, 1],
        mode="markers+text",
        text=[str(i) for i in range(graph_nodes)],
        textposition="top center",
        textfont=dict(size=9),
        marker=dict(
            size=10,
            color=["#F59E0B" if i in path else "#4C8BF5" for i in range(graph_nodes)],
            line=dict(width=1, color="white"),
        ),
        name="Nodes",
        hovertemplate="Node %{text}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[query_point[0]], y=[query_point[1]],
        mode="markers",
        marker=dict(size=18, symbol="star", color="#EF4444"),
        name="Query",
        hovertemplate="Query vector<extra></extra>",
    ))
    fig.update_layout(
        height=520,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="Projection dimension 1",
        yaxis_title="Projection dimension 2",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Illustrative nodes visited", len(path))
    m2.metric("Illustrative graph links", len(edges))
    m3.metric("Path start → end", f"{path[0]} → {path[-1]}")
    st.markdown(
        "**What to notice:** the example follows local links toward a query instead "
        "of comparing every node. Real HNSW uses a hierarchy of graph layers, "
        "construction parameters, and a configurable search breadth (`ef_search`)."
    )


# ---------------------------------------------------------------------------
# Benchmark lab
# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Benchmark Lab")
    st.write(
        "Run your existing C++ benchmark. It builds a synthetic cosine-vector dataset, "
        "measures HNSW search, and compares its top-10 neighbors against exact search."
    )

    if not BENCHMARK.exists():
        st.error(f"Benchmark executable missing: {BENCHMARK}")
    else:
        st.success("C++ benchmark executable found.")

    with st.form("benchmark_form"):
        bc1, bc2 = st.columns(2)
        with bc1:
            bench_vectors = st.select_slider(
                "Vector count",
                options=[500, 1000, 2000, 5000, 10000],
                value=2000,
            )
            bench_queries = st.select_slider(
                "Query count",
                options=[25, 50, 100, 200, 500],
                value=100,
            )
        with bc2:
            bench_dimension = st.select_slider(
                "Dimensions",
                options=[16, 32, 64, 128],
                value=64,
            )
            bench_efs = st.multiselect(
                "ef_search values to compare",
                options=[10, 25, 50, 100, 200],
                default=[25, 50, 100],
                help="Larger values generally trade speed for better recall.",
            )
        run_clicked = st.form_submit_button("Run benchmark comparison", type="primary")

    if run_clicked:
        if not bench_efs:
            st.error("Choose at least one ef_search value.")
        elif min(bench_efs) < 10:
            st.error("ef_search must be at least 10 because the benchmark uses k=10.")
        else:
            rows = []
            progress = st.progress(0)
            status = st.empty()
            try:
                for i, ef in enumerate(bench_efs):
                    status.write(f"Running C++ benchmark: ef_search={ef} …")
                    result, raw_output = run_benchmark(
                        bench_vectors,
                        bench_queries,
                        bench_dimension,
                        ef,
                    )
                    rows.append(result)
                    progress.progress((i + 1) / len(bench_efs))
                st.session_state["benchmark_results"] = rows
                status.success("All benchmark runs completed.")
            except Exception as exc:
                status.error(f"Benchmark failed: {exc}")

    rows = st.session_state.get("benchmark_results", [])
    if rows:
        results_df = pd.DataFrame(rows).sort_values("ef_search")
        display_df = results_df.rename(columns={
            "ef_search": "ef_search",
            "build_seconds": "Build (s)",
            "p50_ms": "p50 (ms)",
            "p95_ms": "p95 (ms)",
            "p99_ms": "p99 (ms)",
            "hnsw_qps": "HNSW QPS",
            "recall_at_10": "Recall@10",
        })
        st.markdown("#### Results")
        st.dataframe(
            display_df[
                [
                    "ef_search", "Build (s)", "p50 (ms)", "p95 (ms)",
                    "p99 (ms)", "HNSW QPS", "Recall@10",
                ]
            ].style.format({
                "Build (s)": "{:.3f}",
                "p50 (ms)": "{:.3f}",
                "p95 (ms)": "{:.3f}",
                "p99 (ms)": "{:.3f}",
                "HNSW QPS": "{:,.0f}",
                "Recall@10": "{:.1%}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        chart_left, chart_right = st.columns(2)
        with chart_left:
            latency_fig = go.Figure()
            for column, label in [
                ("p50_ms", "p50"),
                ("p95_ms", "p95"),
                ("p99_ms", "p99"),
            ]:
                latency_fig.add_trace(go.Scatter(
                    x=results_df["ef_search"],
                    y=results_df[column],
                    mode="lines+markers",
                    name=label,
                ))
            latency_fig.update_layout(
                title="Query latency vs. ef_search",
                xaxis_title="ef_search",
                yaxis_title="Latency (ms)",
                template="plotly_white",
            )
            st.plotly_chart(latency_fig, use_container_width=True)

        with chart_right:
            recall_fig = go.Figure()
            recall_fig.add_trace(go.Scatter(
                x=results_df["ef_search"],
                y=results_df["recall_at_10"] * 100,
                mode="lines+markers",
                name="Recall@10",
            ))
            recall_fig.update_layout(
                title="Recall vs. ef_search",
                xaxis_title="ef_search",
                yaxis_title="Recall@10 (%)",
                yaxis=dict(range=[0, 100]),
                template="plotly_white",
            )
            st.plotly_chart(recall_fig, use_container_width=True)

        st.caption(
            "Results are machine- and workload-dependent. This benchmark uses synthetic "
            "normalized vectors and reports recall against exact top-10 search. "
            "QPS and latency exclude the exact-search recall calculation."
        )
        st.download_button(
            "Download displayed results as CSV",
            data=results_df.to_csv(index=False).encode("utf-8"),
            file_name="vectordb_benchmark_results.csv",
            mime="text/csv",
        )
        st.caption(f"Raw benchmark runs are also appended to `{BENCHMARK_CSV.relative_to(ROOT)}`.")
    else:
        st.info("Choose benchmark settings above and run a comparison to populate the charts.")

st.divider()
st.caption(
    "VectorDB Explorer · Interview showcase · Local server only · "
    "The visualizer is illustrative; benchmark metrics come from the C++ executable."
)
