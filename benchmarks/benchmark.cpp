#include "vectordb/vector_index.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <vector>

using Clock = std::chrono::steady_clock;

namespace {

using Vector = std::vector<float>;

Vector normalize(Vector v) {
    double norm_squared = 0.0;
    for (float value : v) {
        norm_squared += static_cast<double>(value) * value;
    }

    const double norm = std::sqrt(norm_squared);
    if (norm > 0.0) {
        for (float& value : v) {
            value = static_cast<float>(value / norm);
        }
    }
    return v;
}

float cosine_distance(const Vector& a, const Vector& b) {
    double dot = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * b[i];
    }
    return static_cast<float>(1.0 - dot);
}

std::vector<std::size_t> exact_search(
    const std::vector<Vector>& vectors,
    const Vector& query,
    std::size_t k
) {
    std::vector<std::pair<float, std::size_t>> distances;
    distances.reserve(vectors.size());

    for (std::size_t i = 0; i < vectors.size(); ++i) {
        distances.emplace_back(cosine_distance(query, vectors[i]), i);
    }

    const auto count = std::min(k, distances.size());
    std::partial_sort(
        distances.begin(),
        distances.begin() + count,
        distances.end()
    );

    std::vector<std::size_t> ids;
    ids.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        ids.push_back(distances[i].second);
    }
    return ids;
}

double percentile(std::vector<double> values, double p) {
    if (values.empty()) {
        return 0.0;
    }

    std::sort(values.begin(), values.end());
    const auto index = static_cast<std::size_t>(
        std::ceil(p * values.size())
    ) - 1;
    return values[std::min(index, values.size() - 1)];
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::size_t count =
            argc > 1 ? std::stoull(argv[1]) : 10000;
        const std::size_t query_count =
            argc > 2 ? std::stoull(argv[2]) : 200;
        const std::size_t dimension =
            argc > 3 ? std::stoull(argv[3]) : 64;
        const std::size_t ef_search =
            argc > 4 ? std::stoull(argv[4]) : 50;
        const std::size_t k = 10;

        if (count < k || query_count == 0 || dimension == 0 || ef_search < k) {
            throw std::invalid_argument(
                "Use at least 10 vectors, one query, one dimension, and ef_search >= k."
            );
        }

        constexpr unsigned seed = 42;
        std::mt19937 generator(seed);
        std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

        std::vector<Vector> vectors;
        vectors.reserve(count);

        for (std::size_t i = 0; i < count; ++i) {
            Vector v(dimension);
            for (float& value : v) {
                value = distribution(generator);
            }
            vectors.push_back(normalize(std::move(v)));
        }

        std::vector<Vector> queries;
        queries.reserve(query_count);

        for (std::size_t i = 0; i < query_count; ++i) {
            Vector q(dimension);
            for (float& value : q) {
                value = distribution(generator);
            }
            queries.push_back(normalize(std::move(q)));
        }

        vectordb::IndexConfig config;
        config.dimension = dimension;
        config.max_elements = count;
        config.M = 16;
        config.ef_construction = 100;
        config.ef_search = ef_search;
        config.metric = vectordb::Metric::COSINE;

        vectordb::VectorIndex index(config);

        const auto build_start = Clock::now();
        for (std::size_t i = 0; i < count; ++i) {
            index.insert(i, vectors[i]);
        }
        const auto build_end = Clock::now();

        std::vector<double> latencies_ms;
        latencies_ms.reserve(query_count);

        double total_recall = 0.0;

        for (const auto& query : queries) {
            const auto start = Clock::now();
            const auto approximate = index.search(query, k);
            const auto end = Clock::now();

            const double elapsed_ms =
                std::chrono::duration<double, std::milli>(
                    end - start
                ).count();
            latencies_ms.push_back(elapsed_ms);

            const auto exact = exact_search(vectors, query, k);
            std::unordered_set<std::size_t> exact_ids(
                exact.begin(), exact.end()
            );

            std::size_t hits = 0;
            for (const auto& result : approximate) {
                if (exact_ids.count(result.id) != 0) {
                    ++hits;
                }
            }

            total_recall += static_cast<double>(hits) / exact.size();
        }

        const double build_seconds =
            std::chrono::duration<double>(
                build_end - build_start
            ).count();
        double hnsw_seconds = 0.0;
        for (double latency : latencies_ms) {
            hnsw_seconds += latency / 1000.0;
        }

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "\nVectorDB HNSW Benchmark\n";
        std::cout << "-----------------------\n";
        std::cout << "Vectors:       " << count << '\n';
        std::cout << "Queries:       " << query_count << '\n';
        std::cout << "Dimensions:    " << dimension << '\n';
        std::cout << "k:             " << k << '\n';
        std::cout << "Seed:          " << seed << '\n';
        std::cout << "ef_search:     " << config.ef_search << '\n';
        std::cout << "Build time:    " << build_seconds << " s\n";
        std::cout << "Latency p50:   "
                  << percentile(latencies_ms, 0.50) << " ms\n";
        std::cout << "Latency p95:   "
                  << percentile(latencies_ms, 0.95) << " ms\n";
        std::cout << "Latency p99:   "
                  << percentile(latencies_ms, 0.99) << " ms\n";
        const double hnsw_qps = query_count / hnsw_seconds;
        std::cout << "HNSW-only QPS: "
                  << hnsw_qps << '\n';
        std::cout << "Recall@10:     "
                  << total_recall / query_count << '\n';
        std::cout << "\nQPS and latency exclude exact-search recall work.\n";

        if (argc > 5) {
            const bool write_header = !std::ifstream(argv[5]).good();
            std::ofstream csv(argv[5], std::ios::app);
            if (!csv) {
                throw std::runtime_error("Could not open CSV output file");
            }
            if (write_header) {
                csv << "vectors,queries,dimension,k,ef_search,seed,"
                       "build_seconds,p50_ms,p95_ms,p99_ms,hnsw_qps,recall_at_10\n";
            }
            csv << count << ',' << query_count << ',' << dimension << ','
                << k << ',' << config.ef_search << ',' << seed << ','
                << build_seconds << ','
                << percentile(latencies_ms, 0.50) << ','
                << percentile(latencies_ms, 0.95) << ','
                << percentile(latencies_ms, 0.99) << ','
                << hnsw_qps << ','
                << total_recall / query_count << '\n';
            std::cout << "CSV appended to: " << argv[5] << '\n';
        }

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
