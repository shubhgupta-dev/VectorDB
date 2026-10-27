
#pragma once

#include <cstddef>
#include <memory>
#include <mutex>
#include <vector>

#include "hnswlib.h"

namespace vectordb {

enum class Metric {
    L2,
    COSINE,
    INNER_PRODUCT
};

struct SearchResult {
    std::size_t id;
    float distance;
};

struct IndexConfig {
    std::size_t dimension = 128;
    std::size_t max_elements = 10000;
    std::size_t M = 16;
    std::size_t ef_construction = 100;
    std::size_t ef_search = 50;
    Metric metric = Metric::COSINE;
};

class VectorIndex {
public:
    explicit VectorIndex(IndexConfig config);
    ~VectorIndex();

    VectorIndex(const VectorIndex&) = delete;
    VectorIndex& operator=(const VectorIndex&) = delete;

    void insert(
        std::size_t id,
        const std::vector<float>& vector
    );

    std::vector<SearchResult> search(
        const std::vector<float>& query,
        std::size_t k
    );

    std::size_t size() const;

    const IndexConfig& config() const;

private:
    std::vector<float> prepare_vector(
        const std::vector<float>& vector
    ) const;

    IndexConfig config_;

    std::unique_ptr<hnswlib::SpaceInterface<float>> space_;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> index_;

    mutable std::mutex mutex_;
};

}  // namespace vectordb
