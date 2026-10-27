#include "vectordb/vector_index.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_set>

namespace vectordb {

VectorIndex::VectorIndex(IndexConfig config)
    : config_(config) {
    if (config_.dimension == 0 ||
        config_.max_elements == 0 ||
        config_.M < 2 ||
        config_.ef_construction == 0 ||
        config_.ef_search == 0) {
        throw std::invalid_argument("Invalid HNSW configuration");
    }

    switch (config_.metric) {
        case Metric::L2:
            space_ = std::make_unique<hnswlib::L2Space>(
                config_.dimension
            );
            break;

        case Metric::COSINE:
        case Metric::INNER_PRODUCT:
            space_ = std::make_unique<hnswlib::InnerProductSpace>(
                config_.dimension
            );
            break;
    }

    index_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(
        space_.get(),
        config_.max_elements,
        config_.M,
        config_.ef_construction
    );

    index_->setEf(config_.ef_search);
}

VectorIndex::~VectorIndex() = default;

std::vector<float> VectorIndex::prepare_vector(
    const std::vector<float>& vector
) const {
    if (vector.size() != config_.dimension) {
        throw std::invalid_argument("Vector dimension mismatch");
    }

    std::vector<float> result = vector;
    double norm_squared = 0.0;

    for (float value : result) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "Vector contains a non-finite value"
            );
        }
        norm_squared += static_cast<double>(value) * value;
    }

    if (config_.metric == Metric::COSINE) {
        if (norm_squared == 0.0) {
            throw std::invalid_argument(
                "Cosine similarity is undefined for a zero vector"
            );
        }

        const float inverse_norm =
            static_cast<float>(1.0 / std::sqrt(norm_squared));

        for (float& value : result) {
            value *= inverse_norm;
        }
    }

    return result;
}

void VectorIndex::insert(
    std::size_t id,
    const std::vector<float>& vector
) {
    auto prepared = prepare_vector(vector);

    std::lock_guard<std::mutex> lock(mutex_);

    index_->addPoint(
        prepared.data(),
        static_cast<hnswlib::labeltype>(id)
    );
}

std::vector<SearchResult> VectorIndex::search(
    const std::vector<float>& query,
    std::size_t k
) {
    auto prepared = prepare_vector(query);

    if (k == 0) {
        return {};
    }

    std::lock_guard<std::mutex> lock(mutex_);

    const std::size_t count = index_->getCurrentElementCount();
    if (count == 0) {
        return {};
    }

    auto neighbors = index_->searchKnn(
        prepared.data(),
        std::min(k, count)
    );

    std::vector<SearchResult> results;
    results.reserve(neighbors.size());

    while (!neighbors.empty()) {
        const auto item = neighbors.top();
        neighbors.pop();

        results.push_back({
            static_cast<std::size_t>(item.second),
            item.first
        });
    }

    // HNSW returns a max-heap; reverse it for nearest-first output.
    std::reverse(results.begin(), results.end());
    return results;
}

std::size_t VectorIndex::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return index_->getCurrentElementCount();
}

const IndexConfig& VectorIndex::config() const {
    return config_;
}

}  // namespace vectordb
