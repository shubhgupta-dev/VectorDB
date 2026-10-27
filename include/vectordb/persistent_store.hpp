#pragma once

#include "vectordb/vector_index.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace vectordb {

using Metadata = std::unordered_map<std::string, std::string>;

struct VectorRecord {
    std::uint64_t id;
    std::vector<float> vector;
    Metadata metadata;
};

class PersistentStore {
public:
    PersistentStore(
        IndexConfig config,
        const std::filesystem::path& directory
    );

    void insert(
        std::uint64_t id,
        const std::vector<float>& vector,
        const Metadata& metadata = {}
    );

    void insert_batch(const std::vector<VectorRecord>& records);

    std::vector<SearchResult> search(
        const std::vector<float>& query,
        std::size_t k,
        const Metadata& filter = {}
    );

    void snapshot();

    std::size_t size() const;
    const IndexConfig& config() const;

private:
    struct Record {
        std::uint64_t id;
        std::vector<float> vector;
    };

    void recover();
    void load_snapshot();
    void replay_wal();
    void append_wal(const Record& record, std::uint64_t sequence);
    void write_snapshot();

    IndexConfig config_;
    std::filesystem::path directory_;
    std::filesystem::path wal_path_;
    std::filesystem::path snapshot_path_;

    VectorIndex index_;
    std::unordered_map<std::uint64_t, std::vector<float>> records_;
    std::unordered_map<std::uint64_t, Metadata> metadata_;

    std::uint64_t last_sequence_ = 0;
    mutable std::mutex mutex_;
};

}  // namespace vectordb
