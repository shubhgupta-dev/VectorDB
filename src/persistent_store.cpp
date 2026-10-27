#include "vectordb/persistent_store.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace vectordb {
namespace {

constexpr std::array<char, 8> SNAPSHOT_MAGIC{
    'V', 'D', 'B', 'S', 'N', 'A', 'P', '2'
};

constexpr std::uint64_t MAX_RECORD_BYTES = 64ULL * 1024 * 1024;
constexpr std::uint64_t MAX_METADATA_ITEMS = 100000;
constexpr std::uint64_t MAX_STRING_BYTES = 1024 * 1024;

template <typename T>
void write_value(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));

    if (!out) {
        throw std::runtime_error("Failed to write persistent data");
    }
}

template <typename T>
bool read_value(std::istream& in, T& value) {
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    return static_cast<bool>(in);
}

void write_string(std::ostream& out, const std::string& value) {
    const auto length = static_cast<std::uint64_t>(value.size());

    write_value(out, length);
    out.write(value.data(), static_cast<std::streamsize>(value.size()));

    if (!out) {
        throw std::runtime_error("Failed to write metadata string");
    }
}

bool read_string(std::istream& in, std::string& value) {
    std::uint64_t length = 0;

    if (!read_value(in, length)) {
        return false;
    }

    if (length > MAX_STRING_BYTES) {
        throw std::runtime_error("Metadata string exceeds size limit");
    }

    value.resize(static_cast<std::size_t>(length));

    in.read(value.data(), static_cast<std::streamsize>(length));
    return static_cast<bool>(in);
}

void write_record_payload(
    std::ostream& out,
    std::uint64_t sequence,
    std::uint64_t id,
    const std::vector<float>& vector,
    const Metadata& metadata
) {
    write_value(out, sequence);
    write_value(out, id);

    out.write(
        reinterpret_cast<const char*>(vector.data()),
        static_cast<std::streamsize>(vector.size() * sizeof(float))
    );

    if (!out) {
        throw std::runtime_error("Failed to write vector");
    }

    const auto metadata_count =
        static_cast<std::uint64_t>(metadata.size());

    write_value(out, metadata_count);

    for (const auto& [key, value] : metadata) {
        write_string(out, key);
        write_string(out, value);
    }
}

bool read_record_payload(
    std::istream& in,
    std::size_t dimension,
    std::uint64_t& sequence,
    std::uint64_t& id,
    std::vector<float>& vector,
    Metadata& metadata
) {
    if (!read_value(in, sequence) || !read_value(in, id)) {
        return false;
    }

    vector.resize(dimension);

    in.read(
        reinterpret_cast<char*>(vector.data()),
        static_cast<std::streamsize>(dimension * sizeof(float))
    );

    if (!in) {
        return false;
    }

    std::uint64_t metadata_count = 0;

    if (!read_value(in, metadata_count)) {
        return false;
    }

    if (metadata_count > MAX_METADATA_ITEMS) {
        throw std::runtime_error("Metadata item count exceeds limit");
    }

    metadata.clear();

    for (std::uint64_t i = 0; i < metadata_count; ++i) {
        std::string key;
        std::string value;

        if (!read_string(in, key) || !read_string(in, value)) {
            return false;
        }

        if (!metadata.emplace(std::move(key), std::move(value)).second) {
            throw std::runtime_error("Duplicate metadata key");
        }
    }

    return true;
}

}  // namespace

PersistentStore::PersistentStore(
    IndexConfig config,
    const std::filesystem::path& directory
)
    : config_(config),
      directory_(directory),
      wal_path_(directory / "wal.bin"),
      snapshot_path_(directory / "snapshot.bin"),
      index_(config) {
    std::filesystem::create_directories(directory_);
    recover();
}

void PersistentStore::recover() {
    load_snapshot();
    replay_wal();
}

void PersistentStore::load_snapshot() {
    if (!std::filesystem::exists(snapshot_path_)) {
        return;
    }

    std::ifstream in(snapshot_path_, std::ios::binary);

    if (!in) {
        throw std::runtime_error("Cannot open snapshot");
    }

    std::array<char, 8> magic{};
    in.read(magic.data(), static_cast<std::streamsize>(magic.size()));

    if (!in || magic != SNAPSHOT_MAGIC) {
        throw std::runtime_error(
            "Invalid snapshot header or unsupported snapshot version"
        );
    }

    std::uint64_t dimension = 0;
    std::uint64_t count = 0;
    std::uint64_t snapshot_sequence = 0;

    if (!read_value(in, dimension) ||
        !read_value(in, count) ||
        !read_value(in, snapshot_sequence)) {
        throw std::runtime_error("Truncated snapshot header");
    }

    if (dimension != config_.dimension ||
        count > config_.max_elements) {
        throw std::runtime_error("Snapshot configuration mismatch");
    }

    for (std::uint64_t i = 0; i < count; ++i) {
        std::uint64_t id = 0;
        std::vector<float> vector;
        Metadata metadata;
        std::uint64_t ignored_sequence = 0;

        if (!read_record_payload(
                in, config_.dimension, ignored_sequence,
                id, vector, metadata)) {
            throw std::runtime_error("Truncated snapshot record");
        }

        for (float value : vector) {
            if (!std::isfinite(value)) {
                throw std::runtime_error(
                    "Snapshot contains a non-finite vector value"
                );
            }
        }

        if (!records_.emplace(id, vector).second) {
            throw std::runtime_error("Duplicate ID in snapshot");
        }

        metadata_[id] = std::move(metadata);
        index_.insert(id, vector);
    }

    last_sequence_ = snapshot_sequence;
}

void PersistentStore::replay_wal() {
    if (!std::filesystem::exists(wal_path_)) {
        return;
    }

    std::ifstream in(wal_path_, std::ios::binary);

    if (!in) {
        throw std::runtime_error("Cannot open WAL");
    }

    std::uint64_t valid_bytes = 0;

    while (true) {
        std::uint64_t payload_size = 0;

        if (!read_value(in, payload_size)) {
            break;
        }

        if (payload_size > MAX_RECORD_BYTES) {
            throw std::runtime_error("WAL record exceeds size limit");
        }

        std::string payload(static_cast<std::size_t>(payload_size), '\0');

        in.read(
            payload.data(),
            static_cast<std::streamsize>(payload_size)
        );

        if (!in) {
            break;
        }

        valid_bytes += sizeof(payload_size) + payload_size;

        std::istringstream record_stream(
            payload, std::ios::binary | std::ios::in
        );

        std::uint64_t sequence = 0;
        std::uint64_t id = 0;
        std::vector<float> vector;
        Metadata metadata;

        if (!read_record_payload(
                record_stream, config_.dimension, sequence,
                id, vector, metadata)) {
            throw std::runtime_error("Malformed WAL record");
        }

        if (record_stream.peek() != std::char_traits<char>::eof()) {
            throw std::runtime_error("WAL record has trailing data");
        }

        for (float value : vector) {
            if (!std::isfinite(value)) {
                throw std::runtime_error(
                    "WAL contains a non-finite vector value"
                );
            }
        }

        // Records already represented by the snapshot need not be replayed.
        if (sequence <= last_sequence_) {
            continue;
        }

        if (sequence != last_sequence_ + 1) {
            throw std::runtime_error("WAL sequence gap");
        }

        if (records_.size() >= config_.max_elements) {
            throw std::runtime_error("Recovered index exceeds capacity");
        }

        if (records_.find(id) != records_.end()) {
            throw std::runtime_error("Duplicate ID in WAL");
        }

        index_.insert(id, vector);
        records_.emplace(id, vector);
        metadata_[id] = std::move(metadata);
        last_sequence_ = sequence;
    }

    in.close();

    const auto actual_bytes = std::filesystem::file_size(wal_path_);

    if (valid_bytes < actual_bytes) {
        std::filesystem::resize_file(wal_path_, valid_bytes);
    }
}

void PersistentStore::append_wal(
    const Record& record,
    std::uint64_t sequence
) {
    std::ostringstream payload_stream(
        std::ios::binary | std::ios::out
    );

    const auto metadata_it = metadata_.find(record.id);
    const Metadata empty_metadata;

    const Metadata& metadata =
        metadata_it == metadata_.end()
            ? empty_metadata
            : metadata_it->second;

    write_record_payload(
        payload_stream,
        sequence,
        record.id,
        record.vector,
        metadata
    );

    const std::string payload = payload_stream.str();

    if (payload.size() > MAX_RECORD_BYTES) {
        throw std::runtime_error("WAL record exceeds size limit");
    }

    std::ofstream out(wal_path_, std::ios::binary | std::ios::app);

    if (!out) {
        throw std::runtime_error("Cannot open WAL for writing");
    }

    const auto payload_size =
        static_cast<std::uint64_t>(payload.size());

    write_value(out, payload_size);

    out.write(
        payload.data(),
        static_cast<std::streamsize>(payload.size())
    );

    out.flush();

    if (!out) {
        throw std::runtime_error("Failed to append WAL record");
    }
}

void PersistentStore::insert(
    std::uint64_t id,
    const std::vector<float>& vector,
    const Metadata& metadata
) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (records_.find(id) != records_.end()) {
        throw std::invalid_argument("Vector ID already exists");
    }

    if (records_.size() >= config_.max_elements) {
        throw std::runtime_error("Index capacity reached");
    }

    if (vector.size() != config_.dimension) {
        throw std::invalid_argument("Vector dimension mismatch");
    }

    for (float value : vector) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "Vector contains a non-finite value"
            );
        }
    }

    if (last_sequence_ == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("WAL sequence exhausted");
    }

    Record record{id, vector};

    const auto next_sequence = last_sequence_ + 1;

    // Include metadata in the durable log record.
    std::ostringstream payload_stream(
        std::ios::binary | std::ios::out
    );

    write_record_payload(
        payload_stream, next_sequence, id, vector, metadata
    );

    const std::string payload = payload_stream.str();

    if (payload.size() > MAX_RECORD_BYTES) {
        throw std::runtime_error("WAL record exceeds size limit");
    }

    std::ofstream out(wal_path_, std::ios::binary | std::ios::app);

    if (!out) {
        throw std::runtime_error("Cannot open WAL for writing");
    }

    const auto payload_size =
        static_cast<std::uint64_t>(payload.size());

    write_value(out, payload_size);

    out.write(
        payload.data(),
        static_cast<std::streamsize>(payload.size())
    );

    out.flush();

    if (!out) {
        throw std::runtime_error("Failed to append WAL record");
    }

    out.close();

    index_.insert(id, vector);
    records_.emplace(id, vector);
    metadata_[id] = metadata;
    last_sequence_ = next_sequence;
}

void PersistentStore::insert_batch(
    const std::vector<VectorRecord>& records
) {
    for (const auto& record : records) {
        insert(record.id, record.vector, record.metadata);
    }
}

std::vector<SearchResult> PersistentStore::search(
    const std::vector<float>& query,
    std::size_t k,
    const Metadata& filter
) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (k == 0 || records_.empty()) {
        return {};
    }

    // Retrieve candidates from HNSW and apply metadata predicates.
    // This version searches all indexed candidates for correct filtering
    // on this small prototype; large datasets need a more efficient plan.
    auto candidates = index_.search(query, records_.size());

    std::vector<SearchResult> results;
    results.reserve(std::min(k, candidates.size()));

    for (const auto& candidate : candidates) {
        const auto metadata_it = metadata_.find(candidate.id);

        const Metadata empty_metadata;
        const Metadata& metadata =
            metadata_it == metadata_.end()
                ? empty_metadata
                : metadata_it->second;

        bool matches = true;

        for (const auto& [key, value] : filter) {
            const auto it = metadata.find(key);

            if (it == metadata.end() || it->second != value) {
                matches = false;
                break;
            }
        }

        if (matches) {
            results.push_back(candidate);

            if (results.size() == k) {
                break;
            }
        }
    }

    return results;
}

void PersistentStore::write_snapshot() {
    const auto temporary_path = snapshot_path_.string() + ".tmp";

    std::ofstream out(
        temporary_path,
        std::ios::binary | std::ios::trunc
    );

    if (!out) {
        throw std::runtime_error("Cannot create snapshot");
    }

    out.write(
        SNAPSHOT_MAGIC.data(),
        static_cast<std::streamsize>(SNAPSHOT_MAGIC.size())
    );

    const auto dimension =
        static_cast<std::uint64_t>(config_.dimension);

    const auto count =
        static_cast<std::uint64_t>(records_.size());

    write_value(out, dimension);
    write_value(out, count);
    write_value(out, last_sequence_);

    for (const auto& [id, vector] : records_) {
        const auto metadata_it = metadata_.find(id);
        const Metadata empty_metadata;

        const Metadata& metadata =
            metadata_it == metadata_.end()
                ? empty_metadata
                : metadata_it->second;

        // Snapshot records use the same payload representation as WAL.
        write_record_payload(
            out, last_sequence_, id, vector, metadata
        );
    }

    out.flush();

    if (!out) {
        throw std::runtime_error("Failed to flush snapshot");
    }

    out.close();

    std::filesystem::rename(temporary_path, snapshot_path_);
}

void PersistentStore::snapshot() {
    std::lock_guard<std::mutex> lock(mutex_);
    write_snapshot();
}

std::size_t PersistentStore::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return records_.size();
}

}  // namespace vectordb


const vectordb::IndexConfig& vectordb::PersistentStore::config() const {
    return config_;
}
