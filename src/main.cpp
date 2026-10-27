#include "vectordb/persistent_store.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

using vectordb::IndexConfig;
using vectordb::Metadata;
using vectordb::PersistentStore;
using vectordb::VectorRecord;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

int main() {
    const std::filesystem::path storage = "data/metadata_demo";

    try {
        // This is disposable test data.
        std::filesystem::remove_all(storage);

        IndexConfig config;
        config.dimension = 3;
        config.max_elements = 100;
        config.M = 16;
        config.ef_construction = 100;
        config.ef_search = 50;
        config.metric = vectordb::Metric::COSINE;

        const Metadata fruit{{"category", "fruit"}, {"color", "red"}};
        const Metadata vegetable{
            {"category", "vegetable"}, {"color", "green"}
        };

        // Insert individual records and a batch.
        {
            PersistentStore db(config, storage);

            db.insert(101, {1.0f, 0.0f, 0.0f}, fruit);

            db.insert_batch({
                VectorRecord{
                    102, {0.9f, 0.1f, 0.0f},
                    {{"category", "fruit"}, {"color", "green"}}
                },
                VectorRecord{
                    103, {0.0f, 1.0f, 0.0f}, vegetable
                },
                VectorRecord{
                    104, {0.0f, 0.0f, 1.0f},
                    {{"category", "drink"}, {"color", "blue"}}
                }
            });

            require(db.size() == 4, "Insert/batch ingestion failed");

            auto filtered = db.search(
                {1.0f, 0.0f, 0.0f},
                3,
                {{"category", "fruit"}}
            );

            require(filtered.size() == 2, "Metadata filtering failed");
            require(
                filtered[0].id == 101,
                "Unexpected first filtered result"
            );

            std::cout << "Insert + batch ingestion: PASS\n";
            std::cout << "Metadata filtering: PASS\n";
        }

        // Reopen and verify that WAL recovery restores metadata too.
        {
            PersistentStore db(config, storage);

            require(db.size() == 4, "WAL recovery count mismatch");

            auto filtered = db.search(
                {1.0f, 0.0f, 0.0f},
                3,
                {{"category", "fruit"}}
            );

            require(
                filtered.size() == 2,
                "Metadata was not recovered from WAL"
            );

            std::cout << "WAL recovery: PASS\n";
            std::cout << "Metadata recovery from WAL: PASS\n";

            db.snapshot();
            std::cout << "Snapshot creation: PASS\n";
        }

        // Reopen again to verify snapshot recovery.
        {
            PersistentStore db(config, storage);

            require(db.size() == 4, "Snapshot recovery count mismatch");

            auto filtered = db.search(
                {1.0f, 0.0f, 0.0f},
                3,
                {{"category", "fruit"}}
            );

            require(
                filtered.size() == 2,
                "Metadata was not recovered from snapshot"
            );

            auto results = db.search({1.0f, 0.0f, 0.0f}, 3);

            require(
                !results.empty() && results.front().id == 101,
                "Nearest-neighbor search failed after recovery"
            );

            std::cout << "Snapshot recovery: PASS\n";
            std::cout << "Metadata recovery from snapshot: PASS\n";

            std::cout << "\nNearest neighbors:\n";
            for (const auto& result : results) {
                std::cout << "ID: " << result.id
                          << " | Distance: " << result.distance
                          << '\n';
            }
        }

        std::cout << "\nALL TESTS PASSED\n";
        return 0;

    } catch (const std::exception& error) {
        std::cerr << "TEST FAILURE: " << error.what() << '\n';
        return 1;
    }
}
