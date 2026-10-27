#include "vectordb/persistent_store.hpp"
#include "vectordb.grpc.pb.h"

#include <grpcpp/grpcpp.h>

#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using grpc::ServerContext;
using grpc::Status;
using grpc::StatusCode;

using vectordb::Metadata;
using vectordb::PersistentStore;
using vectordb::VectorRecord;

namespace api = ::vectordb::v1;

class VectorDBService final : public api::VectorDB::Service {
public:
    explicit VectorDBService(PersistentStore& store) : store_(store) {}

    Status Insert(
        ServerContext*,
        const api::InsertRequest* request,
        api::InsertResponse* response
    ) override {
        if (!request->has_record()) {
            return Status(StatusCode::INVALID_ARGUMENT, "Missing vector record");
        }

        try {
            const auto& record = request->record();
            store_.insert(
                record.id(),
                to_vector(record.values()),
                to_metadata(record.metadata())
            );

            response->set_success(true);
            response->set_message("Vector inserted");
            return Status::OK;
        } catch (const std::invalid_argument& error) {
            return Status(StatusCode::INVALID_ARGUMENT, error.what());
        } catch (const std::exception& error) {
            return Status(StatusCode::INTERNAL, error.what());
        }
    }

    Status InsertBatch(
        ServerContext*,
        const api::BatchInsertRequest* request,
        api::BatchInsertResponse* response
    ) override {
        try {
            std::vector<VectorRecord> records;
            records.reserve(request->records_size());

            for (const auto& record : request->records()) {
                records.push_back({
                    record.id(),
                    to_vector(record.values()),
                    to_metadata(record.metadata())
                });
            }

            store_.insert_batch(records);
            response->set_inserted_count(records.size());
            return Status::OK;
        } catch (const std::invalid_argument& error) {
            return Status(StatusCode::INVALID_ARGUMENT, error.what());
        } catch (const std::exception& error) {
            return Status(StatusCode::INTERNAL, error.what());
        }
    }

    Status Search(
        ServerContext*,
        const api::SearchRequest* request,
        api::SearchResponse* response
    ) override {
        try {
            const auto results = store_.search(
                to_vector(request->query()),
                request->k(),
                to_metadata(request->metadata_filter())
            );

            for (const auto& result : results) {
                auto* output = response->add_results();
                output->set_id(result.id);
                output->set_distance(result.distance);
            }

            return Status::OK;
        } catch (const std::invalid_argument& error) {
            return Status(StatusCode::INVALID_ARGUMENT, error.what());
        } catch (const std::exception& error) {
            return Status(StatusCode::INTERNAL, error.what());
        }
    }

    Status GetStats(
        ServerContext*,
        const api::StatsRequest*,
        api::StatsResponse* response
    ) override {
        response->set_vector_count(store_.size());
        response->set_dimension(store_.config().dimension);

        switch (store_.config().metric) {
            case vectordb::Metric::L2:
                response->set_metric("l2");
                break;
            case vectordb::Metric::COSINE:
                response->set_metric("cosine");
                break;
            case vectordb::Metric::INNER_PRODUCT:
                response->set_metric("inner_product");
                break;
        }

        return Status::OK;
    }

    Status CreateSnapshot(
        ServerContext*,
        const api::SnapshotRequest*,
        api::SnapshotResponse* response
    ) override {
        try {
            store_.snapshot();
            response->set_success(true);
            response->set_message("Snapshot created");
            return Status::OK;
        } catch (const std::exception& error) {
            return Status(StatusCode::INTERNAL, error.what());
        }
    }

private:
    static std::vector<float> to_vector(
        const google::protobuf::RepeatedField<float>& values
    ) {
        return {values.begin(), values.end()};
    }

    static Metadata to_metadata(
        const google::protobuf::Map<std::string, std::string>& values
    ) {
        Metadata result;

        for (const auto& item : values) {
            result.emplace(item.first, item.second);
        }

        return result;
    }

    PersistentStore& store_;
};

vectordb::Metric parse_metric(const std::string& value) {
    if (value == "cosine") {
        return vectordb::Metric::COSINE;
    }
    if (value == "l2") {
        return vectordb::Metric::L2;
    }
    if (value == "inner_product") {
        return vectordb::Metric::INNER_PRODUCT;
    }

    throw std::invalid_argument(
        "Unknown metric '" + value +
        "'. Choose cosine, l2, or inner_product."
    );
}

const char* metric_name(vectordb::Metric metric) {
    switch (metric) {
        case vectordb::Metric::L2:
            return "l2";
        case vectordb::Metric::COSINE:
            return "cosine";
        case vectordb::Metric::INNER_PRODUCT:
            return "inner_product";
    }
    return "unknown";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 5) {
        std::cerr
            << "Usage: " << argv[0]
            << " [address] [dimension] [metric] [storage_directory]\n"
            << "Metrics: cosine, l2, inner_product\n";
        return 2;
    }

    const std::string address =
        argc > 1 ? argv[1] : "127.0.0.1:50051";
    const std::string dimension_arg =
        argc > 2 ? argv[2] : "3";
    const std::string metric_arg =
        argc > 3 ? argv[3] : "cosine";
    const std::string storage_directory =
        argc > 4 ? argv[4] : "data/server_store";

    try {
        std::size_t parsed_characters = 0;
        const unsigned long long parsed_dimension =
            std::stoull(dimension_arg, &parsed_characters);

        if (parsed_characters != dimension_arg.size() ||
            parsed_dimension == 0 ||
            parsed_dimension > 1000000) {
            throw std::invalid_argument(
                "Dimension must be an integer between 1 and 1000000."
            );
        }

        vectordb::IndexConfig config;
        config.dimension = static_cast<std::size_t>(parsed_dimension);
        config.max_elements = 100000;
        config.M = 16;
        config.ef_construction = 100;
        config.ef_search = 50;
        config.metric = parse_metric(metric_arg);

        PersistentStore store(config, storage_directory);
        VectorDBService service(store);

        grpc::ServerBuilder builder;
        builder.AddListeningPort(
            address,
            grpc::InsecureServerCredentials()
        );
        builder.RegisterService(&service);

        std::unique_ptr<grpc::Server> server = builder.BuildAndStart();

        if (!server) {
            std::cerr << "Failed to start gRPC server at "
                      << address << '\n';
            return 1;
        }

        std::cout << "VectorDB gRPC server listening on "
                  << address << '\n';
        std::cout << "Dimension: " << config.dimension
                  << ", metric: " << metric_name(config.metric) << '\n';
        std::cout << "Storage: " << storage_directory << '\n';
        std::cout << "Press Ctrl+C to stop.\n";

        server->Wait();
    } catch (const std::exception& error) {
        std::cerr << "Server startup failed: "
                  << error.what() << '\n';
        return 1;
    }

    return 0;
}
