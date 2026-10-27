cmake_minimum_required(VERSION 3.20)

project(VectorDB
    VERSION 0.1.0
    DESCRIPTION "A lightweight C++ HNSW vector search engine"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)
find_package(Protobuf REQUIRED)
find_package(gRPC CONFIG REQUIRED)

# HNSW vector index and persistence layer.
add_library(vectordb_core
    src/vector_index.cpp
    src/persistent_store.cpp
)

target_include_directories(vectordb_core
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/hnswlib/hnswlib"
)

target_link_libraries(vectordb_core
    PUBLIC Threads::Threads
)

target_compile_options(vectordb_core PRIVATE
    -Wall -Wextra -Wpedantic
)

# Generate protobuf and gRPC C++ bindings during the build.
set(PROTO_FILE "${CMAKE_CURRENT_SOURCE_DIR}/proto/vectordb.proto")
set(PROTO_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")

file(MAKE_DIRECTORY "${PROTO_GEN_DIR}")

add_custom_command(
    OUTPUT
        "${PROTO_GEN_DIR}/vectordb.pb.cc"
        "${PROTO_GEN_DIR}/vectordb.pb.h"
        "${PROTO_GEN_DIR}/vectordb.grpc.pb.cc"
        "${PROTO_GEN_DIR}/vectordb.grpc.pb.h"
    COMMAND
        protobuf::protoc
        "--proto_path=${CMAKE_CURRENT_SOURCE_DIR}/proto"
        "--cpp_out=${PROTO_GEN_DIR}"
        "--grpc_out=${PROTO_GEN_DIR}"
        "--plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>"
        "${PROTO_FILE}"
    DEPENDS
        "${PROTO_FILE}"
        protobuf::protoc
        gRPC::grpc_cpp_plugin
    COMMENT "Generating VectorDB protobuf and gRPC C++ bindings"
    VERBATIM
)

add_library(vectordb_proto
    "${PROTO_GEN_DIR}/vectordb.pb.cc"
    "${PROTO_GEN_DIR}/vectordb.grpc.pb.cc"
)

target_include_directories(vectordb_proto
    PUBLIC "${PROTO_GEN_DIR}"
)

target_link_libraries(vectordb_proto
    PUBLIC
        protobuf::libprotobuf
        gRPC::grpc++
)

# Standalone C++ integration demo.
add_executable(vectordb_demo
    src/main.cpp
)

target_link_libraries(vectordb_demo PRIVATE vectordb_core)

# gRPC server.
add_executable(vectordb_server
    src/server.cpp
)

target_link_libraries(vectordb_server
    PRIVATE
        vectordb_core
        vectordb_proto
        gRPC::grpc++
)

# HNSW versus brute-force benchmark.
add_executable(vectordb_benchmark
    benchmarks/benchmark.cpp
)

target_link_libraries(vectordb_benchmark PRIVATE vectordb_core)
