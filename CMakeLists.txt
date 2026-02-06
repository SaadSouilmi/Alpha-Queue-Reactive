cmake_minimum_required(VERSION 3.15)
project(queue_reactive CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")

# Find Arrow and Parquet (vcpkg)
find_package(Arrow CONFIG REQUIRED)
find_package(Parquet CONFIG REQUIRED)
find_package(Threads REQUIRED)
find_package(RapidJSON CONFIG REQUIRED)

# Print available targets for debugging
message(STATUS "Arrow found: ${Arrow_FOUND}")
message(STATUS "Parquet found: ${Parquet_FOUND}")

# Fetch Google Test
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

# Enable testing
enable_testing()

# Include directories
include_directories(cpp/include)

# Source files (library code)
set(QR_SOURCES
    cpp/src/orderbook.cpp
    cpp/src/qr_model.cpp
    cpp/src/simulation.cpp
    cpp/src/race.cpp
)

# Simple simulation executable
add_executable(sample
    cpp/bin/sample.cpp
    ${QR_SOURCES}
)

target_link_libraries(sample
    Arrow::arrow_static
    Parquet::parquet_static
)

# Metaorder simulation executable
add_executable(run_metaorder
    cpp/bin/run_metaorder.cpp
    ${QR_SOURCES}
)

target_link_libraries(run_metaorder
    Arrow::arrow_static
    Parquet::parquet_static
    Threads::Threads
)

# Baseline simulation executable (no metaorder)
add_executable(run_baseline
    cpp/bin/run_baseline.cpp
    ${QR_SOURCES}
)

target_link_libraries(run_baseline
    Arrow::arrow_static
    Parquet::parquet_static
    Threads::Threads
)

# Sample with alpha executable
add_executable(sample_alpha
    cpp/bin/sample_alpha.cpp
    ${QR_SOURCES}
)

target_link_libraries(sample_alpha
    Arrow::arrow_static
    Parquet::parquet_static
)

# Aggressive strategy simulation executable
add_executable(sample_strategy
    cpp/bin/sample_strategy.cpp
    ${QR_SOURCES}
)

target_link_libraries(sample_strategy
    Arrow::arrow_static
    Parquet::parquet_static
)

# HFT alpha with configurable race params (JSON config)
add_executable(sample_hft_alpha
    cpp/bin/sample_hft_alpha.cpp
    ${QR_SOURCES}
)

target_include_directories(sample_hft_alpha PRIVATE ${RAPIDJSON_INCLUDE_DIRS})

target_link_libraries(sample_hft_alpha
    Arrow::arrow_static
    Parquet::parquet_static
)

# OrderBook test executable
add_executable(orderbook_tests
    tests/orderbook_test.cpp
    cpp/src/orderbook.cpp
)

target_link_libraries(orderbook_tests
    GTest::gtest_main
)

# QRModel test executable
add_executable(qr_model_tests
    tests/qr_model_test.cpp
    cpp/src/orderbook.cpp
    cpp/src/qr_model.cpp
)

target_link_libraries(qr_model_tests
    GTest::gtest_main
)

# Register tests with CTest
include(GoogleTest)
gtest_discover_tests(orderbook_tests)
gtest_discover_tests(qr_model_tests)
