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

target_include_directories(sample PRIVATE ${RAPIDJSON_INCLUDE_DIRS})

target_link_libraries(sample
    Arrow::arrow_static
    Parquet::parquet_static
)

# Strategy simulation executable
add_executable(sample_strategy
    cpp/bin/sample_strategy.cpp
    ${QR_SOURCES}
)

target_include_directories(sample_strategy PRIVATE ${RAPIDJSON_INCLUDE_DIRS})

target_link_libraries(sample_strategy
    Arrow::arrow_static
    Parquet::parquet_static
    Threads::Threads
)

# Metaorder simulation executable
add_executable(run_metaorder
    cpp/bin/run_metaorder.cpp
    ${QR_SOURCES}
)

target_include_directories(run_metaorder PRIVATE ${RAPIDJSON_INCLUDE_DIRS})

target_link_libraries(run_metaorder
    Arrow::arrow_static
    Parquet::parquet_static
    Threads::Threads
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
