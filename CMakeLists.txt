cmake_minimum_required(VERSION 3.15)
project(queue_reactive CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")
set(CMAKE_TOOLCHAIN_FILE "~/vcpkg/scripts/buildsystems/vcpkg.cmake")

# Find Arrow and Parquet
find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)

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

# Simple simulation executable
add_executable(sample
    sample.cpp
    orderbook.cpp
    qr_model.cpp
    simulation.cpp
)

target_link_libraries(sample
    Arrow::arrow_shared
    Parquet::parquet_shared
)

# Metaorder simulation executable
add_executable(run_metaorder
    run_metaorder.cpp
    orderbook.cpp
    qr_model.cpp
    simulation.cpp
)

target_link_libraries(run_metaorder
    Arrow::arrow_shared
    Parquet::parquet_shared
)

# OrderBook test executable
add_executable(orderbook_tests
    tests/orderbook_test.cpp
    orderbook.cpp
)

target_link_libraries(orderbook_tests
    GTest::gtest_main
)

# QRModel test executable
add_executable(qr_model_tests
    tests/qr_model_test.cpp
    orderbook.cpp
    qr_model.cpp
)

target_link_libraries(qr_model_tests
    GTest::gtest_main
)

# Register tests with CTest
include(GoogleTest)
gtest_discover_tests(orderbook_tests)
gtest_discover_tests(qr_model_tests)