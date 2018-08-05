PROJECT (dgt LANGUAGES CXX)

CMAKE_MINIMUM_REQUIRED(VERSION 3.1.0)

ENABLE_LANGUAGE(CUDA)

SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(GTest REQUIRED)

LIST(APPEND CUDA_NVCC_FLAGS "-G -std=c++11 -default-stream per-thread")

ENABLE_TESTING()
INCLUDE_DIRECTORIES(${GTEST_INCLUDE_DIRS})

INCLUDE_DIRECTORIES(
    include
    include/
 )
#
# TEST
#
CUDA_ADD_EXECUTABLE(
    dgt_test
    src/tests/test.cu
    src/dgt_cuda.cu
)

TARGET_LINK_LIBRARIES(
    dgt_test
    ${GTEST_BOTH_LIBRARIES}
    -lntl 
    -lgmp 
)

CUDA_ADD_EXECUTABLE(
    dgt_benchmark
    src/tests/benchmark.cu
    src/dgt_cuda.cu
)

TARGET_LINK_LIBRARIES(
    dgt_benchmark
    ${GTEST_BOTH_LIBRARIES}
    -lntl 
    -lgmp 
    -lcufft
)

ADD_TEST(DGTTests test)