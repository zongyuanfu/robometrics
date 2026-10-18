cmake_minimum_required(VERSION 3.15...4.4)

project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(
    _esdf_cpp
    robometrics/cpp/esdf.cpp
)

# 将构建信息传入 C++
target_compile_definitions(_esdf_cpp PRIVATE
    ESDF_BUILD_TYPE="$<CONFIG>"
    ESDF_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
)

target_compile_features(_esdf_cpp PRIVATE cxx_std_17)

install(
    TARGETS _esdf_cpp
    LIBRARY DESTINATION robometrics
    RUNTIME DESTINATION robometrics
)
# Standalone numerical API: this target never includes Python headers.
add_library(robometrics_evaluation_core STATIC robometrics/cpp/src/evaluation.cpp)
set_target_properties(robometrics_evaluation_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(robometrics_evaluation_core PUBLIC robometrics/cpp/include)
target_compile_features(robometrics_evaluation_core PUBLIC cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(robometrics_evaluation_core PRIVATE -fno-fast-math -ffp-contract=off)
endif()
pybind11_add_module(_evaluation_cpp robometrics/cpp/bindings/evaluation.cpp)
target_link_libraries(_evaluation_cpp PRIVATE robometrics_evaluation_core)

target_compile_definitions(_evaluation_cpp PRIVATE
    EVALUATION_BUILD_TYPE="$<CONFIG>"
    EVALUATION_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
install(TARGETS _evaluation_cpp LIBRARY DESTINATION robometrics RUNTIME DESTINATION robometrics)
option(ROBOMETRICS_BUILD_TESTS "Build standalone CPU evaluation test" OFF)
if(ROBOMETRICS_BUILD_TESTS)
    enable_testing()
    add_executable(evaluation_core_test tests/evaluation_core_test.cpp)
    target_link_libraries(evaluation_core_test PRIVATE robometrics_evaluation_core)
    add_test(NAME evaluation_core_test COMMAND evaluation_core_test)
endif()
