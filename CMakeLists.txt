cmake_minimum_required(VERSION 3.15...4.4)

project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(
    _esdf_cpp
    robometrics/cpp/esdf.cpp
)

# 计算 C++ 源码的 SHA256
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/robometrics/cpp/esdf.cpp" ESDF_SOURCE_HASH)

# 将构建信息传入 C++
target_compile_definitions(_esdf_cpp PRIVATE
    ESDF_SOURCE_SHA256="${ESDF_SOURCE_HASH}"
    ESDF_BUILD_TYPE="$<CONFIG>"
    ESDF_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
)

target_compile_features(_esdf_cpp PRIVATE cxx_std_17)

install(
    TARGETS _esdf_cpp
    LIBRARY DESTINATION robometrics
    RUNTIME DESTINATION robometrics
)