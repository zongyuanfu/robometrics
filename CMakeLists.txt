cmake_minimum_required(VERSION 3.18)
project(robometrics_esdf LANGUAGES CXX)

find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
set(PYBIND11_FINDPYTHON ON)
if(NOT pybind11_DIR)
  execute_process(COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE result)
  if(NOT result EQUAL 0)
    message(FATAL_ERROR "Install pybind11>=2.12 for ${Python_EXECUTABLE} before configuring")
  endif()
endif()
find_package(pybind11 2.12 CONFIG REQUIRED)

pybind11_add_module(_esdf_cpp NO_EXTRAS cpp/esdf.cpp)
target_compile_features(_esdf_cpp PRIVATE cxx_std_17)
if(MSVC)
  target_compile_options(_esdf_cpp PRIVATE /fp:strict)
else()
  target_compile_options(_esdf_cpp PRIVATE -fno-fast-math -ffp-contract=off)
endif()
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/cpp/esdf.cpp" source_sha256)
target_compile_definitions(_esdf_cpp PRIVATE
  ESDF_SOURCE_SHA256="${source_sha256}" ESDF_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
  ESDF_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
install(TARGETS _esdf_cpp LIBRARY DESTINATION robometrics RUNTIME DESTINATION robometrics)
