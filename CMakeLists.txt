cmake_minimum_required(VERSION 3.18)
project(aerospace_telemetry_cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(TELEMETRY_SOURCES
    src/cpp/ccsds_frame_sync.cpp
    src/cpp/reed_solomon.cpp
    src/cpp/telemetry_parser.cpp
    src/cpp/shared_memory.cpp
)

set(TELEMETRY_HEADERS
    include/ccsds_frame_sync.h
    include/reed_solomon.h
    include/telemetry_parser.h
    include/shared_memory.h
    include/matrix_math.h
    include/ekf_core.h
)

pybind11_add_module(telemetry_core
    src/cpp/pybind_wrapper.cpp
    ${TELEMETRY_SOURCES}
)

target_compile_definitions(telemetry_core PRIVATE BUILD_PYTHON_BINDINGS)

target_include_directories(telemetry_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Python3_INCLUDE_DIRS}
)

add_library(telemetry_core_static STATIC
    ${TELEMETRY_SOURCES}
    ${TELEMETRY_HEADERS}
)

target_include_directories(telemetry_core_static PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

enable_testing()
