cmake_minimum_required(VERSION 3.1)
project(auto_typing_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Required for pybind11 integration
find_package(pybind11 REQUIRED)

# === Paths ===
# Your own headers
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/../auto_typing/motor_control
)

# Phoenix headers from the Python package path
include_directories(${CMAKE_SOURCE_DIR}/../auto_typing/auto_typing/motor_control)

# Detect architecture and set .so lib path
if (${CMAKE_LIBRARY_ARCHITECTURE} STREQUAL "arm-linux-gnueabihf")
    set(PHOENIX_LIBS_DIR ${CMAKE_SOURCE_DIR}/lib/arm32)
elseif (${CMAKE_LIBRARY_ARCHITECTURE} STREQUAL "aarch64-linux-gnu")
    set(PHOENIX_LIBS_DIR ${CMAKE_SOURCE_DIR}/lib/arm64)
else()
    set(PHOENIX_LIBS_DIR ${CMAKE_SOURCE_DIR}/lib/x86-64)
endif()

# === Build shared library with pybind11 ===
pybind11_add_module(motor_cpp
    src/bindings.cpp
    src/motor_controller.cpp
)

# Link Phoenix .so libraries
target_link_libraries(motor_cpp
    PRIVATE
        ${PHOENIX_LIBS_DIR}/libCTRE_Phoenix.so
        ${PHOENIX_LIBS_DIR}/libCTRE_PhoenixCCI.so
        ${PHOENIX_LIBS_DIR}/libCTRE_PhoenixTools.so
)

# === Install the .so into the Python package ===
# This is relative to the scikit-build install root
install(TARGETS motor_cpp DESTINATION auto_typing/motor_control)
