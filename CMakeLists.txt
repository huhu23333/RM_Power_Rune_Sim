cmake_minimum_required(VERSION 3.16)

project(RM_ruin)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIGURATION>")

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/third_party
)

# Build SDL3 from source (the only external dependency)
add_subdirectory(third_party/SDL3)

find_package(OpenCV REQUIRED)

# Core library — all non-demo source files compiled into a shared library
add_library(render_core SHARED
    src/camera_projection.cpp
    src/stb_image_impl.cpp
    src/scene_node.cpp
    src/file_utils.cpp
    src/render_utils.cpp
)

target_link_libraries(render_core
    SDL3::SDL3
    ${OpenCV_LIBS}
)

# Test executable
add_executable(test
    src/test.cpp
)

target_link_libraries(test
    render_core
    SDL3::SDL3
    ${OpenCV_LIBS}
)

# Demo executable
add_executable(demo
    src/demo.cpp
)

target_link_libraries(demo
    render_core
    SDL3::SDL3
    ${OpenCV_LIBS}
)