cmake_minimum_required(VERSION 3.16)

project(RM_Power_Rune)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIGURATION>")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)   # 共享库需要 PIC

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/third_party
)

# Build SDL3 from source
add_subdirectory(third_party/SDL3)

find_package(OpenCV REQUIRED)

# ===== 核心渲染库（动态库） =====
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

if(WIN32)
    set_target_properties(render_core PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

# ===== C API 动态库（供 Python ctypes） =====
add_library(power_rune_c_api SHARED
    src/power_rune_c_api.cpp
)

target_link_libraries(power_rune_c_api
    render_core
    SDL3::SDL3
    ${OpenCV_LIBS}
)

if(WIN32)
    set_target_properties(power_rune_c_api PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

# 可选：简化输出名称，方便 Python 加载
set_target_properties(power_rune_c_api PROPERTIES
    OUTPUT_NAME "power_rune_c_api"
)

# ===== 原有测试和演示程序 =====
add_executable(test
    src/test.cpp
)

target_link_libraries(test
    render_core
    SDL3::SDL3
    ${OpenCV_LIBS}
)

add_executable(demo
    src/demo.cpp
)

target_link_libraries(demo
    render_core
    SDL3::SDL3
    ${OpenCV_LIBS}
)
