cmake_minimum_required(VERSION 2.8.6)

project(nv_perf_test)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIR})

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
