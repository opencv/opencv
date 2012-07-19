cmake_minimum_required(VERSION 2.8)

set(target "utility_lib")

include_directories(${OpenCV_INCLUDE_DIR})

file(GLOB sources "*.h*" "*.cpp")
add_library(${target} ${sources})

target_link_libraries(${target} ${OpenCV_LIBS})

