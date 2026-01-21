cmake_minimum_required(VERSION 3.10)

project(matexpr_bug)

set(OpenCV_DIR "~/opencv/build")
find_package(OpenCV REQUIRED)

add_executable(matexpr_bug matexpr_bug.cpp)
target_link_libraries(matexpr_bug ${OpenCV_LIBS})
