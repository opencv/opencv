cmake_minimum_required(VERSION 2.8.3)

set(target opencv_gpu_demo_pack)
project(${target})

find_package(OpenCV REQUIRED)

file(COPY "data" DESTINATION ${CMAKE_BINARY_DIR}/bin)

add_subdirectory("utility_lib")
add_subdirectory("demos/surf")
add_subdirectory("demos/orb")
add_subdirectory("demos/stereo_matching")
add_subdirectory("demos/performance")
add_subdirectory("demos/face_detect")
add_subdirectory("demos/pedestrian_detect")
add_subdirectory("demos/optical_flow")
add_subdirectory("demos/stitching")
