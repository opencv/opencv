# HIP backend of the non-CPU HAL (AMD GPUs), as a plugin: libopencv_core_accel_hip.so
#
#   cmake -S modules/core/misc/plugins/accel_hip -B build_accel_hip \
#         -DOpenCV_DIR=<OpenCV build directory> \
#         -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx1100"
#   cmake --build build_accel_hip
#
# Put the library next to the OpenCV libraries, in a directory of OPENCV_CORE_PLUGIN_PATH, or
# load it with cv::accel::loadPlugin(). It only loads into the OpenCV version it was built against.

cmake_minimum_required(VERSION 3.21)
project(opencv_core_accel_hip CXX HIP)

get_filename_component(OpenCV_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../.." ABSOLUTE)
include("${OpenCV_SOURCE_DIR}/cmake/OpenCVPluginStandalone.cmake")

find_package(hip REQUIRED)

ocv_create_plugin(core "opencv_core_accel_hip" "hip::host" "HIP"
                  "src/accel/backend_hip.cpp"
                  "src/accel/hip/accel_kernels.hip")
target_compile_definitions(${OPENCV_PLUGIN_NAME} PRIVATE HAVE_ACCEL_HIP=1)
