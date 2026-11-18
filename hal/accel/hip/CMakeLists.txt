# HIP backend of the non-CPU HAL (AMD GPUs), as a plugin: libopencv_core_accel_hip.so
#
#   cmake -S hal/accel/hip -B build_accel_hip -DOpenCV_DIR=<OpenCV build directory>
#   cmake --build build_accel_hip
#
# ROCm and the GPU architectures are detected; the library is placed next to the OpenCV
# libraries, where it is loaded on first use. It only loads into the OpenCV version it was
# built against, so rebuild it whenever OpenCV is rebuilt.

cmake_minimum_required(VERSION 3.21)
project(opencv_core_accel_hip CXX)

get_filename_component(OpenCV_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

# Before enable_language(HIP): without a build type the HIP sources get no -O flag at all,
# and an unoptimised kernel spills to scratch instead of keeping values in registers.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

include(CheckLanguage)
check_language(HIP)
if(NOT CMAKE_HIP_COMPILER)
  message(FATAL_ERROR "accel_hip: no HIP compiler found. Check the ROCm install, or set ROCM_PATH.")
endif()

# Must precede enable_language(HIP): CMake caches its own CMAKE_HIP_ARCHITECTURES there.
include("${OpenCV_SOURCE_DIR}/cmake/OpenCVHipArchitectures.cmake")
ocv_hip_select_architectures()
enable_language(HIP)

if(NOT DEFINED OPENCV_PLUGIN_DESTINATION AND OpenCV_DIR)
  set(OPENCV_PLUGIN_DESTINATION "${OpenCV_DIR}/lib" CACHE PATH "")
endif()

include("${OpenCV_SOURCE_DIR}/cmake/OpenCVPluginStandalone.cmake")

find_package(hip QUIET
  HINTS "${ROCM_PATH}" ENV ROCM_PATH ENV HIP_PATH PATHS /opt/rocm
  PATH_SUFFIXES lib/cmake/hip lib64/cmake/hip
  NO_DEFAULT_PATH)
if(NOT hip_FOUND)
  find_package(hip REQUIRED)
endif()

include("${OpenCV_SOURCE_DIR}/cmake/OpenCVDetectROCm.cmake")

set(OPENCV_PLUGIN_EXTRA_SRC_FILES
    "${CMAKE_CURRENT_LIST_DIR}/core/ops.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/core/core.hip"
    "${CMAKE_CURRENT_LIST_DIR}/core/flip.hip"
    "${CMAKE_CURRENT_LIST_DIR}/core/transpose.hip")
set(OPENCV_PLUGIN_EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}")

ocv_create_plugin(core "opencv_core_accel_hip" "hip::host" "HIP"
                  "src/accel/hip/backend_hip.cpp")

target_compile_definitions(${OPENCV_PLUGIN_NAME} PRIVATE HAVE_ACCEL_HIP=1 __HIP_PLATFORM_AMD__=1)
target_include_directories(${OPENCV_PLUGIN_NAME} PRIVATE ${hip_INCLUDE_DIRS})
if(ROCM_FOUND)
  target_compile_definitions(${OPENCV_PLUGIN_NAME} PRIVATE
      OPENCV_ROCM_VERSION_MAJOR=${ROCM_VERSION_MAJOR}
      OPENCV_ROCM_VERSION_MINOR=${ROCM_VERSION_MINOR})
  message(STATUS "accel_hip: ROCm ${ROCM_VERSION}, HIP ${hip_VERSION}, building for ${CMAKE_HIP_ARCHITECTURES}")
endif()
