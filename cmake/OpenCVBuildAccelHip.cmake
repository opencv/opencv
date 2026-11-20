# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
# Copyright (C) 2026, BigVision LLC, all rights reserved.
# Third party copyrights are property of their respective owners.

# Builds the HIP backend of the non-CPU HAL from the main tree. It is a plugin, so it is a
# separate CMake project built against this one; ExternalProject drives it after opencv_core.

if(NOT HAVE_HIP OR NOT BUILD_PLUGIN_ACCEL_HIP)
  return()
endif()

if(NOT BUILD_SHARED_LIBS)
  message(STATUS "accel_hip: plugins need a shared build, not building the HIP backend.")
  return()
endif()

include(ExternalProject)

ExternalProject_Add(opencv_core_accel_hip_plugin
  SOURCE_DIR        "${OpenCV_SOURCE_DIR}/hal/accel/hip"
  BINARY_DIR        "${CMAKE_BINARY_DIR}/accel_hip"
  DEPENDS           opencv_core
  CMAKE_ARGS
    "-DOpenCV_DIR=${CMAKE_BINARY_DIR}"
    "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
    "-DCMAKE_HIP_ARCHITECTURES=${CMAKE_HIP_ARCHITECTURES}"
    "-DOPENCV_PLUGIN_DESTINATION=${LIBRARY_OUTPUT_PATH}"
  BUILD_ALWAYS      TRUE
  INSTALL_COMMAND   ""
)

install(DIRECTORY "${LIBRARY_OUTPUT_PATH}/"
        DESTINATION "${OPENCV_LIB_INSTALL_PATH}"
        COMPONENT libs
        FILES_MATCHING PATTERN "libopencv_core_accel_hip*")
