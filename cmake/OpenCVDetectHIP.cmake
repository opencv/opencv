# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
# Copyright (C) 2026, BigVision LLC, all rights reserved.
# Third party copyrights are property of their respective owners.

# Detection and reporting only. The HIP backend is a plugin (hal/accel/hip); no OpenCV module
# compiles HIP, so nothing here may change flags or definitions for other targets.

if(NOT CMAKE_HIP_COMPILER)
  message(STATUS "HIP: hipcc not enabled as a language, HIP support disabled.")
  return()
endif()

find_package(hip QUIET
  HINTS "${ROCM_PATH}" ENV ROCM_PATH ENV HIP_PATH PATHS /opt/rocm
  PATH_SUFFIXES lib/cmake/hip lib64/cmake/hip
  NO_DEFAULT_PATH)
if(NOT hip_FOUND)
  find_package(hip QUIET)
endif()
if(NOT hip_FOUND)
  message(STATUS "HIP: find_package(hip) failed, HIP support disabled.")
  return()
endif()

set(HAVE_HIP 1)

include("${OpenCV_SOURCE_DIR}/cmake/OpenCVDetectROCm.cmake")

if(ROCM_FOUND)
  message(STATUS "HIP: ROCm ${ROCM_VERSION}, HIP ${hip_VERSION}, building for ${CMAKE_HIP_ARCHITECTURES}")
else()
  message(STATUS "HIP: ${hip_VERSION} (ROCm version unknown), building for ${CMAKE_HIP_ARCHITECTURES}")
endif()
