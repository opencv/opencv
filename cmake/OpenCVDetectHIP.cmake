# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
# Copyright (C) 2026, BigVision LLC, all rights reserved.
# Third party copyrights are property of their respective owners.

# Detection and reporting. With HAVE_HIP, opencv_core builds the HIP backend of the non-CPU HAL
# (modules/core/CMakeLists.txt); nothing here may change flags or definitions for other targets.

# check_language() finds hipcc even when no GPU architecture was found and HIP was not enabled
get_property(_ocv_enabled_languages GLOBAL PROPERTY ENABLED_LANGUAGES)
if(NOT "HIP" IN_LIST _ocv_enabled_languages)
  unset(_ocv_enabled_languages)
  message(STATUS "HIP: hipcc not enabled as a language, HIP support disabled.")
  return()
endif()
unset(_ocv_enabled_languages)

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
