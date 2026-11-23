# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
# Copyright (C) 2026, BigVision LLC, all rights reserved.
# Third party copyrights are property of their respective owners.

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

# CUDA has been detected by now. Next to HIP it has to be the first-class language too: switching
# it on here would come after the CUDA language was (not) enabled, and lose CUDA silently.
if(HAVE_CUDA AND NOT ENABLE_CUDA_FIRST_CLASS_LANGUAGE)
  message(FATAL_ERROR "HIP: building it together with CUDA needs ENABLE_CUDA_FIRST_CLASS_LANGUAGE=ON")
endif()

set(HAVE_HIP 1)

# hal/gpu needs no device link: a HIP link of the libraries that take it in would drop ROCm from
# their RPATH as an implicit directory, so they stay C++ links
set(CMAKE_HIP_LINKER_PREFERENCE_PROPAGATES 0)

set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_HIP_ARCHITECTURES OR CMAKE_HIP_ARCHITECTURES STREQUAL "")
  set(_hip_default_archs "gfx90a;gfx1100;gfx1101;gfx1201;gfx1036")
  set(_hip_detected_archs "")
  find_program(OPENCV_AMDGPU_ARCH_EXECUTABLE amdgpu-arch
    HINTS "${ROCM_PATH}" ENV ROCM_PATH PATHS /opt/rocm
    PATH_SUFFIXES bin llvm/bin)
  if(NOT OPENCV_AMDGPU_ARCH_EXECUTABLE)
    find_program(OPENCV_ROCM_AGENT_ENUMERATOR rocm_agent_enumerator
      HINTS "${ROCM_PATH}" ENV ROCM_PATH PATHS /opt/rocm PATH_SUFFIXES bin)
  endif()
  if(OPENCV_AMDGPU_ARCH_EXECUTABLE)
    execute_process(COMMAND "${OPENCV_AMDGPU_ARCH_EXECUTABLE}"
      OUTPUT_VARIABLE _amdgpu_arch_out OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE _amdgpu_arch_res ERROR_QUIET)
    if(_amdgpu_arch_res EQUAL 0 AND _amdgpu_arch_out)
      string(REPLACE "\n" ";" _amdgpu_arch_list "${_amdgpu_arch_out}")
      list(REMOVE_DUPLICATES _amdgpu_arch_list)
      foreach(_a IN LISTS _amdgpu_arch_list)
        if(_a MATCHES "^gfx[0-9a-fA-F]+$")
          list(APPEND _hip_detected_archs "${_a}")
        endif()
      endforeach()
    endif()
  elseif(OPENCV_ROCM_AGENT_ENUMERATOR)
    execute_process(COMMAND "${OPENCV_ROCM_AGENT_ENUMERATOR}"
      OUTPUT_VARIABLE _agent_out OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE _agent_res ERROR_QUIET)
    if(_agent_res EQUAL 0 AND _agent_out)
      string(REPLACE "\n" ";" _agent_list "${_agent_out}")
      foreach(_a IN LISTS _agent_list)
        if(_a MATCHES "^gfx[0-9a-fA-F]+$" AND NOT _a STREQUAL "gfx000")
          list(APPEND _hip_detected_archs "${_a}")
        endif()
      endforeach()
      if(_hip_detected_archs)
        list(REMOVE_DUPLICATES _hip_detected_archs)
      endif()
    endif()
  endif()
  if(_hip_detected_archs)
    set(CMAKE_HIP_ARCHITECTURES "${_hip_detected_archs}" CACHE STRING "HIP architectures to build for" FORCE)
  else()
    set(CMAKE_HIP_ARCHITECTURES "${_hip_default_archs}" CACHE STRING "HIP architectures to build for" FORCE)
  endif()
endif()

# the HIP language does not pass on the include paths of hip::amdhip64
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -I${hip_INCLUDE_DIRS}")

message(STATUS "HIP: ROCm ${hip_VERSION}, building for ${CMAKE_HIP_ARCHITECTURES}")
