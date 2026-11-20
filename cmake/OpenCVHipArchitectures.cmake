# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
# Copyright (C) 2026, BigVision LLC, all rights reserved.
# Third party copyrights are property of their respective owners.

function(ocv_hip_match_archs _text _regex _out)
  string(REPLACE "\n" ";" _list "${_text}")
  string(REPLACE "," ";" _list "${_list}")
  set(_archs "")
  foreach(_a IN LISTS _list)
    string(STRIP "${_a}" _a)
    if(_a MATCHES "${_regex}" AND NOT _a STREQUAL "gfx000")
      list(APPEND _archs "${_a}")
    endif()
  endforeach()
  if(_archs)
    list(REMOVE_DUPLICATES _archs)
  endif()
  set(${_out} "${_archs}" PARENT_SCOPE)
endfunction()

# The GPUs plugged into this machine, for CMAKE_HIP_ARCHITECTURES=native.
function(ocv_hip_detect_native _out)
  set(_archs "")
  find_program(OPENCV_AMDGPU_ARCH_EXECUTABLE NAMES amdgpu-arch offload-arch
    HINTS "${ROCM_PATH}" ENV ROCM_PATH PATHS /opt/rocm
    PATH_SUFFIXES bin llvm/bin)
  if(OPENCV_AMDGPU_ARCH_EXECUTABLE)
    execute_process(COMMAND "${OPENCV_AMDGPU_ARCH_EXECUTABLE}"
      OUTPUT_VARIABLE _out_txt OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE _res ERROR_QUIET)
    if(_res EQUAL 0)
      ocv_hip_match_archs("${_out_txt}" "^gfx[0-9a-f]+$" _archs)
    endif()
  endif()
  if(NOT _archs)
    find_program(OPENCV_ROCM_AGENT_ENUMERATOR rocm_agent_enumerator
      HINTS "${ROCM_PATH}" ENV ROCM_PATH PATHS /opt/rocm PATH_SUFFIXES bin)
    if(OPENCV_ROCM_AGENT_ENUMERATOR)
      execute_process(COMMAND "${OPENCV_ROCM_AGENT_ENUMERATOR}"
        OUTPUT_VARIABLE _out_txt OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _res ERROR_QUIET)
      if(_res EQUAL 0)
        ocv_hip_match_archs("${_out_txt}" "^gfx[0-9a-f]+$" _archs)
      endif()
    endif()
  endif()
  set(${_out} "${_archs}" PARENT_SCOPE)
endfunction()

# Family-wide targets the toolchain can emit. One per GPU family, forward compatible,
# and independent of what is plugged into the build machine.
function(ocv_hip_detect_generic _out)
  set(_archs "")
  execute_process(COMMAND "${CMAKE_HIP_COMPILER}"
    --target=amdgcn-amd-amdhsa -print-supported-cpus
    OUTPUT_VARIABLE _out_txt ERROR_VARIABLE _out_txt
    RESULT_VARIABLE _res OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(_res EQUAL 0)
    ocv_hip_match_archs("${_out_txt}" "^gfx[0-9]+(-[0-9]+)?-generic$" _archs)
  endif()
  set(${_out} "${_archs}" PARENT_SCOPE)
endfunction()

# Sets CMAKE_HIP_ARCHITECTURES and OCV_HIP_ARCH_FOUND. Must run before enable_language(HIP),
# which otherwise caches its own value from the build machine's GPUs.
#
# Default: the generic targets, so a package built anywhere runs anywhere.
# CMAKE_HIP_ARCHITECTURES=native: the GPUs of this machine, for a local build.
# Anything else is taken as given.
macro(ocv_hip_select_architectures)
  set(OCV_HIP_ARCH_FOUND TRUE)
  if(NOT DEFINED CMAKE_HIP_ARCHITECTURES OR CMAKE_HIP_ARCHITECTURES STREQUAL ""
     OR CMAKE_HIP_ARCHITECTURES STREQUAL "native")
    set(_hip_archs "")
    if(CMAKE_HIP_ARCHITECTURES STREQUAL "native")
      ocv_hip_detect_native(_hip_archs)
      if(NOT _hip_archs)
        message(WARNING "HIP: CMAKE_HIP_ARCHITECTURES=native but no GPU was found.")
      endif()
    else()
      ocv_hip_detect_generic(_hip_archs)
    endif()

    if(NOT _hip_archs)
      ocv_hip_detect_generic(_hip_archs)
    endif()
    if(NOT _hip_archs)
      ocv_hip_detect_native(_hip_archs)
    endif()

    if(_hip_archs)
      set(CMAKE_HIP_ARCHITECTURES "${_hip_archs}" CACHE STRING "HIP architectures to build for" FORCE)
    else()
      set(OCV_HIP_ARCH_FOUND FALSE)
      message(STATUS "HIP: no GPU architecture could be determined, HIP support disabled.")
    endif()
    unset(_hip_archs)
  endif()
endmacro()
