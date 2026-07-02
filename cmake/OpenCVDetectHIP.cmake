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

set(HAVE_HIP 1)
# HIP needs the first-class CUDA language; legacy FindCUDA can't compile .hip files.
set(ENABLE_CUDA_FIRST_CLASS_LANGUAGE ON)

# Configs: HIP-only (HAVE_CUDA faked, HIP_STANDALONE=1), HIP+real-CUDA, CUDA-only.
if(NOT HAVE_CUDA AND NOT CMAKE_CUDA_COMPILER)
  # No NVIDIA toolkit -> HIP standalone: fake CUDA so CUDA-aware modules still
  # configure. Test CMAKE_CUDA_COMPILER, not HAVE_CUDA (set later).
  set(HAVE_CUDA 1)
  set(HAVE_HIP_STANDALONE 1)
  # Stub CUDA::cudart targets so modules linking them still configure.
  foreach(_cuda_stub cudart cudart_static)
    if(NOT TARGET CUDA::${_cuda_stub})
      add_library(CUDA::${_cuda_stub} INTERFACE IMPORTED GLOBAL)
    endif()
  endforeach()
  unset(_cuda_stub)
endif()

# Standalone: point CUDAToolkit_INCLUDE_DIRS at ROCm headers so CUDA-aware modules
# get a non-empty include list (combined mode keeps the real CUDA path).
if(HAVE_HIP_STANDALONE)
  get_target_property(_hip_iface_dirs hip::device INTERFACE_INCLUDE_DIRECTORIES)
  if(_hip_iface_dirs)
    set(CUDAToolkit_INCLUDE_DIRS ${_hip_iface_dirs})
  else()
    set(CUDAToolkit_INCLUDE_DIRS "${hip_INCLUDE_DIRS}")
  endif()
  unset(_hip_iface_dirs)
endif()

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

# Define __HIP_PLATFORM_AMD__ for non-CUDA compiles only (nvcc must not see it).
add_compile_options($<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-D__HIP_PLATFORM_AMD__>)
if(NOT HAVE_HIP_STANDALONE)
  # hip::device forces -D__HIP_PLATFORM_AMD__=1 onto nvcc; undo it and set the
  # NVIDIA platform macro via CUDA_FLAGS (appended after CUDA_DEFINES).
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -U__HIP_PLATFORM_AMD__ -D__HIP_PLATFORM_NVIDIA__")
  # Force NVIDIA's thrust ahead of ROCm's: drop the cccl dir from the implicit
  # include list so cmake emits it as an explicit -I (which precedes -isystem).
  foreach(_cuda_inc ${CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES})
    if(EXISTS "${_cuda_inc}/thrust")
      set(OPENCV_HIP_CUDA_THRUST_INCLUDE "${_cuda_inc}" CACHE INTERNAL "" FORCE)
      list(REMOVE_ITEM CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES "${_cuda_inc}")
      break()
    endif()
  endforeach()
  unset(_cuda_inc)
endif()
# CMake's HIP language doesn't propagate hip::amdhip64 include dirs; add explicitly.
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -I${hip_INCLUDE_DIRS}")

set(CUDA_VERSION_STRING "${hip_VERSION}")
if(NOT CUDA_VERSION)
  set(CUDA_VERSION "${hip_VERSION}")
endif()

# Standalone has no NVIDIA toolkit: disable NVIDIA-only cuda* modules.
if(HAVE_HIP_STANDALONE)
  foreach(_nv_mod
      cudaarithm cudabgsegm cudacodec cudafeatures2d cudafilters
      cudaimgproc cudalegacy cudaobjdetect cudaoptflow cudastereo cudawarping)
    if(NOT DEFINED BUILD_opencv_${_nv_mod} OR BUILD_opencv_${_nv_mod})
      set(BUILD_opencv_${_nv_mod} OFF CACHE BOOL
          "Disabled: HIP standalone build has no NVIDIA CUDA libraries" FORCE)
    endif()
  endforeach()
  unset(_nv_mod)
endif()

include("${OpenCV_SOURCE_DIR}/cmake/OpenCVDetectCUDAUtils.cmake")

message(STATUS "HIP: ROCm ${hip_VERSION}, building for ${CMAKE_HIP_ARCHITECTURES}")
