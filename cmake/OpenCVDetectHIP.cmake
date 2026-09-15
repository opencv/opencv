# Detect and configure AMD ROCm/HIP support.
#
# This mirrors OpenCVDetectCUDALanguage.cmake. The CUDA device sources keep
# their .cu extension and are compiled by hipcc as the HIP language; the
# existing HAVE_CUDA gated build paths (the src/cuda/*.cu glob, ocv_glob_module_sources,
# ocv_create_module link lines) are reused unchanged by also setting HAVE_CUDA.
# A force included compatibility header maps the CUDA spellings the device code
# uses to their HIP equivalents.
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# Author: Jeff Daily <jeff.daily@amd.com>

if(NOT CMAKE_HIP_COMPILER)
  message(STATUS "HIP: hipcc not enabled as a language, HIP support disabled.")
  return()
endif()

find_package(hip QUIET)
if(NOT hip_FOUND)
  message(STATUS "HIP: find_package(hip) failed, HIP support disabled.")
  return()
endif()

set(HAVE_HIP 1)
# Reuse the existing CUDA code paths (module .cu glob, link lines, gated headers).
set(HAVE_CUDA 1)

# rocPRIM/rocThrust and the cudev templates require C++17.
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_HIP_ARCHITECTURES OR CMAKE_HIP_ARCHITECTURES STREQUAL "")
  set(_hip_default_archs "gfx90a;gfx1100;gfx1101;gfx1201")
  # Native detection: query the GPUs present on the build host.
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

# Force include the CUDA->HIP compatibility shim into every HIP device source.
# The host C++ wrappers pull the same shim explicitly through private.cuda.hpp.
# The AMD platform selector is defined globally so the HIP host headers behave.
set(OPENCV_HIP_COMPAT_HEADER "${OpenCV_SOURCE_DIR}/modules/core/include/opencv2/core/cuda/cuda_to_hip.h")
add_definitions(-D__HIP_PLATFORM_AMD__)
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -include ${OPENCV_HIP_COMPAT_HEADER}")

# hipBLAS and hipFFT provide the cuBLAS/cuFFT functionality cudaarithm uses for
# gemm and dft. Detect them here (before cvconfig.h is generated) and set the
# HAVE_CUBLAS/HAVE_CUFFT switches so both the module sources and the gated
# accuracy tests see them, mirroring the CUDA detection.
find_package(hipblas QUIET)
if(hipblas_FOUND)
  set(HAVE_CUBLAS 1)
endif()
find_package(hipfft QUIET)
if(hipfft_FOUND)
  set(HAVE_CUFFT 1)
endif()

# rocDecode is the ROCm hardware-video-decode library cudacodec::VideoReader uses
# in place of NVCUVID. Detect it (when WITH_ROCDECODE) so the module and its tests
# see HAVE_ROCDECODE; mirrors the NVCUVID detection on the CUDA path.
if(WITH_ROCDECODE)
  find_package(rocdecode QUIET)
  if(rocdecode_FOUND)
    set(HAVE_ROCDECODE 1)
    set(ROCDECODE_LIBRARIES "${rocdecode_LIBRARIES}")
  endif()
endif()

# Expose the same feature defines the CUDA path exposes so gated code lights up.
set(CUDA_VERSION_STRING "${hip_VERSION}")
# Some shared CMake (packaging, version suffixes) keys on CUDA_VERSION even on
# the HIP path; give it the ROCm version so those code paths stay well formed.
if(NOT CUDA_VERSION)
  set(CUDA_VERSION "${hip_VERSION}")
endif()

message(STATUS "HIP: ROCm ${hip_VERSION}, building for ${CMAKE_HIP_ARCHITECTURES}")
