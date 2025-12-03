#######################
# Previously in FindCUDA and still required for FindCUDNN
macro(FIND_CUDA_HELPER_LIBS _name)
  if(CMAKE_CROSSCOMPILING AND (ARM OR AARCH64))
    set(_cuda_cross_arm_lib_dir "lib/stubs")
  endif()
  find_library(CUDA_${_name}_LIBRARY ${_name}
    NAMES ${_name}
    PATHS "${CUDAToolkit_LIBRARY_ROOT}"
    PATH_SUFFIXES "lib/x64" "lib64" ${_cuda_cross_arm_lib_dir} "lib/Win32" "lib"
    DOC "\"${_name}\" library"
    )
  mark_as_advanced(CUDA_${_name}_LIBRARY)
endmacro()
#######################
include(cmake/OpenCVDetectCUDAUtils.cmake)

if((WIN32 AND NOT MSVC) OR OPENCV_CMAKE_FORCE_CUDA)
  message(STATUS "CUDA: Compilation is disabled (due to only Visual Studio compiler supported on your platform).")
  return()
endif()

if((NOT UNIX AND CV_CLANG) OR OPENCV_CMAKE_FORCE_CUDA)
  message(STATUS "CUDA: Compilation is disabled (due to Clang unsupported on your platform).")
  return()
endif()

#set(OPENCV_CMAKE_CUDA_DEBUG 1)

find_package(CUDAToolkit)
if(CMAKE_CUDA_COMPILER AND CUDAToolkit_FOUND)
  set(CUDA_FOUND TRUE)
  set(CUDA_TOOLKIT_INCLUDE ${CUDAToolkit_INCLUDE_DIRS})
  set(CUDA_VERSION_STRING ${CUDAToolkit_VERSION})
  set(CUDA_VERSION ${CUDAToolkit_VERSION})
  if(CUDA_VERSION VERSION_LESS 11.0)
      set(CMAKE_CUDA_STANDARD 11)
  elseif(CUDA_VERSION VERSION_LESS 12.8)
      set(CMAKE_CUDA_STANDARD 14)
  else()
      set(CMAKE_CUDA_STANDARD 17)
  endif()
  if(UNIX AND NOT BUILD_SHARED_LIBS)
      set(CUDA_LIB_EXT "_static")
  endif()
endif()

if(NOT CUDA_FOUND)
  unset(CUDA_ARCH_BIN CACHE)
  unset(CUDA_ARCH_PTX CACHE)
  return()
endif()

set(HAVE_CUDA 1)

if(WITH_CUFFT)
  set(HAVE_CUFFT 1)
endif()

if(WITH_CUBLAS)
  set(HAVE_CUBLAS 1)
endif()

if(WITH_CUDNN)
    set(CMAKE_MODULE_PATH "${OpenCV_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
    find_host_package(CUDNN "${MIN_VER_CUDNN}")
    list(REMOVE_AT CMAKE_MODULE_PATH 0)

    if(CUDNN_FOUND)
      set(HAVE_CUDNN 1)
    endif()
endif()

if(WITH_NVCUVID OR WITH_NVCUVENC)
  ocv_check_for_nvidia_video_codec_sdk("${CUDAToolkit_LIBRARY_ROOT}")
endif()

ocv_check_for_cmake_cuda_architectures()
ocv_set_cuda_detection_nvcc_flags(CMAKE_CUDA_HOST_COMPILER)
ocv_set_cuda_arch_bin_and_ptx(${CUDAToolkit_NVCC_EXECUTABLE})

# NVCC flags to be set
set(NVCC_FLAGS_EXTRA "")

# These vars will be passed into the templates
set(OPENCV_CUDA_ARCH_BIN "")
set(OPENCV_CUDA_ARCH_PTX "")
set(OPENCV_CUDA_ARCH_FEATURES "")

# Tell NVCC to add binaries for the specified GPUs
string(REGEX MATCHALL "[0-9()]+" ARCH_LIST "${ARCH_BIN_NO_POINTS}")
foreach(ARCH IN LISTS ARCH_LIST)
  if(ARCH MATCHES "([0-9]+)\\(([0-9]+)\\)")
    # User explicitly specified PTX for the concrete BIN
    set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES} ${CMAKE_MATCH_2}-virtual;${CMAKE_MATCH_1}-real;)
    set(OPENCV_CUDA_ARCH_BIN "${OPENCV_CUDA_ARCH_BIN} ${CMAKE_MATCH_1}")
    set(OPENCV_CUDA_ARCH_FEATURES "${OPENCV_CUDA_ARCH_FEATURES} ${CMAKE_MATCH_2}")
  else()
    # User didn't explicitly specify PTX for the concrete BIN, we assume PTX=BIN
    set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES} ${ARCH}-real;)
    set(OPENCV_CUDA_ARCH_BIN "${OPENCV_CUDA_ARCH_BIN} ${ARCH}")
    set(OPENCV_CUDA_ARCH_FEATURES "${OPENCV_CUDA_ARCH_FEATURES} ${ARCH}")
  endif()
endforeach()
set(NVCC_FLAGS_EXTRA ${NVCC_FLAGS_EXTRA} -D_FORCE_INLINES)

# Tell NVCC to add PTX intermediate code for the specified architectures
string(REGEX MATCHALL "[0-9]+" ARCH_LIST "${ARCH_PTX_NO_POINTS}")
foreach(ARCH IN LISTS ARCH_LIST)
  set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES} ${ARCH}-virtual;)
  set(OPENCV_CUDA_ARCH_PTX "${OPENCV_CUDA_ARCH_PTX} ${ARCH}")
  set(OPENCV_CUDA_ARCH_FEATURES "${OPENCV_CUDA_ARCH_FEATURES} ${ARCH}")
endforeach()

ocv_set_nvcc_threads_for_vs()

# These vars will be processed in other scripts
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${NVCC_FLAGS_EXTRA})
set(OpenCV_CUDA_CC "${CMAKE_CUDA_ARCHITECTURES}")

if(ANDROID)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xptxas;-dlcm=ca")
endif()

message(STATUS "CUDA: NVCC target flags ${CUDA_NVCC_FLAGS}")

OCV_OPTION(CUDA_FAST_MATH "Enable --use_fast_math for CUDA compiler " OFF)

if(CUDA_FAST_MATH)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --use_fast_math)
endif()

OCV_OPTION(CUDA_ENABLE_DELAYLOAD "Enable delayed loading of CUDA DLLs" OFF VISIBLE_IF MSVC)

mark_as_advanced(CUDA_BUILD_CUBIN CUDA_BUILD_EMULATION CUDA_VERBOSE_BUILD CUDA_SDK_ROOT_DIR)

macro(ocv_cuda_unfilter_options)
  foreach(var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_DEBUG)
    set(${var} "${${var}_backup_in_cuda_compile_}")
    unset(${var}_backup_in_cuda_compile_)
  endforeach()
endmacro()

macro(ocv_cuda_compile_flags)
  ocv_cuda_filter_options()
  ocv_nvcc_flags()
  set(CMAKE_CXX_FLAGS_CUDA ${CMAKE_CXX_FLAGS})
  set(CMAKE_CXX_FLAGS_RELEASE_CUDA ${CMAKE_CXX_FLAGS_RELEASE})
  set(CMAKE_CXX_FLAGS_DEBUG_CUDA ${CMAKE_CXX_FLAGS_DEBUG})
  ocv_cuda_unfilter_options()
endmacro()

if(HAVE_CUDA)
  ocv_apply_cuda_stub_workaround("${CUDA_cuda_driver_LIBRARY}")
  ocv_check_cuda_delayed_load("${cuda_toolkit_root_dir}")
endif()