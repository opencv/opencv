if((WIN32 AND NOT MSVC) OR OPENCV_CMAKE_FORCE_CUDA)
  message(STATUS "CUDA: Compilation is disabled (due to only Visual Studio compiler supported on your platform).")
  return()
endif()

if((NOT UNIX AND CV_CLANG) OR OPENCV_CMAKE_FORCE_CUDA)
  message(STATUS "CUDA: Compilation is disabled (due to Clang unsupported on your platform).")
  return()
endif()

#set(OPENCV_CMAKE_CUDA_DEBUG 1)

if(CUDA_TOOLKIT_ROOT_DIR)
  set(CUDA_TOOLKIT_TARGET_DIR ${CUDA_TOOLKIT_ROOT_DIR})
endif()

if(NOT OPENCV_CUDA_FORCE_BUILTIN_CMAKE_MODULE)
  ocv_update(CUDA_LINK_LIBRARIES_KEYWORD "PRIVATE")
  find_host_package(CUDA "${MIN_VER_CUDA}" QUIET)
else()
  # Use OpenCV's patched "FindCUDA" module
  set(CMAKE_MODULE_PATH "${OpenCV_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

  if(ANDROID)
    set(CUDA_TARGET_OS_VARIANT "Android")
  endif()
  find_host_package(CUDA "${MIN_VER_CUDA}" QUIET)

  list(REMOVE_AT CMAKE_MODULE_PATH 0)
endif()

if(NOT CUDA_FOUND)
  unset(CUDA_ARCH_BIN CACHE)
  unset(CUDA_ARCH_PTX CACHE)
  return()
endif()

unset(CUDA_nvcuvenc_LIBRARY CACHE)
set(HAVE_CUDA 1)
if(NOT CUDA_VERSION VERSION_LESS 11.0)
  # CUDA 11.0 removes nppicom
  ocv_list_filterout(CUDA_nppi_LIBRARY "nppicom")
  ocv_list_filterout(CUDA_npp_LIBRARY "nppicom")
endif()

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

include(cmake/OpenCVDetectCUDAUtils.cmake)

if(WITH_NVCUVID OR WITH_NVCUVENC)
  set(cuda_toolkit_dirs "${CUDA_TOOLKIT_TARGET_DIR}" "${CUDA_TOOLKIT_ROOT_DIR}")
  ocv_check_for_nvidia_video_codec_sdk("${cuda_toolkit_dirs}")
endif()

message(STATUS "CUDA detected: " ${CUDA_VERSION})

ocv_set_cuda_detection_nvcc_flags(CUDA_HOST_COMPILER)
ocv_set_cuda_arch_bin_and_ptx(${CUDA_NVCC_EXECUTABLE})

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
    set(NVCC_FLAGS_EXTRA ${NVCC_FLAGS_EXTRA} -gencode arch=compute_${CMAKE_MATCH_2},code=sm_${CMAKE_MATCH_1})
    set(OPENCV_CUDA_ARCH_BIN "${OPENCV_CUDA_ARCH_BIN} ${CMAKE_MATCH_1}")
    set(OPENCV_CUDA_ARCH_FEATURES "${OPENCV_CUDA_ARCH_FEATURES} ${CMAKE_MATCH_2}")
  else()
    # User didn't explicitly specify PTX for the concrete BIN, we assume PTX=BIN
    set(NVCC_FLAGS_EXTRA ${NVCC_FLAGS_EXTRA} -gencode arch=compute_${ARCH},code=sm_${ARCH})
    set(OPENCV_CUDA_ARCH_BIN "${OPENCV_CUDA_ARCH_BIN} ${ARCH}")
    set(OPENCV_CUDA_ARCH_FEATURES "${OPENCV_CUDA_ARCH_FEATURES} ${ARCH}")
  endif()
endforeach()
set(NVCC_FLAGS_EXTRA ${NVCC_FLAGS_EXTRA} -D_FORCE_INLINES)

# Tell NVCC to add PTX intermediate code for the specified architectures
string(REGEX MATCHALL "[0-9]+" ARCH_LIST "${ARCH_PTX_NO_POINTS}")
foreach(ARCH IN LISTS ARCH_LIST)
  set(NVCC_FLAGS_EXTRA ${NVCC_FLAGS_EXTRA} -gencode arch=compute_${ARCH},code=compute_${ARCH})
  set(OPENCV_CUDA_ARCH_PTX "${OPENCV_CUDA_ARCH_PTX} ${ARCH}")
  set(OPENCV_CUDA_ARCH_FEATURES "${OPENCV_CUDA_ARCH_FEATURES} ${ARCH}")
endforeach()

# These vars will be processed in other scripts
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${NVCC_FLAGS_EXTRA})
set(OpenCV_CUDA_CC "${NVCC_FLAGS_EXTRA}")

if(ANDROID)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xptxas;-dlcm=ca")
endif()

ocv_set_nvcc_threads_for_vs()

message(STATUS "CUDA: NVCC target flags ${CUDA_NVCC_FLAGS}")

OCV_OPTION(CUDA_FAST_MATH "Enable --use_fast_math for CUDA compiler " OFF)

if(CUDA_FAST_MATH)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --use_fast_math)
endif()

OCV_OPTION(CUDA_ENABLE_DELAYLOAD "Enable delayed loading of CUDA DLLs" OFF VISIBLE_IF MSVC)

mark_as_advanced(CUDA_BUILD_CUBIN CUDA_BUILD_EMULATION CUDA_VERBOSE_BUILD CUDA_SDK_ROOT_DIR)

macro(ocv_check_windows_crt_linkage)
  # The new MSVC runtime abstraction is only useable if CUDA is a first class language
  if(WIN32 AND POLICY CMP0091)
    cmake_policy(GET CMP0091 MSVC_RUNTIME_SET_BY_ABSTRACTION)
    if(MSVC_RUNTIME_SET_BY_ABSTRACTION STREQUAL "NEW")
      if(NOT BUILD_SHARED_LIBS AND BUILD_WITH_STATIC_CRT)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
      else()
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
      endif()
    endif()
  endif()
endmacro()

macro(ocv_cuda_compile VAR)
  ocv_cuda_filter_options()
  ocv_check_windows_crt_linkage()
  ocv_nvcc_flags()

  if(UNIX OR APPLE)
    if(NOT " ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} ${CMAKE_CXX_FLAGS_DEBUG} ${CUDA_NVCC_FLAGS}" MATCHES "-std=")
      if(CUDA_VERSION VERSION_LESS "11.0")
        list(APPEND CUDA_NVCC_FLAGS "--std=c++11")
      else()
        list(APPEND CUDA_NVCC_FLAGS "--std=c++14")
      endif()
    endif()
  endif()

  CUDA_COMPILE(${VAR} ${ARGN})

  foreach(var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_DEBUG)
    set(${var} "${${var}_backup_in_cuda_compile_}")
    unset(${var}_backup_in_cuda_compile_)
  endforeach()
endmacro()

if(HAVE_CUDA)
  set(CUDA_LIBS_PATH "")
  foreach(p ${CUDA_LIBRARIES} ${CUDA_npp_LIBRARY})
    get_filename_component(_tmp ${p} PATH)
    list(APPEND CUDA_LIBS_PATH ${_tmp})
  endforeach()

  if(HAVE_CUBLAS)
    foreach(p ${CUDA_cublas_LIBRARY})
      get_filename_component(_tmp ${p} PATH)
      list(APPEND CUDA_LIBS_PATH ${_tmp})
    endforeach()
  endif()

  if(HAVE_CUDNN)
    foreach(p ${CUDNN_LIBRARIES})
      get_filename_component(_tmp ${p} PATH)
      list(APPEND CUDA_LIBS_PATH ${_tmp})
    endforeach()
  endif()

  if(HAVE_CUFFT)
    foreach(p ${CUDA_cufft_LIBRARY})
      get_filename_component(_tmp ${p} PATH)
      list(APPEND CUDA_LIBS_PATH ${_tmp})
    endforeach()
  endif()

  list(REMOVE_DUPLICATES CUDA_LIBS_PATH)
  link_directories(${CUDA_LIBS_PATH})

  set(CUDA_LIBRARIES_ABS ${CUDA_LIBRARIES})
  ocv_convert_to_lib_name(CUDA_LIBRARIES ${CUDA_LIBRARIES})
  set(CUDA_npp_LIBRARY_ABS ${CUDA_npp_LIBRARY})
  ocv_convert_to_lib_name(CUDA_npp_LIBRARY ${CUDA_npp_LIBRARY})
  if(HAVE_CUBLAS)
    set(CUDA_cublas_LIBRARY_ABS ${CUDA_cublas_LIBRARY})
    ocv_convert_to_lib_name(CUDA_cublas_LIBRARY ${CUDA_cublas_LIBRARY})
  endif()
  if(HAVE_CUDNN)
    set(CUDNN_LIBRARIES_ABS ${CUDNN_LIBRARIES})
    ocv_convert_to_lib_name(CUDNN_LIBRARIES ${CUDNN_LIBRARIES})
  endif()
  if(HAVE_CUFFT)
    set(CUDA_cufft_LIBRARY_ABS ${CUDA_cufft_LIBRARY})
    ocv_convert_to_lib_name(CUDA_cufft_LIBRARY ${CUDA_cufft_LIBRARY})
  endif()

  if(CMAKE_GENERATOR MATCHES "Visual Studio"
      AND NOT OPENCV_SKIP_CUDA_CMAKE_SUPPRESS_REGENERATION
  )
    message(STATUS "CUDA: MSVS generator is detected. Disabling CMake re-run checks (CMAKE_SUPPRESS_REGENERATION=ON). You need to run CMake manually if updates are required.")
    set(CMAKE_SUPPRESS_REGENERATION ON)
  endif()
endif()

if(HAVE_CUDA)
  ocv_apply_cuda_stub_workaround("${CUDA_CUDA_LIBRARY}")
  ocv_check_cuda_delayed_load("${CUDA_TOOLKIT_ROOT_DIR}")

  # ----------------------------------------------------------------------------
  # Add CUDA libraries (needed for apps/tools, samples)
  # ----------------------------------------------------------------------------
  set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} ${CUDA_LIBRARIES} ${CUDA_npp_LIBRARY})
  if(HAVE_CUBLAS)
    set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} ${CUDA_cublas_LIBRARY})
  endif()
  if(HAVE_CUDNN)
    set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} ${CUDNN_LIBRARIES})
  endif()
  if(HAVE_CUFFT)
    set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} ${CUDA_cufft_LIBRARY})
  endif()
  foreach(p ${CUDA_LIBS_PATH})
    if(MSVC AND CMAKE_GENERATOR MATCHES "Ninja|JOM")
      set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} ${CMAKE_LIBRARY_PATH_FLAG}"${p}")
    else()
      set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} ${CMAKE_LIBRARY_PATH_FLAG}${p})
    endif()
  endforeach()
endif()
