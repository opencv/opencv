if(WIN32 AND NOT MSVC)
  message(STATUS "CUDA compilation is disabled (due to only Visual Studio compiler supported on your platform).")
  return()
endif()

if(CMAKE_COMPILER_IS_GNUCXX AND NOT APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  message(STATUS "CUDA compilation is disabled (due to Clang unsupported on your platform).")
  return()
endif()

set(CMAKE_MODULE_PATH "${OpenCV_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

if(ANDROID)
  set(CUDA_TARGET_OS_VARIANT "Android")
endif()
find_host_package(CUDA "${MIN_VER_CUDA}" QUIET)

list(REMOVE_AT CMAKE_MODULE_PATH 0)

if(CUDA_FOUND)
  set(HAVE_CUDA 1)

  if(WITH_CUFFT)
    set(HAVE_CUFFT 1)
  endif()

  if(WITH_CUBLAS)
    set(HAVE_CUBLAS 1)
  endif()

  if(WITH_NVCUVID)
    find_cuda_helper_libs(nvcuvid)
    if(WIN32)
      find_cuda_helper_libs(nvcuvenc)
    endif()
    if(CUDA_nvcuvid_LIBRARY)
      set(HAVE_NVCUVID 1)
    endif()
    if(CUDA_nvcuvenc_LIBRARY)
      set(HAVE_NVCUVENC 1)
    endif()
  endif()

  message(STATUS "CUDA detected: " ${CUDA_VERSION})

  set(_generations "Fermi" "Kepler" "Maxwell" "Pascal")
  if(NOT CMAKE_CROSSCOMPILING)
    list(APPEND _generations "Auto")
  endif()
  set(CUDA_GENERATION "" CACHE STRING "Build CUDA device code only for specific GPU architecture. Leave empty to build for all architectures.")
  if( CMAKE_VERSION VERSION_GREATER "2.8" )
    set_property( CACHE CUDA_GENERATION PROPERTY STRINGS "" ${_generations} )
  endif()

  if(CUDA_GENERATION)
    if(NOT ";${_generations};" MATCHES ";${CUDA_GENERATION};")
      string(REPLACE ";" ", " _generations "${_generations}")
      message(FATAL_ERROR "ERROR: ${_generations} Generations are suppered.")
    endif()
    unset(CUDA_ARCH_BIN CACHE)
    unset(CUDA_ARCH_PTX CACHE)
  endif()

  set(__cuda_arch_ptx "")
  if(CUDA_GENERATION STREQUAL "Fermi")
    set(__cuda_arch_bin "2.0")
  elseif(CUDA_GENERATION STREQUAL "Kepler")
    set(__cuda_arch_bin "3.0 3.5 3.7")
  elseif(CUDA_GENERATION STREQUAL "Maxwell")
    set(__cuda_arch_bin "5.0 5.2")
  elseif(CUDA_GENERATION STREQUAL "Pascal")
    set(__cuda_arch_bin "6.0 6.1")
  elseif(CUDA_GENERATION STREQUAL "Auto")
    execute_process( COMMAND "${CUDA_NVCC_EXECUTABLE}" "${OpenCV_SOURCE_DIR}/cmake/checks/OpenCVDetectCudaArch.cu" "--run"
                     WORKING_DIRECTORY "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/"
                     RESULT_VARIABLE _nvcc_res OUTPUT_VARIABLE _nvcc_out
                     ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT _nvcc_res EQUAL 0)
      message(STATUS "Automatic detection of CUDA generation failed. Going to build for all known architectures.")
    else()
      set(__cuda_arch_bin "${_nvcc_out}")
      string(REPLACE "2.1" "2.1(2.0)" __cuda_arch_bin "${__cuda_arch_bin}")
    endif()
  endif()

  if(NOT DEFINED __cuda_arch_bin)
    if(ARM)
      set(__cuda_arch_bin "3.2")
      set(__cuda_arch_ptx "")
    elseif(AARCH64)
      execute_process( COMMAND "${CUDA_NVCC_EXECUTABLE}" "${OpenCV_SOURCE_DIR}/cmake/checks/OpenCVDetectCudaArch.cu" "--run"
                       WORKING_DIRECTORY "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/"
                       RESULT_VARIABLE _nvcc_res OUTPUT_VARIABLE _nvcc_out
                       ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
      if(NOT _nvcc_res EQUAL 0)
        message(STATUS "Automatic detection of CUDA generation failed. Going to build for all known architectures.")
        set(__cuda_arch_bin "5.3 6.2")
      else()
        set(__cuda_arch_bin "${_nvcc_out}")
        string(REPLACE "2.1" "2.1(2.0)" __cuda_arch_bin "${__cuda_arch_bin}")
      endif()
      set(__cuda_arch_ptx "")
    else()
      if(${CUDA_VERSION} VERSION_LESS "8.0")
        set(__cuda_arch_bin "2.0 3.0 3.5 3.7 5.0 5.2")
      else()
        set(__cuda_arch_bin "2.0 3.0 3.5 3.7 5.0 5.2 6.0 6.1")
      endif()
    endif()
  endif()

  set(CUDA_ARCH_BIN ${__cuda_arch_bin} CACHE STRING "Specify 'real' GPU architectures to build binaries for, BIN(PTX) format is supported")
  set(CUDA_ARCH_PTX ${__cuda_arch_ptx} CACHE STRING "Specify 'virtual' PTX architectures to build PTX intermediate code for")

  string(REGEX REPLACE "\\." "" ARCH_BIN_NO_POINTS "${CUDA_ARCH_BIN}")
  string(REGEX REPLACE "\\." "" ARCH_PTX_NO_POINTS "${CUDA_ARCH_PTX}")

  # Ckeck if user specified 1.0 compute capability: we don't support it
  string(REGEX MATCH "1.0" HAS_ARCH_10 "${CUDA_ARCH_BIN} ${CUDA_ARCH_PTX}")
  set(CUDA_ARCH_BIN_OR_PTX_10 0)
  if(NOT ${HAS_ARCH_10} STREQUAL "")
    set(CUDA_ARCH_BIN_OR_PTX_10 1)
  endif()

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

  message(STATUS "CUDA NVCC target flags: ${CUDA_NVCC_FLAGS}")

  OCV_OPTION(CUDA_FAST_MATH "Enable --use_fast_math for CUDA compiler " OFF)

  if(CUDA_FAST_MATH)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --use_fast_math)
  endif()

  mark_as_advanced(CUDA_BUILD_CUBIN CUDA_BUILD_EMULATION CUDA_VERBOSE_BUILD CUDA_SDK_ROOT_DIR)

  macro(ocv_cuda_filter_options)
    foreach(var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_DEBUG)
      set(${var}_backup_in_cuda_compile_ "${${var}}")

      # we remove /EHa as it generates warnings under windows
      string(REPLACE "/EHa" "" ${var} "${${var}}")

      # we remove -ggdb3 flag as it leads to preprocessor errors when compiling CUDA files (CUDA 4.1)
      string(REPLACE "-ggdb3" "" ${var} "${${var}}")

      # we remove -Wsign-promo as it generates warnings under linux
      string(REPLACE "-Wsign-promo" "" ${var} "${${var}}")

      # we remove -Wno-sign-promo as it generates warnings under linux
      string(REPLACE "-Wno-sign-promo" "" ${var} "${${var}}")

      # we remove -Wno-delete-non-virtual-dtor because it's used for C++ compiler
      # but NVCC uses C compiler by default
      string(REPLACE "-Wno-delete-non-virtual-dtor" "" ${var} "${${var}}")

      # we remove -frtti because it's used for C++ compiler
      # but NVCC uses C compiler by default
      string(REPLACE "-frtti" "" ${var} "${${var}}")

      string(REPLACE "-fvisibility-inlines-hidden" "" ${var} "${${var}}")
    endforeach()
  endmacro()

  macro(ocv_cuda_compile VAR)
    ocv_cuda_filter_options()

    if(BUILD_SHARED_LIBS)
      set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xcompiler -DCVAPI_EXPORTS)
    endif()

    if(UNIX OR APPLE)
      set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xcompiler -fPIC)
    endif()
    if(APPLE)
      set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xcompiler -fno-finite-math-only)
    endif()

    if(CMAKE_CROSSCOMPILING AND (ARM OR AARCH64))
      set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xlinker --unresolved-symbols=ignore-in-shared-libs)
    endif()

    # disabled because of multiple warnings during building nvcc auto generated files
    if(CMAKE_COMPILER_IS_GNUCXX AND CMAKE_GCC_REGEX_VERSION VERSION_GREATER "4.6.0")
      ocv_warnings_disable(CMAKE_CXX_FLAGS -Wunused-but-set-variable)
    endif()

    CUDA_COMPILE(${VAR} ${ARGN})

    foreach(var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_DEBUG)
      set(${var} "${${var}_backup_in_cuda_compile_}")
      unset(${var}_backup_in_cuda_compile_)
    endforeach()
  endmacro()
else()
  unset(CUDA_ARCH_BIN CACHE)
  unset(CUDA_ARCH_PTX CACHE)
endif()

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

  if(HAVE_CUFFT)
    set(CUDA_cufft_LIBRARY_ABS ${CUDA_cufft_LIBRARY})
    ocv_convert_to_lib_name(CUDA_cufft_LIBRARY ${CUDA_cufft_LIBRARY})
  endif()
endif()
