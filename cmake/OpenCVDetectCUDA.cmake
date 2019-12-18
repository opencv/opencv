if(WIN32 AND NOT MSVC)
  message(STATUS "CUDA compilation is disabled (due to only Visual Studio compiler supported on your platform).")
  return()
endif()

if(NOT UNIX AND CV_CLANG)
  message(STATUS "CUDA compilation is disabled (due to Clang unsupported on your platform).")
  return()
endif()


if(((NOT CMAKE_VERSION VERSION_LESS "3.9.0")  # requires https://gitlab.kitware.com/cmake/cmake/merge_requests/663
      OR OPENCV_CUDA_FORCE_EXTERNAL_CMAKE_MODULE)
    AND NOT OPENCV_CUDA_FORCE_BUILTIN_CMAKE_MODULE)
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

if(CUDA_FOUND)
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

  set(_generations "Fermi" "Kepler" "Maxwell" "Pascal" "Volta" "Turing")
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

  SET(DETECT_ARCHS_COMMAND "${CUDA_NVCC_EXECUTABLE}" ${CUDA_NVCC_FLAGS} "${OpenCV_SOURCE_DIR}/cmake/checks/OpenCVDetectCudaArch.cu" "--run")
  if(WIN32 AND CMAKE_LINKER) #Workaround for VS cl.exe not being in the env. path
    get_filename_component(host_compiler_bindir ${CMAKE_LINKER} DIRECTORY)
    SET(DETECT_ARCHS_COMMAND ${DETECT_ARCHS_COMMAND} "-ccbin" "${host_compiler_bindir}")
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
  elseif(CUDA_GENERATION STREQUAL "Volta")
    set(__cuda_arch_bin "7.0")
  elseif(CUDA_GENERATION STREQUAL "Turing")
    set(__cuda_arch_bin "7.5")
  elseif(CUDA_GENERATION STREQUAL "Auto")
    execute_process( COMMAND ${DETECT_ARCHS_COMMAND}
                     WORKING_DIRECTORY "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/"
                     RESULT_VARIABLE _nvcc_res OUTPUT_VARIABLE _nvcc_out
                     ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    string(REGEX REPLACE ".*\n" "" _nvcc_out "${_nvcc_out}") #Strip leading warning messages, if any
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
      execute_process( COMMAND ${DETECT_ARCHS_COMMAND}
                       WORKING_DIRECTORY "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/"
                       RESULT_VARIABLE _nvcc_res OUTPUT_VARIABLE _nvcc_out
                       ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
      string(REGEX REPLACE ".*\n" "" _nvcc_out "${_nvcc_out}") #Strip leading warning messages, if any
      if(NOT _nvcc_res EQUAL 0)
        message(STATUS "Automatic detection of CUDA generation failed. Going to build for all known architectures.")
        set(__cuda_arch_bin "5.3 6.2 7.2")
      else()
        set(__cuda_arch_bin "${_nvcc_out}")
        string(REPLACE "2.1" "2.1(2.0)" __cuda_arch_bin "${__cuda_arch_bin}")
      endif()
      set(__cuda_arch_ptx "")
    else()
      if(CUDA_VERSION VERSION_LESS "9.0")
        set(__cuda_arch_bin "2.0 3.0 3.5 3.7 5.0 5.2 6.0 6.1")
      elseif(CUDA_VERSION VERSION_LESS "10.0")
        set(__cuda_arch_bin "3.0 3.5 3.7 5.0 5.2 6.0 6.1 7.0")
      else()
        set(__cuda_arch_bin "3.0 3.5 3.7 5.0 5.2 6.0 6.1 7.0 7.5")
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

      if (CV_CLANG)
        # we remove -Winconsistent-missing-override and -Qunused-arguments
        # just in case we are compiling CUDA with gcc but OpenCV with clang
        string(REPLACE "-Winconsistent-missing-override" "" ${var} "${${var}}")
        string(REPLACE "-Qunused-arguments" "" ${var} "${${var}}")
      endif()

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

      # cc1: warning: command line option '-Wsuggest-override' is valid for C++/ObjC++ but not for C
      string(REPLACE "-Wsuggest-override" "" ${var} "${${var}}")

      # issue: #11552 (from OpenCVCompilerOptions.cmake)
      string(REGEX REPLACE "-Wimplicit-fallthrough(=[0-9]+)? " "" ${var} "${${var}}")

      # removal of custom specified options
      if(OPENCV_CUDA_NVCC_FILTEROUT_OPTIONS)
        foreach(__flag ${OPENCV_CUDA_NVCC_FILTEROUT_OPTIONS})
          string(REPLACE "${__flag}" "" ${var} "${${var}}")
        endforeach()
      endif()
    endforeach()
  endmacro()

  macro(ocv_cuda_compile VAR)
    ocv_cuda_filter_options()

    if(BUILD_SHARED_LIBS)
      set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xcompiler -DCVAPI_EXPORTS)
    endif()

    if(UNIX OR APPLE)
      set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xcompiler -fPIC --std=c++11)
    endif()
    if(APPLE)
      set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xcompiler -fno-finite-math-only)
    endif()

    if(CMAKE_CROSSCOMPILING AND (ARM OR AARCH64))
      set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xlinker --unresolved-symbols=ignore-in-shared-libs)
    endif()

    # disabled because of multiple warnings during building nvcc auto generated files
    if(CV_GCC AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "4.6.0")
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
endif()


# ----------------------------------------------------------------------------
# Add CUDA libraries (needed for apps/tools, samples)
# ----------------------------------------------------------------------------
if(HAVE_CUDA)
  # details: https://github.com/NVIDIA/nvidia-docker/issues/775
  if(" ${CUDA_CUDA_LIBRARY}" MATCHES "/stubs/libcuda.so" AND NOT OPENCV_SKIP_CUDA_STUB_WORKAROUND)
    set(CUDA_STUB_ENABLED_LINK_WORKAROUND 1)
    if(EXISTS "${CUDA_CUDA_LIBRARY}" AND NOT OPENCV_SKIP_CUDA_STUB_WORKAROUND_RPATH_LINK)
      set(CUDA_STUB_TARGET_PATH "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/")
      execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink "${CUDA_CUDA_LIBRARY}" "${CUDA_STUB_TARGET_PATH}/libcuda.so.1"
          RESULT_VARIABLE CUDA_STUB_SYMLINK_RESULT)
      if(NOT CUDA_STUB_SYMLINK_RESULT EQUAL 0)
        execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CUDA_CUDA_LIBRARY}" "${CUDA_STUB_TARGET_PATH}/libcuda.so.1"
          RESULT_VARIABLE CUDA_STUB_COPY_RESULT)
        if(NOT CUDA_STUB_COPY_RESULT EQUAL 0)
          set(CUDA_STUB_ENABLED_LINK_WORKAROUND 0)
        endif()
      endif()
      if(CUDA_STUB_ENABLED_LINK_WORKAROUND)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath-link,\"${CUDA_STUB_TARGET_PATH}\"")
      endif()
    else()
      set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")
    endif()
    if(NOT CUDA_STUB_ENABLED_LINK_WORKAROUND)
      message(WARNING "CUDA: workaround for stubs/libcuda.so.1 is not applied")
    endif()
  endif()

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
