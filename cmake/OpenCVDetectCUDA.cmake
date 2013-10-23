if(${CMAKE_VERSION} VERSION_LESS "2.8.3")
  message(STATUS "WITH_CUDA flag requires CMake 2.8.3 or newer. CUDA support is disabled.")
  return()
endif()

if(WIN32 AND NOT MSVC)
  message(STATUS "CUDA compilation is disabled (due to only Visual Studio compiler supported on your platform).")
  return()
endif()

if(CMAKE_COMPILER_IS_GNUCXX AND NOT APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  message(STATUS "CUDA compilation is disabled (due to Clang unsupported on your platform).")
  return()
endif()

find_host_package(CUDA 4.2 QUIET)

if(CUDA_FOUND)
  set(HAVE_CUDA 1)

  if(WITH_CUFFT)
    set(HAVE_CUFFT 1)
  endif()

  if(WITH_CUBLAS)
    set(HAVE_CUBLAS 1)
  endif()

  ##############################################################################################
  # Hack for CUDA >5.5 support
  #
  # The patch was submitted to CMake and might be available
  # in the next CMake release.
  #
  # In the future we should check CMake version here, like
  # if(CMAKE_VERSION VERSION_LESS "2.8.13")

  set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
  set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY NEVER)
  set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE NEVER)

  if(NOT "${CUDA_TOOLKIT_ROOT_DIR}" STREQUAL "${OPENCV_CUDA_TOOLKIT_ROOT_DIR_INTERNAL}")
    unset(CUDA_TOOLKIT_TARGET_DIR CACHE)
  endif()

  if(CUDA_VERSION VERSION_GREATER "5.0" AND CMAKE_CROSSCOMPILING AND ${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm" AND EXISTS "${CUDA_TOOLKIT_ROOT_DIR}/targets/armv7-linux-gnueabihf")
    set(CUDA_TOOLKIT_TARGET_DIR "${CUDA_TOOLKIT_ROOT_DIR}/targets/armv7-linux-gnueabihf" CACHE PATH "Toolkit target location.")
  else()
    set(CUDA_TOOLKIT_TARGET_DIR "${CUDA_TOOLKIT_ROOT_DIR}" CACHE PATH "Toolkit target location.")
  endif()

  if(NOT "${CUDA_TOOLKIT_TARGET_DIR}" STREQUAL "${OPENCV_CUDA_TOOLKIT_TARGET_DIR_INTERNAL}")
    unset(CUDA_TOOLKIT_INCLUDE CACHE)
    unset(CUDA_CUDART_LIBRARY CACHE)
    unset(CUDA_CUDA_LIBRARY CACHE)
    unset(CUDA_cupti_LIBRARY CACHE)
    unset(CUDA_cublas_LIBRARY CACHE)
    unset(CUDA_cublasemu_LIBRARY CACHE)
    unset(CUDA_cufft_LIBRARY CACHE)
    unset(CUDA_cufftemu_LIBRARY CACHE)
    unset(CUDA_curand_LIBRARY CACHE)
    unset(CUDA_cusparse_LIBRARY CACHE)
    unset(CUDA_npp_LIBRARY CACHE)
    unset(CUDA_nppc_LIBRARY CACHE)
    unset(CUDA_nppi_LIBRARY CACHE)
    unset(CUDA_npps_LIBRARY CACHE)
    unset(CUDA_nvcuvenc_LIBRARY CACHE)
    unset(CUDA_nvcuvid_LIBRARY CACHE)
  endif()

  # CUDA_TOOLKIT_INCLUDE
  find_path(CUDA_TOOLKIT_INCLUDE
    device_functions.h # Header included in toolkit
    PATHS "${CUDA_TOOLKIT_TARGET_DIR}" "${CUDA_TOOLKIT_ROOT_DIR}"
    ENV CUDA_PATH
    ENV CUDA_INC_PATH
    PATH_SUFFIXES include
    NO_DEFAULT_PATH
  )

  # Search default search paths, after we search our own set of paths.
  find_path(CUDA_TOOLKIT_INCLUDE device_functions.h)
  mark_as_advanced(CUDA_TOOLKIT_INCLUDE)

  macro(opencv_cuda_find_library_local_first_with_path_ext _var _names _doc _path_ext)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
      # CUDA 3.2+ on Windows moved the library directories, so we need the new
      # and old paths.
      set(_cuda_64bit_lib_dir "${_path_ext}lib/x64" "${_path_ext}lib64" "${_path_ext}libx64" )
    endif()
    # CUDA 3.2+ on Windows moved the library directories, so we need to new
    # (lib/Win32) and the old path (lib).
    find_library(${_var}
      NAMES ${_names}
      PATHS "${CUDA_TOOLKIT_TARGET_DIR}" "${CUDA_TOOLKIT_ROOT_DIR}"
      ENV CUDA_PATH
      ENV CUDA_LIB_PATH
      PATH_SUFFIXES ${_cuda_64bit_lib_dir} "${_path_ext}lib/Win32" "${_path_ext}lib" "${_path_ext}libWin32"
      DOC ${_doc}
      NO_DEFAULT_PATH
    )
    # Search default search paths, after we search our own set of paths.
    find_library(${_var} NAMES ${_names} DOC ${_doc})
  endmacro()

  macro(opencv_cuda_find_library_local_first _var _names _doc )
    opencv_cuda_find_library_local_first_with_path_ext( "${_var}" "${_names}" "${_doc}" "" )
  endmacro()

  macro(opencv_find_library_local_first _var _names _doc )
    opencv_cuda_find_library_local_first( "${_var}" "${_names}" "${_doc}" "" )
  endmacro()

  # CUDA_LIBRARIES
  opencv_cuda_find_library_local_first(CUDA_CUDART_LIBRARY cudart "\"cudart\" library")
  if(CUDA_VERSION VERSION_EQUAL "3.0")
    # The cudartemu library only existed for the 3.0 version of CUDA.
    opencv_cuda_find_library_local_first(CUDA_CUDARTEMU_LIBRARY cudartemu "\"cudartemu\" library")
    mark_as_advanced(
      CUDA_CUDARTEMU_LIBRARY
    )
  endif()

  # CUPTI library showed up in cuda toolkit 4.0
  if(NOT CUDA_VERSION VERSION_LESS "4.0")
    opencv_cuda_find_library_local_first_with_path_ext(CUDA_cupti_LIBRARY cupti "\"cupti\" library" "extras/CUPTI/")
    mark_as_advanced(CUDA_cupti_LIBRARY)
  endif()

  # If we are using emulation mode and we found the cudartemu library then use
  # that one instead of cudart.
  if(CUDA_BUILD_EMULATION AND CUDA_CUDARTEMU_LIBRARY)
    set(CUDA_LIBRARIES ${CUDA_CUDARTEMU_LIBRARY})
  else()
    set(CUDA_LIBRARIES ${CUDA_CUDART_LIBRARY})
  endif()
  if(APPLE)
    # We need to add the path to cudart to the linker using rpath, since the
    # library name for the cuda libraries is prepended with @rpath.
    if(CUDA_BUILD_EMULATION AND CUDA_CUDARTEMU_LIBRARY)
      get_filename_component(_cuda_path_to_cudart "${CUDA_CUDARTEMU_LIBRARY}" PATH)
    else()
      get_filename_component(_cuda_path_to_cudart "${CUDA_CUDART_LIBRARY}" PATH)
    endif()
    if(_cuda_path_to_cudart)
      list(APPEND CUDA_LIBRARIES -Wl,-rpath "-Wl,${_cuda_path_to_cudart}")
    endif()
  endif()

  # 1.1 toolkit on linux doesn't appear to have a separate library on
  # some platforms.
  opencv_cuda_find_library_local_first(CUDA_CUDA_LIBRARY cuda "\"cuda\" library (older versions only).")

  mark_as_advanced(
    CUDA_CUDA_LIBRARY
    CUDA_CUDART_LIBRARY
  )

  #######################
  # Look for some of the toolkit helper libraries
  macro(OPENCV_FIND_CUDA_HELPER_LIBS _name)
    opencv_cuda_find_library_local_first(CUDA_${_name}_LIBRARY ${_name} "\"${_name}\" library")
    mark_as_advanced(CUDA_${_name}_LIBRARY)
  endmacro()

  # Search for additional CUDA toolkit libraries.
  if(CUDA_VERSION VERSION_LESS "3.1")
    # Emulation libraries aren't available in version 3.1 onward.
    opencv_find_cuda_helper_libs(cufftemu)
    opencv_find_cuda_helper_libs(cublasemu)
  endif()
  opencv_find_cuda_helper_libs(cufft)
  opencv_find_cuda_helper_libs(cublas)
  if(NOT CUDA_VERSION VERSION_LESS "3.2")
    # cusparse showed up in version 3.2
    opencv_find_cuda_helper_libs(cusparse)
    opencv_find_cuda_helper_libs(curand)
    if (WIN32)
      opencv_find_cuda_helper_libs(nvcuvenc)
      opencv_find_cuda_helper_libs(nvcuvid)
    endif()
  endif()
  if(CUDA_VERSION VERSION_GREATER "5.0")
    # In CUDA 5.5 NPP was splitted onto 3 separate libraries.
    opencv_find_cuda_helper_libs(nppc)
    opencv_find_cuda_helper_libs(nppi)
    opencv_find_cuda_helper_libs(npps)
    set(CUDA_npp_LIBRARY "${CUDA_nppc_LIBRARY};${CUDA_nppi_LIBRARY};${CUDA_npps_LIBRARY}")
  elseif(NOT CUDA_VERSION VERSION_LESS "4.0")
    opencv_find_cuda_helper_libs(npp)
  endif()

  if(CUDA_BUILD_EMULATION)
    set(CUDA_CUFFT_LIBRARIES ${CUDA_cufftemu_LIBRARY})
    set(CUDA_CUBLAS_LIBRARIES ${CUDA_cublasemu_LIBRARY})
  else()
    set(CUDA_CUFFT_LIBRARIES ${CUDA_cufft_LIBRARY})
    set(CUDA_CUBLAS_LIBRARIES ${CUDA_cublas_LIBRARY})
  endif()

  set(OPENCV_CUDA_TOOLKIT_ROOT_DIR_INTERNAL "${CUDA_TOOLKIT_ROOT_DIR}" CACHE INTERNAL
    "This is the value of the last time CUDA_TOOLKIT_ROOT_DIR was set successfully." FORCE)
  set(OPENCV_CUDA_TOOLKIT_TARGET_DIR_INTERNAL "${CUDA_TOOLKIT_TARGET_DIR}" CACHE INTERNAL
    "This is the value of the last time CUDA_TOOLKIT_TARGET_DIR was set successfully." FORCE)

  # Hack for CUDA >5.5 support
  ##############################################################################################

  if(WITH_NVCUVID)
    opencv_find_cuda_helper_libs(nvcuvid)
    set(HAVE_NVCUVID 1)
  endif()

  message(STATUS "CUDA detected: " ${CUDA_VERSION})

  set(_generations "Fermi" "Kepler")
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
    set(__cuda_arch_bin "2.0 2.1(2.0)")
  elseif(CUDA_GENERATION STREQUAL "Kepler")
    if(${CUDA_VERSION} VERSION_LESS "5.0")
      set(__cuda_arch_bin "3.0")
    else()
      set(__cuda_arch_bin "3.0 3.5")
    endif()
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
    if(${CUDA_VERSION} VERSION_LESS "5.0")
      set(__cuda_arch_bin "1.1 1.2 1.3 2.0 2.1(2.0) 3.0")
    else()
      set(__cuda_arch_bin "1.1 1.2 1.3 2.0 2.1(2.0) 3.0 3.5")
    endif()
    set(__cuda_arch_ptx "3.0")
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

  # Tell NVCC to add PTX intermediate code for the specified architectures
  string(REGEX MATCHALL "[0-9]+" ARCH_LIST "${ARCH_PTX_NO_POINTS}")
  foreach(ARCH IN LISTS ARCH_LIST)
    set(NVCC_FLAGS_EXTRA ${NVCC_FLAGS_EXTRA} -gencode arch=compute_${ARCH},code=compute_${ARCH})
    set(OPENCV_CUDA_ARCH_PTX "${OPENCV_CUDA_ARCH_PTX} ${ARCH}")
    set(OPENCV_CUDA_ARCH_FEATURES "${OPENCV_CUDA_ARCH_FEATURES} ${ARCH}")
  endforeach()

  if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "arm")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --target-cpu-architecture=ARM")
  endif()

  # These vars will be processed in other scripts
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${NVCC_FLAGS_EXTRA})
  set(OpenCV_CUDA_CC "${NVCC_FLAGS_EXTRA}")

  message(STATUS "CUDA NVCC target flags: ${CUDA_NVCC_FLAGS}")

  OCV_OPTION(CUDA_FAST_MATH "Enable --use_fast_math for CUDA compiler " OFF)

  if(CUDA_FAST_MATH)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --use_fast_math)
  endif()

  mark_as_advanced(CUDA_BUILD_CUBIN CUDA_BUILD_EMULATION CUDA_VERBOSE_BUILD CUDA_SDK_ROOT_DIR)

  macro(ocv_cuda_compile VAR)
    foreach(var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_DEBUG)
      set(${var}_backup_in_cuda_compile_ "${${var}}")

      # we remove /EHa as it generates warnings under windows
      string(REPLACE "/EHa" "" ${var} "${${var}}")

      # we remove -ggdb3 flag as it leads to preprocessor errors when compiling CUDA files (CUDA 4.1)
      string(REPLACE "-ggdb3" "" ${var} "${${var}}")

      # we remove -Wsign-promo as it generates warnings under linux
      string(REPLACE "-Wsign-promo" "" ${var} "${${var}}")
    endforeach()

    if(BUILD_SHARED_LIBS)
      set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xcompiler -DCVAPI_EXPORTS)
    endif()

    if(UNIX OR APPLE)
      set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xcompiler -fPIC)
    endif()
    if(APPLE)
      set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xcompiler -fno-finite-math-only)
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
