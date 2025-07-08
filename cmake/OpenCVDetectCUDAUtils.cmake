macro(ocv_check_for_nvidia_video_codec_sdk cuda_toolkit_dirs)
  macro(ocv_cuda_SEARCH_NVCUVID_HEADER _filename _result)
    # place header file under CUDAToolkit_LIBRARY_ROOT
    find_path(_header_result
      ${_filename}
      PATHS ${cuda_toolkit_dirs}
      PATH_SUFFIXES include
      NO_DEFAULT_PATH
      )
    if("x${_header_result}" STREQUAL "x_header_result-NOTFOUND")
      set(${_result} 0)
    else()
      set(${_result} 1)
    endif()
    unset(_header_result CACHE)
  endmacro()
  if(WITH_NVCUVID)
    ocv_cuda_SEARCH_NVCUVID_HEADER("nvcuvid.h" HAVE_NVCUVID_HEADER)
    # make sure to have both header and library before enabling
    if(${HAVE_NVCUVID_HEADER})
      find_cuda_helper_libs(nvcuvid)
      if(CUDA_nvcuvid_LIBRARY)
        set(HAVE_NVCUVID 1)
        message(STATUS "Found NVCUVID: ${CUDA_nvcuvid_LIBRARY}")
      else()
        if(WIN32)
          message(STATUS "NVCUVID: Library not found, WITH_NVCUVID requires Nvidia decoding library nvcuvid.lib to either be inside ${cuda_toolkit_dirs}/lib or its location manually set with CUDA_nvcuvid_LIBRARY, i.e. CUDA_nvcuvid_LIBRARY=${cuda_toolkit_dirs}/lib/nvcuvid.lib")
        else()
          message(STATUS "NVCUVID: Library not found, WITH_NVCUVID requires the Nvidia decoding shared library nvcuvid.so from the driver installation or the location of the stub library to be manually set with CUDA_nvcuvid_LIBRARY i.e. CUDA_nvcuvid_LIBRARY=/home/user/Video_Codec_SDK_X.X.X/Lib/linux/stubs/x86_64/nvcuvid.so")
        endif()
      endif()
    else()
      message(STATUS "NVCUVID: Header not found, WITH_NVCUVID requires Nvidia decoding library header ${cuda_toolkit_dirs}/include/nvcuvid.h")
    endif()
  endif()

  if(WITH_NVCUVENC)
    ocv_cuda_SEARCH_NVCUVID_HEADER("nvEncodeAPI.h" HAVE_NVCUVENC_HEADER)
    if(${HAVE_NVCUVENC_HEADER})
      if(WIN32)
        find_cuda_helper_libs(nvencodeapi)
      else()
        find_cuda_helper_libs(nvidia-encode)
      endif()
      if(CUDA_nvencodeapi_LIBRARY OR CUDA_nvidia-encode_LIBRARY)
        set(HAVE_NVCUVENC 1)
        message(STATUS "Found NVCUVENC: ${CUDA_nvencodeapi_LIBRARY} ${CUDA_nvidia-encode_LIBRARY}")
      else()
        if(WIN32)
          message(STATUS "NVCUVENC: Library not found, WITH_NVCUVENC requires Nvidia encoding library nvencodeapi.lib to either be inside ${cuda_toolkit_dirs}/lib or its location manually set with CUDA_nvencodeapi_LIBRARY, i.e. CUDA_nvencodeapi_LIBRARY=${cuda_toolkit_dirs}/lib/nvencodeapi.lib")
        else()
          message(STATUS "NVCUVENC: Library not found, WITH_NVCUVENC requires the Nvidia encoding shared library libnvidia-encode.so from the driver installation or the location of the stub library to be manually set with CUDA_nvidia-encode_LIBRARY i.e. CUDA_nvidia-encode_LIBRARY=/home/user/Video_Codec_SDK_X.X.X/Lib/linux/stubs/x86_64/libnvidia-encode.so")
        endif()
      endif()
    else()
      message(STATUS "NVCUVENC: Header not found, WITH_NVCUVENC requires Nvidia encoding library header ${cuda_toolkit_dirs}/include/nvEncodeAPI.h")
    endif()
  endif()
endmacro()

# Use CMAKE_CUDA_ARCHITECTURES if provided: order of preference CMAKE_CUDA_ARCHITECTURES > CUDA_GENERATION > CUDA_ARCH_BIN and/or CUDA_ARCH_PTX
function(ocv_check_for_cmake_cuda_architectures)
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    return()
  endif()
  if(CMAKE_CUDA_ARCHITECTURES STREQUAL "all" OR CMAKE_CUDA_ARCHITECTURES STREQUAL "all-major" OR CMAKE_CUDA_ARCHITECTURES STREQUAL "native")
    message(WARNING "CUDA: CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}, special values all, all-major and native are not supported by OpenCV, specify only CUDA real and/or virtual architectures or use combinations of CUDA_ARCH_BIN and CUDA_ARCH_PTX or specify the CUDA_GENERATION where -DCUDA_GENERATION=Auto is equivalent to native!")
    return()
  endif()
  set(internal_ptx "")
  set(internal_bin "")
  foreach(ARCH IN LISTS CMAKE_CUDA_ARCHITECTURES)
    if(ARCH MATCHES "([0-9]+)\-real")
      set(internal_bin ${internal_bin} ${CMAKE_MATCH_1};)
    elseif(ARCH MATCHES "([0-9]+)\-virtual")
      set(internal_ptx ${internal_ptx} ${CMAKE_MATCH_1};)
    elseif(ARCH MATCHES "([0-9]+)")
      set(internal_bin ${internal_bin} ${CMAKE_MATCH_1};)
      set(internal_ptx ${internal_ptx} ${CMAKE_MATCH_1};)
    endif()
  endforeach()
  if(internal_bin OR internal_ptx)
    unset(CUDA_ARCH_BIN CACHE)
    unset(CUDA_ARCH_PTX CACHE)
  endif()
  if(internal_ptx)
    set(CUDA_ARCH_PTX ${internal_ptx} CACHE STRING "Specify 'virtual' PTX architectures to build PTX intermediate code for (see https://docs.opencv.org/4.x/d2/dbc/cuda_intro.html)")
  endif()
  if(internal_bin)
    set(CUDA_ARCH_BIN ${internal_bin} CACHE STRING "Specify 'real' GPU architectures to build binaries for, BIN(PTX) format is supported (see https://docs.opencv.org/4.x/d2/dbc/cuda_intro.html)")
  endif()
  set(CMAKE_CUDA_ARCHITECTURES "" PARENT)
  unset(CUDA_GENERATION CACHE)
endfunction()

macro(ocv_initialize_nvidia_device_generations)
  OCV_OPTION(CUDA_ENABLE_DEPRECATED_GENERATION "Enable deprecated generations in the list" OFF)
  set(_generations "Maxwell" "Pascal" "Volta" "Turing" "Ampere" "Lovelace" "Hopper" "Blackwell")
  if(CUDA_ENABLE_DEPRECATED_GENERATION)
    set(_generations "Fermi" "${_generations}")
    set(_generations "Kepler" "${_generations}")
  endif()
  set(_arch_fermi    "2.0")
  set(_arch_kepler   "3.0;3.5;3.7")
  set(_arch_maxwell  "5.0;5.2")
  set(_arch_pascal   "6.0;6.1")
  set(_arch_volta    "7.0")
  set(_arch_turing   "7.5")
  set(_arch_ampere   "8.0;8.6")
  set(_arch_lovelace "8.9")
  set(_arch_hopper   "9.0")
  set(_arch_blackwell "10.0;12.0")
  if(NOT CMAKE_CROSSCOMPILING)
    list(APPEND _generations "Auto")
  endif()
  set(CUDA_GENERATION "" CACHE STRING "Build CUDA device code only for specific GPU architecture. Leave empty to build for all architectures (see https://docs.opencv.org/4.x/d2/dbc/cuda_intro.html).")
  if( CMAKE_VERSION VERSION_GREATER "2.8" )
    set_property( CACHE CUDA_GENERATION PROPERTY STRINGS "" ${_generations} )
  endif()

  if(CUDA_GENERATION)
    if(NOT ";${_generations};" MATCHES ";${CUDA_GENERATION};")
      string(REPLACE ";" ", " _generations "${_generations}")
      message(FATAL_ERROR "ERROR: ${_generations} Generations are supported.")
    endif()
    unset(CUDA_ARCH_BIN CACHE)
    unset(CUDA_ARCH_PTX CACHE)
  endif()
endmacro()

macro(ocv_set_cuda_detection_nvcc_flags cuda_host_compiler_var)
  if(OPENCV_CUDA_DETECTION_NVCC_FLAGS MATCHES "-ccbin")
  # already specified by user
  elseif(${cuda_host_compiler_var} AND EXISTS "${${cuda_host_compiler_var}}")
    get_filename_component(c_compiler_realpath "${CMAKE_C_COMPILER}" REALPATH)
    # C compiler doesn't work with --run option, forcing C++ compiler instead
    if(${cuda_host_compiler_var} STREQUAL c_compiler_realpath OR ${cuda_host_compiler_var} STREQUAL CMAKE_C_COMPILER)
      if(DEFINED CMAKE_CXX_COMPILER)
        get_filename_component(cxx_compiler_realpath "${CMAKE_CXX_COMPILER}" REALPATH)
        LIST(APPEND OPENCV_CUDA_DETECTION_NVCC_FLAGS -ccbin "${cxx_compiler_realpath}")
      else()
        message(STATUS "CUDA: CMAKE_CXX_COMPILER is not available. You may need to specify ${cuda_host_compiler_var}.")
      endif()
    else()
      LIST(APPEND OPENCV_CUDA_DETECTION_NVCC_FLAGS -ccbin "${${cuda_host_compiler_var}}")
    endif()
  elseif(WIN32 AND CMAKE_LINKER) # Workaround for VS cl.exe not being in the env. path
    get_filename_component(host_compiler_bindir ${CMAKE_LINKER} DIRECTORY)
    LIST(APPEND OPENCV_CUDA_DETECTION_NVCC_FLAGS -ccbin "${host_compiler_bindir}")
  else()
    if(${cuda_host_compiler_var})
      message(STATUS "CUDA: ${cuda_host_compiler_var}='${cuda_host_compiler}' is not valid, autodetection may not work. Specify OPENCV_CUDA_DETECTION_NVCC_FLAGS with -ccbin option for fix that")
    endif()
  endif()
endmacro()

macro(ocv_filter_available_architecture nvcc_executable result_list)
  set(__cache_key_check "${ARGN} : ${nvcc_executable} ${OPENCV_CUDA_DETECTION_NVCC_FLAGS}")
  if(DEFINED OPENCV_CACHE_CUDA_SUPPORTED_CC AND OPENCV_CACHE_CUDA_SUPPORTED_CC_check STREQUAL __cache_key_check)
    set(${result_list} "${OPENCV_CACHE_CUDA_SUPPORTED_CC}")
  else()
    set(CC_LIST ${ARGN})
    foreach(target_arch ${CC_LIST})
      string(REPLACE "." "" target_arch_short "${target_arch}")
      set(NVCC_OPTION "-gencode;arch=compute_${target_arch_short},code=sm_${target_arch_short}")
      set(_cmd "${nvcc_executable}" ${OPENCV_CUDA_DETECTION_NVCC_FLAGS} ${NVCC_OPTION} "${OpenCV_SOURCE_DIR}/cmake/checks/OpenCVDetectCudaArch.cu" --compile)
      execute_process(
          COMMAND ${_cmd}
          WORKING_DIRECTORY "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/"
          RESULT_VARIABLE _nvcc_res
          OUTPUT_VARIABLE _nvcc_out
          ERROR_VARIABLE _nvcc_err
          #ERROR_QUIET
          OUTPUT_STRIP_TRAILING_WHITESPACE
      )
      if(OPENCV_CMAKE_CUDA_DEBUG)
        message(WARNING "COMMAND: ${_cmd}")
        message(STATUS "Result: ${_nvcc_res}")
        message(STATUS "Out: ${_nvcc_out}")
        message(STATUS "Err: ${_nvcc_err}")
      endif()
      if(_nvcc_res EQUAL 0)
        LIST(APPEND ${result_list} "${target_arch}")
      endif()
    endforeach()
    string(STRIP "${${result_list}}" ${result_list})
    if(" ${${result_list}}" STREQUAL " ")
      message(WARNING "CUDA: Autodetection arch list is empty. Please enable OPENCV_CMAKE_CUDA_DEBUG=1 and check/specify OPENCV_CUDA_DETECTION_NVCC_FLAGS variable")
    endif()

    # cache detected values
    set(OPENCV_CACHE_CUDA_SUPPORTED_CC ${${result_list}} CACHE INTERNAL "")
    set(OPENCV_CACHE_CUDA_SUPPORTED_CC_check "${__cache_key_check}" CACHE INTERNAL "")
  endif()
endmacro()

macro(ocv_detect_native_cuda_arch nvcc_executable status output)
  set(OPENCV_CUDA_DETECT_ARCHS_COMMAND "${nvcc_executable}" ${OPENCV_CUDA_DETECTION_NVCC_FLAGS} "${OpenCV_SOURCE_DIR}/cmake/checks/OpenCVDetectCudaArch.cu" "--run")
  set(__cache_key_check "${OPENCV_CUDA_DETECT_ARCHS_COMMAND}")
  if(DEFINED OPENCV_CACHE_CUDA_ACTIVE_CC AND OPENCV_CACHE_CUDA_ACTIVE_CC_check STREQUAL __cache_key_check)
    set(${output} "${OPENCV_CACHE_CUDA_ACTIVE_CC}")
    set(${status} 0)
  else()
    execute_process(
        COMMAND ${OPENCV_CUDA_DETECT_ARCHS_COMMAND}
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/"
        RESULT_VARIABLE ${status}
        OUTPUT_VARIABLE _nvcc_out
        ERROR_VARIABLE _nvcc_err
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(OPENCV_CMAKE_CUDA_DEBUG)
      message(WARNING "COMMAND: ${OPENCV_CUDA_DETECT_ARCHS_COMMAND}")
      message(STATUS "Result: ${${status}}")
      message(STATUS "Out: ${_nvcc_out}")
      message(STATUS "Err: ${_nvcc_err}")
    endif()
    string(REGEX REPLACE ".*\n" "" ${output} "${_nvcc_out}") #Strip leading warning messages, if any

    if(${status} EQUAL 0)
      # cache detected values
      set(OPENCV_CACHE_CUDA_ACTIVE_CC ${${output}} CACHE INTERNAL "")
      set(OPENCV_CACHE_CUDA_ACTIVE_CC_check "${__cache_key_check}" CACHE INTERNAL "")
    endif()
  endif()
endmacro()

macro(ocv_set_cuda_arch_bin_and_ptx nvcc_executable)
  ocv_initialize_nvidia_device_generations()
  set(__cuda_arch_ptx ${CUDA_ARCH_PTX})
  if(CUDA_GENERATION STREQUAL "Fermi")
    set(__cuda_arch_bin ${_arch_fermi})
  elseif(CUDA_GENERATION STREQUAL "Kepler")
    set(__cuda_arch_bin ${_arch_kepler})
  elseif(CUDA_GENERATION STREQUAL "Maxwell")
    set(__cuda_arch_bin ${_arch_maxwell})
  elseif(CUDA_GENERATION STREQUAL "Pascal")
    set(__cuda_arch_bin ${_arch_pascal})
  elseif(CUDA_GENERATION STREQUAL "Volta")
    set(__cuda_arch_bin ${_arch_volta})
  elseif(CUDA_GENERATION STREQUAL "Turing")
    set(__cuda_arch_bin ${_arch_turing})
  elseif(CUDA_GENERATION STREQUAL "Ampere")
    set(__cuda_arch_bin ${_arch_ampere})
  elseif(CUDA_GENERATION STREQUAL "Lovelace")
    set(__cuda_arch_bin ${_arch_lovelace})
  elseif(CUDA_GENERATION STREQUAL "Hopper")
    set(__cuda_arch_bin ${_arch_hopper})
  elseif(CUDA_GENERATION STREQUAL "Blackwell")
    set(__cuda_arch_bin ${_arch_blackwell})
  elseif(CUDA_GENERATION STREQUAL "Auto")
    ocv_detect_native_cuda_arch(${nvcc_executable} _nvcc_res _nvcc_out)
    if(NOT _nvcc_res EQUAL 0)
      message(STATUS "CUDA: Automatic detection of CUDA generation failed. Going to build for all known architectures")
    else()
      string(REGEX MATCHALL "[0-9]+\\.[0-9]" __cuda_arch_bin "${_nvcc_out}")
    endif()
  elseif(CUDA_ARCH_BIN)
    message(STATUS "CUDA: Using CUDA_ARCH_BIN=${CUDA_ARCH_BIN}")
    set(__cuda_arch_bin ${CUDA_ARCH_BIN})
  endif()

  if(NOT DEFINED __cuda_arch_bin AND NOT DEFINED __cuda_arch_ptx)
    if(ARM)
      set(__cuda_arch_bin "3.2")
      set(__cuda_arch_ptx "")
    elseif(AARCH64)
      if(NOT CMAKE_CROSSCOMPILING)
        ocv_detect_native_cuda_arch(${nvcc_executable} _nvcc_res _nvcc_out)
      else()
        set(_nvcc_res -1)  # emulate error, see below
      endif()
      if(NOT _nvcc_res EQUAL 0)
        message(STATUS "CUDA: Automatic detection of CUDA generation failed. Going to build for all known architectures")
        # TX1 (5.3) TX2 (6.2) Xavier (7.2) V100 (7.0) Orin (8.7) Thor (10.1)
        ocv_filter_available_architecture(${nvcc_executable} __cuda_arch_bin
            5.3
            6.2
            7.2
            7.0
            8.7
            10.1
        )
      else()
        set(__cuda_arch_bin "${_nvcc_out}")
      endif()
      set(__cuda_arch_ptx "")
    else()
      ocv_filter_available_architecture(${nvcc_executable} __cuda_arch_bin
          ${_arch_fermi}
          ${_arch_kepler}
          ${_arch_maxwell}
          ${_arch_pascal}
          ${_arch_volta}
          ${_arch_turing}
          ${_arch_ampere}
          ${_arch_lovelace}
          ${_arch_hopper}
          ${_arch_blackwell}
      )
      list(GET __cuda_arch_bin -1 __cuda_arch_ptx)
    endif()
  endif()

  set(CUDA_ARCH_BIN ${__cuda_arch_bin} CACHE STRING "Specify 'real' GPU architectures to build binaries for, BIN(PTX) format is supported (see https://docs.opencv.org/4.x/d2/dbc/cuda_intro.html)")
  set(CUDA_ARCH_PTX ${__cuda_arch_ptx} CACHE STRING "Specify 'virtual' PTX architectures to build PTX intermediate code for (see https://docs.opencv.org/4.x/d2/dbc/cuda_intro.html)")
  string(REGEX REPLACE "\\." "" ARCH_BIN_NO_POINTS "${CUDA_ARCH_BIN}")
  string(REGEX REPLACE "\\." "" ARCH_PTX_NO_POINTS "${CUDA_ARCH_PTX}")

  # Check if user specified 1.0/2.1 compute capability: we don't support it
  macro(ocv_wipeout_deprecated_cc target_cc)
    if(${target_cc} IN_LIST ARCH_BIN_NO_POINTS OR ${target_cc} IN_LIST ARCH_PTX_NO_POINTS)
      message(SEND_ERROR "CUDA: ${target_cc} compute capability is not supported - exclude it from ARCH/PTX list and re-run CMake")
    endif()
  endmacro()
  ocv_wipeout_deprecated_cc("10")
  ocv_wipeout_deprecated_cc("21")
endmacro()

macro(ocv_set_nvcc_threads_for_vs)
  # Tell NVCC the maximum number of threads to be used to execute the compilation steps in parallel
  # (option --threads was introduced in version 11.2)
  if(NOT CUDA_VERSION VERSION_LESS "11.2")
    if(CMAKE_GENERATOR MATCHES "Visual Studio" AND NOT $ENV{CMAKE_BUILD_PARALLEL_LEVEL} STREQUAL "")
      set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--threads=$ENV{CMAKE_BUILD_PARALLEL_LEVEL}")
    endif()
  endif()
endmacro()

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

macro(ocv_nvcc_flags)
  if(BUILD_SHARED_LIBS)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xcompiler=-DCVAPI_EXPORTS)
  endif()

  if(UNIX OR APPLE)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xcompiler=-fPIC)
  endif()
  if(APPLE)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xcompiler=-fno-finite-math-only)
  endif()

  if(WIN32)
	if (NOT (CUDA_VERSION VERSION_LESS "11.2"))
		set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xcudafe --display_error_number --diag-suppress 1394,1388)
	endif()
	if(CUDA_VERSION GREATER "12.8")
		set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr)
		set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xcompiler /Zc:preprocessor)
	endif()
  endif()

  if(CMAKE_CROSSCOMPILING AND (ARM OR AARCH64))
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xlinker --unresolved-symbols=ignore-in-shared-libs)
  endif()

  # disabled because of multiple warnings during building nvcc auto generated files
  if(CV_GCC AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "4.6.0")
    ocv_warnings_disable(CMAKE_CXX_FLAGS -Wunused-but-set-variable)
  endif()
endmacro()

macro(ocv_apply_cuda_stub_workaround cuda_driver_library_path)
  # details: https://github.com/NVIDIA/nvidia-docker/issues/775
  if(" ${cuda_driver_library_path}" MATCHES "/stubs/libcuda.so" AND NOT OPENCV_SKIP_CUDA_STUB_WORKAROUND)
    set(CUDA_STUB_ENABLED_LINK_WORKAROUND 1)
    if(EXISTS "${cuda_driver_library_path}" AND NOT OPENCV_SKIP_CUDA_STUB_WORKAROUND_RPATH_LINK)
      set(CUDA_STUB_TARGET_PATH "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/")
      execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink "${cuda_driver_library_path}" "${CUDA_STUB_TARGET_PATH}/libcuda.so.1"
          RESULT_VARIABLE CUDA_STUB_SYMLINK_RESULT)
      if(NOT CUDA_STUB_SYMLINK_RESULT EQUAL 0)
        execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different "${cuda_driver_library_path}" "${CUDA_STUB_TARGET_PATH}/libcuda.so.1"
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
      message(WARNING "CUDA: Workaround for stubs/libcuda.so.1 is not applied")
    endif()
  endif()
endmacro()

macro(ocv_check_cuda_delayed_load cuda_toolkit_root_dir)
  if(MSVC AND CUDA_ENABLE_DELAYLOAD)
    set(DELAYFLAGS "delayimp.lib")
    file(GLOB CUDA_DLLS "${cuda_toolkit_root_dir}/bin/*.dll")
    foreach(d ${CUDA_DLLS})
      cmake_path(GET "d" FILENAME DLL_NAME)
      if(NOT ${DLL_NAME} MATCHES "cudart")
        set(DELAYFLAGS "${DELAYFLAGS} /DELAYLOAD:${DLL_NAME}")
      endif()
    endforeach()
    set(DELAYFLAGS "${DELAYFLAGS} /DELAYLOAD:nvcuda.dll /DELAYLOAD:nvml.dll /IGNORE:4199")
    set(CMAKE_EXE_LINKER_FLAGS       "${CMAKE_EXE_LINKER_FLAGS} ${DELAYFLAGS}")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${DELAYFLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${DELAYFLAGS}")
  endif()
endmacro()
