set(the_description "The Core Functionality")

ocv_add_dispatched_file(mathfuncs_core SSE2 AVX AVX2)
ocv_add_dispatched_file(stat SSE4_2 AVX2)
ocv_add_dispatched_file(arithm SSE2 SSE4_1 AVX2 VSX3)
ocv_add_dispatched_file(convert SSE2 AVX2 VSX3)
ocv_add_dispatched_file(convert_scale SSE2 AVX2)
ocv_add_dispatched_file(count_non_zero SSE2 AVX2)
ocv_add_dispatched_file(matmul SSE2 SSE4_1 AVX2 AVX512_SKX)
ocv_add_dispatched_file(mean SSE2 AVX2)
ocv_add_dispatched_file(merge SSE2 AVX2)
ocv_add_dispatched_file(split SSE2 AVX2)
ocv_add_dispatched_file(sum SSE2 AVX2)

# dispatching for accuracy tests
ocv_add_dispatched_file_force_all(test_intrin128 TEST SSE2 SSE3 SSSE3 SSE4_1 SSE4_2 AVX FP16 AVX2 AVX512_SKX)
ocv_add_dispatched_file_force_all(test_intrin256 TEST AVX2 AVX512_SKX)
ocv_add_dispatched_file_force_all(test_intrin512 TEST AVX512_SKX)

ocv_add_module(core
               OPTIONAL opencv_cudev
               WRAP java python js)

set(extra_libs "")

if(WINRT AND WINDOWS_STORE AND CMAKE_SYSTEM_VERSION MATCHES "8.0")
  list(APPEND extra_libs ole32.lib)
endif()

if(HAVE_TBB)
  list(APPEND extra_libs tbb)
endif()

if(DEFINED WINRT AND NOT DEFINED ENABLE_WINRT_MODE_NATIVE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /ZW")
endif()

if(HAVE_CUDA)
  if(NOT HAVE_opencv_cudev)
    message(FATAL_ERROR "CUDA: OpenCV requires enabled 'cudev' module from 'opencv_contrib' repository: https://github.com/opencv/opencv_contrib")
  endif()
  ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef -Wenum-compare -Wunused-function -Wshadow)
endif()

if(CV_TRACE AND HAVE_ITT)
  add_definitions(-DOPENCV_WITH_ITT=1)
endif()

file(GLOB lib_cuda_hdrs
    "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/cuda/*.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/cuda/*.h")
file(GLOB lib_cuda_hdrs_detail
    "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/cuda/detail/*.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/cuda/detail/*.h")
file(GLOB_RECURSE module_opencl_hdrs
    "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/opencl/*")

source_group("Include\\Cuda Headers"         FILES ${lib_cuda_hdrs})
source_group("Include\\Cuda Headers\\Detail" FILES ${lib_cuda_hdrs_detail})

source_group("Src" FILES "${OPENCV_MODULE_opencv_core_BINARY_DIR}/version_string.inc")

ocv_glob_module_sources(SOURCES "${OPENCV_MODULE_opencv_core_BINARY_DIR}/version_string.inc"
                        HEADERS ${module_opencl_hdrs} ${lib_cuda_hdrs} ${lib_cuda_hdrs_detail})

ocv_module_include_directories(${the_module} ${ZLIB_INCLUDE_DIRS} ${OPENCL_INCLUDE_DIRS})
if(ANDROID AND HAVE_CPUFEATURES)
  ocv_append_source_file_compile_definitions(${CMAKE_CURRENT_SOURCE_DIR}/src/system.cpp "HAVE_CPUFEATURES=1")
  ocv_module_include_directories(${CPUFEATURES_INCLUDE_DIRS})
endif()
if(ITT_INCLUDE_DIRS)
  ocv_module_include_directories(${ITT_INCLUDE_DIRS})
endif()
if(HAVE_POSIX_MEMALIGN)
  ocv_append_source_file_compile_definitions(${CMAKE_CURRENT_SOURCE_DIR}/src/alloc.cpp "HAVE_POSIX_MEMALIGN=1")
endif()
if(HAVE_MALLOC_H)
  ocv_append_source_file_compile_definitions(${CMAKE_CURRENT_SOURCE_DIR}/src/alloc.cpp "HAVE_MALLOC_H=1")
endif()
if(HAVE_MEMALIGN)
  ocv_append_source_file_compile_definitions(${CMAKE_CURRENT_SOURCE_DIR}/src/alloc.cpp "HAVE_MEMALIGN=1")
endif()

ocv_create_module(${extra_libs})

ocv_target_link_libraries(${the_module} PRIVATE
    "${ZLIB_LIBRARIES}" "${OPENCL_LIBRARIES}" "${VA_LIBRARIES}"
    "${LAPACK_LIBRARIES}" "${CPUFEATURES_LIBRARIES}" "${HALIDE_LIBRARIES}"
    "${ITT_LIBRARIES}"
    "${OPENCV_HAL_LINKER_LIBS}"
)

if(HAVE_HPX)
  ocv_target_link_libraries(${the_module} LINK_PRIVATE "${HPX_LIBRARIES}")
endif()

ocv_add_accuracy_tests()
ocv_add_perf_tests()

ocv_install_3rdparty_licenses(SoftFloat "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/SoftFloat/COPYING.txt")


# generate data (samples data) config file
set(OPENCV_DATA_CONFIG_FILE "${CMAKE_BINARY_DIR}/opencv_data_config.hpp")
set(OPENCV_DATA_CONFIG_STR "")

if(CMAKE_INSTALL_PREFIX)
  set(OPENCV_DATA_CONFIG_STR "${OPENCV_DATA_CONFIG_STR}
#define OPENCV_INSTALL_PREFIX \"${CMAKE_INSTALL_PREFIX}\"
")
endif()
if(OPENCV_OTHER_INSTALL_PATH)
  set(OPENCV_DATA_CONFIG_STR "${OPENCV_DATA_CONFIG_STR}
#define OPENCV_DATA_INSTALL_PATH \"${OPENCV_OTHER_INSTALL_PATH}\"
")
endif()

set(OPENCV_DATA_CONFIG_STR "${OPENCV_DATA_CONFIG_STR}
#define OPENCV_BUILD_DIR \"${CMAKE_BINARY_DIR}\"
")

file(RELATIVE_PATH SOURCE_DIR_RELATIVE ${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR})
set(OPENCV_DATA_CONFIG_STR "${OPENCV_DATA_CONFIG_STR}
#define OPENCV_DATA_BUILD_DIR_SEARCH_PATHS \\
    \"${SOURCE_DIR_RELATIVE}/\"
")

if(WIN32)
  file(RELATIVE_PATH INSTALL_DATA_DIR_RELATIVE "${CMAKE_INSTALL_PREFIX}/${OPENCV_BIN_INSTALL_PATH}" "${CMAKE_INSTALL_PREFIX}/${OPENCV_OTHER_INSTALL_PATH}")
else()
  file(RELATIVE_PATH INSTALL_DATA_DIR_RELATIVE "${CMAKE_INSTALL_PREFIX}/${OPENCV_LIB_INSTALL_PATH}" "${CMAKE_INSTALL_PREFIX}/${OPENCV_OTHER_INSTALL_PATH}")
endif()
list(APPEND OPENCV_INSTALL_DATA_DIR_RELATIVE "${INSTALL_DATA_DIR_RELATIVE}")
string(REPLACE ";" "\",\\\n    \"" OPENCV_INSTALL_DATA_DIR_RELATIVE_STR "\"${OPENCV_INSTALL_DATA_DIR_RELATIVE}\"")
set(OPENCV_DATA_CONFIG_STR "${OPENCV_DATA_CONFIG_STR}
#define OPENCV_INSTALL_DATA_DIR_RELATIVE ${OPENCV_INSTALL_DATA_DIR_RELATIVE_STR}
")

if(EXISTS "${OPENCV_DATA_CONFIG_FILE}")
  file(READ "${OPENCV_DATA_CONFIG_FILE}" __content)
endif()
if(NOT OPENCV_DATA_CONFIG_STR STREQUAL "${__content}")
  file(WRITE "${OPENCV_DATA_CONFIG_FILE}" "${OPENCV_DATA_CONFIG_STR}")
endif()
