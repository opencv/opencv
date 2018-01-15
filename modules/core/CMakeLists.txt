set(the_description "The Core Functionality")

ocv_add_dispatched_file(mathfuncs_core SSE2 AVX AVX2)
ocv_add_dispatched_file(stat SSE4_2 AVX2)

ocv_add_module(core
               OPTIONAL opencv_cudev
               WRAP java python js)

set(extra_libs "")

if(WINRT AND CMAKE_SYSTEM_NAME MATCHES WindowsStore AND CMAKE_SYSTEM_VERSION MATCHES "8.0")
  list(APPEND extra_libs ole32.lib)
endif()

if(DEFINED WINRT AND NOT DEFINED ENABLE_WINRT_MODE_NATIVE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /ZW")
endif()

if(HAVE_CUDA)
  ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef -Wenum-compare -Wunused-function -Wshadow)
endif()

if(CV_TRACE AND HAVE_ITT AND BUILD_ITT)
  add_definitions(-DOPENCV_WITH_ITT=1)
endif()

file(GLOB lib_cuda_hdrs        "include/opencv2/${name}/cuda/*.hpp"        "include/opencv2/${name}/cuda/*.h")
file(GLOB lib_cuda_hdrs_detail "include/opencv2/${name}/cuda/detail/*.hpp" "include/opencv2/${name}/cuda/detail/*.h")

source_group("Include\\Cuda Headers"         FILES ${lib_cuda_hdrs})
source_group("Include\\Cuda Headers\\Detail" FILES ${lib_cuda_hdrs_detail})

source_group("Src" FILES "${OPENCV_MODULE_opencv_core_BINARY_DIR}/version_string.inc")

ocv_glob_module_sources(SOURCES "${OPENCV_MODULE_opencv_core_BINARY_DIR}/version_string.inc"
                        HEADERS ${lib_cuda_hdrs} ${lib_cuda_hdrs_detail})

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

ocv_target_link_libraries(${the_module} LINK_PRIVATE
    "${ZLIB_LIBRARIES}" "${OPENCL_LIBRARIES}" "${VA_LIBRARIES}"
    "${LAPACK_LIBRARIES}" "${CPUFEATURES_LIBRARIES}" "${HALIDE_LIBRARIES}"
    "${ITT_LIBRARIES}"
    "${OPENCV_HAL_LINKER_LIBS}"
)

if(HAVE_CUDA)
  ocv_target_compile_definitions(${the_module} PUBLIC OPENCV_TRAITS_ENABLE_DEPRECATED)
endif()

ocv_add_accuracy_tests()
ocv_add_perf_tests()
