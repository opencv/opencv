set(the_description "The Core Functionality")
ocv_add_module(core
               PRIVATE_REQUIRED ${ZLIB_LIBRARIES} "${OPENCL_LIBRARIES}" "${VA_LIBRARIES}"
               OPTIONAL opencv_cudev
               WRAP java python)

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

file(GLOB lib_cuda_hdrs        "include/opencv2/${name}/cuda/*.hpp"        "include/opencv2/${name}/cuda/*.h")
file(GLOB lib_cuda_hdrs_detail "include/opencv2/${name}/cuda/detail/*.hpp" "include/opencv2/${name}/cuda/detail/*.h")

source_group("Cuda Headers"         FILES ${lib_cuda_hdrs})
source_group("Cuda Headers\\Detail" FILES ${lib_cuda_hdrs_detail})

ocv_glob_module_sources(SOURCES "${OPENCV_MODULE_opencv_core_BINARY_DIR}/version_string.inc"
                        HEADERS ${lib_cuda_hdrs} ${lib_cuda_hdrs_detail})

ocv_module_include_directories(${the_module} ${ZLIB_INCLUDE_DIRS})
ocv_create_module(${extra_libs})

ocv_add_accuracy_tests()
ocv_add_perf_tests()
