set(the_description "The Core Functionality")
ocv_add_module(core ${ZLIB_LIBRARIES})
ocv_module_include_directories(${ZLIB_INCLUDE_DIR})

if(HAVE_CUDA)
  ocv_include_directories("${OpenCV_SOURCE_DIR}/modules/gpu/include")
  ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef)
endif()

file(GLOB lib_cuda_hdrs        "include/opencv2/${name}/cuda/*.hpp"        "include/opencv2/${name}/cuda/*.h")
file(GLOB lib_cuda_hdrs_detail "include/opencv2/${name}/cuda/detail/*.hpp" "include/opencv2/${name}/cuda/detail/*.h")

source_group("Cuda Headers"         FILES ${lib_cuda_hdrs})
source_group("Cuda Headers\\Detail" FILES ${lib_cuda_hdrs_detail})

ocv_glob_module_sources(SOURCES "${opencv_core_BINARY_DIR}/version_string.inc"
                        HEADERS ${lib_cuda_hdrs} ${lib_cuda_hdrs_detail})

ocv_create_module()
ocv_add_precompiled_headers(${the_module})

ocv_add_accuracy_tests()
ocv_add_perf_tests()
