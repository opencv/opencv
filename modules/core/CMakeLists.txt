set(the_description "The Core Functionality")
ocv_add_module(core ${ZLIB_LIBRARIES})
ocv_module_include_directories(${ZLIB_INCLUDE_DIR})

if(HAVE_CUDA)
  ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef)
endif()

file(GLOB lib_device_hdrs        "include/opencv2/${name}/device/*.hpp"        "include/opencv2/${name}/device/*.h")
file(GLOB lib_device_hdrs_detail "include/opencv2/${name}/device/detail/*.hpp" "include/opencv2/${name}/device/detail/*.h")
ocv_glob_module_sources(SOURCES "${opencv_core_BINARY_DIR}/version_string.inc"
                        HEADERS ${lib_device_hdrs} ${lib_device_hdrs_detail})

ocv_create_module()
ocv_add_precompiled_headers(${the_module})

ocv_add_accuracy_tests()
ocv_add_perf_tests()
