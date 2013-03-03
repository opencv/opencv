set(the_description "The Core Functionality")
ocv_add_module(core ${ZLIB_LIBRARIES} OPTIONAL opencv_cudevice)
ocv_module_include_directories(${ZLIB_INCLUDE_DIR})

if(HAVE_CUDA)
  ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef)
endif()

ocv_glob_module_sources(SOURCES "${opencv_core_BINARY_DIR}/version_string.inc")
ocv_create_module()
ocv_add_precompiled_headers(${the_module})

ocv_add_accuracy_tests()
ocv_add_perf_tests()
