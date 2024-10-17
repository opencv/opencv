set(the_description "The ts module")

if(NOT BUILD_opencv_ts AND NOT BUILD_TESTS AND NOT BUILD_PERF_TESTS)
  ocv_module_disable(ts)
endif()

set(OPENCV_MODULE_TYPE STATIC)
set(OPENCV_MODULE_IS_PART_OF_WORLD FALSE)

ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef)

ocv_add_module(ts INTERNAL opencv_core opencv_imgproc opencv_imgcodecs opencv_videoio opencv_highgui)

ocv_glob_module_sources()
ocv_module_include_directories()
ocv_create_module()

# generate config file
set(OPENCV_TESTS_CONFIG_FILE "${CMAKE_BINARY_DIR}/opencv_tests_config.hpp")
set(OPENCV_TESTS_CONFIG_STR "")
if(CMAKE_INSTALL_PREFIX)
  set(OPENCV_TESTS_CONFIG_STR "${OPENCV_TESTS_CONFIG_STR}
#define OPENCV_INSTALL_PREFIX \"${CMAKE_INSTALL_PREFIX}\"
")
endif()
if(OPENCV_TEST_DATA_INSTALL_PATH)
  set(OPENCV_TESTS_CONFIG_STR "${OPENCV_TESTS_CONFIG_STR}
#define OPENCV_TEST_DATA_INSTALL_PATH \"${OPENCV_TEST_DATA_INSTALL_PATH}\"
")
endif()
if(EXISTS "${OPENCV_TESTS_CONFIG_FILE}")
  file(READ "${OPENCV_TESTS_CONFIG_FILE}" __content)
endif()
if(NOT OPENCV_TESTS_CONFIG_STR STREQUAL "${__content}")
  file(WRITE "${OPENCV_TESTS_CONFIG_FILE}" "${OPENCV_TESTS_CONFIG_STR}")
endif()

if(OPENCV_DISABLE_THREAD_SUPPORT)
  # This is required to disable threads in the ts module, as
  # described in `ts_gtest.h`.
  ocv_target_compile_definitions(${the_module} PUBLIC GTEST_HAS_PTHREAD=0)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "QNX")
  ocv_target_link_libraries(${the_module} PUBLIC regex)
endif()
