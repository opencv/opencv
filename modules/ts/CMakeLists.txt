set(the_description "The ts module")

if(NOT BUILD_opencv_ts AND NOT BUILD_TESTS AND NOT BUILD_PERF_TESTS)
  ocv_module_disable(ts)
endif()

set(OPENCV_MODULE_TYPE STATIC)
set(OPENCV_MODULE_IS_PART_OF_WORLD FALSE)

if(WINRT)
  # WINRT doesn't have access to environment variables
  # so adding corresponding macros during CMake run
  add_env_definitions(OPENCV_TEST_DATA_PATH)
  add_env_definitions(OPENCV_PERF_VALIDATION_DIR)
endif()

ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef)

ocv_add_module(ts INTERNAL opencv_core opencv_imgproc opencv_imgcodecs opencv_videoio opencv_highgui)

ocv_glob_module_sources()
ocv_module_include_directories()
ocv_create_module()
