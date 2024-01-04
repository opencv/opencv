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

if(BUILD_FUZZ_TESTS)
  set(CAN_BUILD_FUZZ_TEST OFF)
  if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/fuzztest)
    add_subdirectory(fuzztest)
    set(CAN_BUILD_FUZZ_TEST ON)
  elseif(${CMAKE_VERSION} VERSION_GREATER "3.11.0")
    include(FetchContent)
    FetchContent_Declare(fuzztest
      GIT_REPOSITORY https://github.com/google/fuzztest.git
      GIT_TAG 5c14a8d~ # Right before CMake 3.19 is mandatory.
    )
    FetchContent_MakeAvailable(fuzztest)
    set_target_properties(fuzztest_googletest_adaptor PROPERTIES LINKER_LANGUAGE CXX)
    set_target_properties(fuzztest_googletest_fixture_adapter PROPERTIES LINKER_LANGUAGE CXX)
    set(CAN_BUILD_FUZZ_TEST ON)
  endif()

  if(CAN_BUILD_FUZZ_TEST)
    link_fuzztest(${the_module})
    ocv_target_compile_definitions(${the_module} PUBLIC OPENCV_BUILD_FUZZ_TESTS)
    ocv_target_link_libraries(${the_module} PUBLIC fuzztest_gtest_main)
  endif()
endif(BUILD_FUZZ_TESTS)
