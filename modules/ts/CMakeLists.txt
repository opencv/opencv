set(the_description "The ts module")

if(NOT BUILD_opencv_ts
    AND NOT BUILD_TESTS
    AND NOT BUILD_PERF_TESTS
    AND NOT OPENCV_EXPORT_TS
)
  ocv_module_disable(ts)
endif()

if(NOT OPENCV_EXPORT_TS)
  set(OPENCV_MODULE_TYPE STATIC)
endif()
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

if(OPENCV_EXPORT_TS)
  set(m ${the_module})
  foreach(hdr ${OPENCV_MODULE_${m}_HEADERS})
    string(REGEX REPLACE "^.*opencv2/" "opencv2/" hdr2 "${hdr}")
    if(NOT hdr2 MATCHES "opencv2/${m}/private.*" AND hdr2 MATCHES "^(opencv2/?.*)/[^/]+.h(..)?$" )
      install(FILES ${hdr} OPTIONAL DESTINATION "${OPENCV_INCLUDE_INSTALL_PATH}/${CMAKE_MATCH_1}" COMPONENT dev)
    endif()
  endforeach()

  ocv_install_target(${the_module} EXPORT OpenCVTest OPTIONAL
    RUNTIME DESTINATION ${OPENCV_BIN_INSTALL_PATH} COMPONENT dev
    LIBRARY DESTINATION ${OPENCV_LIB_INSTALL_PATH} COMPONENT dev NAMELINK_SKIP
    ARCHIVE DESTINATION ${OPENCV_LIB_INSTALL_PATH} COMPONENT dev
  )
endif()
