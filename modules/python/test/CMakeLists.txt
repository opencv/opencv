set(MODULE_NAME "python_tests")
set(OPENCV_MODULE_IS_PART_OF_WORLD FALSE)
ocv_add_module(${MODULE_NAME} INTERNAL)

set(OPENCV_PYTHON_TESTS_CONFIG_FILE_DIR "${OpenCV_BINARY_DIR}" CACHE INTERNAL "")
set(OPENCV_PYTHON_TESTS_CONFIG_FILE "${OPENCV_PYTHON_TESTS_CONFIG_FILE_DIR}/opencv_python_tests.cfg" CACHE INTERNAL "")

# get list of modules to wrap
set(OPENCV_PYTHON_MODULES)
foreach(m ${OPENCV_MODULES_BUILD})
  if(";${OPENCV_MODULE_${m}_WRAPPERS};" MATCHES ";python.*;" AND HAVE_${m})
    list(APPEND OPENCV_PYTHON_MODULES ${m})
    #message(STATUS "\t${m}")
  endif()
endforeach()

file(RELATIVE_PATH __loc_relative "${OPENCV_PYTHON_TESTS_CONFIG_FILE_DIR}" "${CMAKE_CURRENT_LIST_DIR}")
set(opencv_tests_locations "${__loc_relative}")
foreach(m ${OPENCV_PYTHON_MODULES})
  set(__loc "${OPENCV_MODULE_${m}_LOCATION}/misc/python/test")
  if(EXISTS "${__loc}")
    file(RELATIVE_PATH __loc_relative "${OPENCV_PYTHON_TESTS_CONFIG_FILE_DIR}" "${__loc}")
    list(APPEND opencv_tests_locations "${__loc_relative}")
  endif()
endforeach(m)

string(REPLACE ";" "\n" opencv_tests_locations_ "${opencv_tests_locations}")
ocv_update_file("${OPENCV_PYTHON_TESTS_CONFIG_FILE}" "${opencv_tests_locations_}")

#
# TODO: Install rules (with test data?)
#
