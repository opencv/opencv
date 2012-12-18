ocv_check_dependencies(opencv_java ${OPENCV_MODULE_opencv_java_OPT_DEPS} ${OPENCV_MODULE_opencv_java_REQ_DEPS})

if(NOT OCV_DEPENDENCIES_FOUND OR NOT ANT_EXECUTABLE)
  return()
endif()

# TODO: This has the same name as the Android test project. That project should
# probably be renamed.
project(opencv_test_java)

set(opencv_test_java_bin_dir "${CMAKE_CURRENT_BINARY_DIR}/.build")

set(android_source_dir "${CMAKE_CURRENT_SOURCE_DIR}/../android_test")

set(java_source_dir ${CMAKE_CURRENT_SOURCE_DIR})

# get project sources
file(GLOB_RECURSE opencv_test_java_files RELATIVE "${android_source_dir}" "${android_source_dir}/res/*" "${android_source_dir}/src/*")
ocv_list_filterout(opencv_test_java_files ".svn")
ocv_list_filterout(opencv_test_java_files ".*#.*")
# These files need to be updated for pure Java.
ocv_list_filterout(opencv_test_java_files ".*OpenCVTestCase.*")
ocv_list_filterout(opencv_test_java_files ".*OpenCVTestRunner.*")
ocv_list_filterout(opencv_test_java_files ".*FASTFeatureDetectorTest.*")

# These are files updated for pure Java.
file(GLOB_RECURSE modified_files RELATIVE "${java_source_dir}" "${java_source_dir}/src/*")
ocv_list_filterout(modified_files ".svn")
ocv_list_filterout(modified_files ".*#.*")

# These are extra jars needed to run the tests.
file(GLOB_RECURSE lib_files RELATIVE "${java_source_dir}" "${java_source_dir}/lib/*")
ocv_list_filterout(lib_files ".svn")
ocv_list_filterout(lib_files ".*#.*")

# This is the generated jar for OpenCV that we're testing.
file(GLOB_RECURSE opencv_jar RELATIVE "${CMAKE_BINARY_DIR}" "${CMAKE_BINARY_DIR}/bin/*.jar")
ocv_list_filterout(opencv_jar ".svn")
ocv_list_filterout(opencv_jar ".*#.*")

# copy sources out from the build tree
set(opencv_test_java_file_deps "")
foreach(f ${opencv_test_java_files})
  add_custom_command(
      OUTPUT "${opencv_test_java_bin_dir}/${f}"
      COMMAND ${CMAKE_COMMAND} -E copy "${android_source_dir}/${f}" "${opencv_test_java_bin_dir}/${f}"
      MAIN_DEPENDENCY "${android_source_dir}/${f}"
      COMMENT "Copying ${f}")
  list(APPEND opencv_test_java_file_deps "${android_source_dir}/${f}" "${opencv_test_java_bin_dir}/${f}")
endforeach()

# Overwrite select Android sources with Java-specific sources.
# Also, copy over the libs we'll need for testing.
foreach(f ${modified_files} ${lib_files})
  add_custom_command(
      OUTPUT "${opencv_test_java_bin_dir}/${f}"
      COMMAND ${CMAKE_COMMAND} -E copy "${java_source_dir}/${f}" "${opencv_test_java_bin_dir}/${f}"
      MAIN_DEPENDENCY "${java_source_dir}/${f}"
      COMMENT "Copying ${f}")
  list(APPEND opencv_test_java_file_deps "${java_source_dir}/${f}")
endforeach()

# Copy the OpenCV jar.
foreach(f ${opencv_jar})
  add_custom_command(
      OUTPUT "${opencv_test_java_bin_dir}/${f}"
      COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_BINARY_DIR}/${f}" "${opencv_test_java_bin_dir}/${f}"
      MAIN_DEPENDENCY "${CMAKE_BINARY_DIR}/${f}"
      COMMENT "Copying ${f}")
  list(APPEND opencv_test_java_file_deps "${CMAKE_BINARY_DIR}/${f}" "${opencv_test_java_bin_dir}/${f}")
endforeach()

# Copy the ant build file.
add_custom_command(
  OUTPUT "${opencv_test_java_bin_dir}/build.xml"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/build.xml" "${opencv_test_java_bin_dir}/build.xml"
  MAIN_DEPENDENCY "${CMAKE_CURRENT_SOURCE_DIR}/build.xml"
  COMMENT "Copying build.xml")
list(APPEND opencv_test_java_file_deps "${CMAKE_CURRENT_SOURCE_DIR}/build.xml" "${opencv_test_java_bin_dir}/build.xml")

# Create a script for running the Java tests and place it in build/bin.
file(WRITE "${CMAKE_BINARY_DIR}/bin/opencv_test_java.sh" "cd ${opencv_test_java_bin_dir};\nant -DjavaLibraryPath=${CMAKE_BINARY_DIR}/lib buildAndTest;\ncd -")

add_custom_target(${PROJECT_NAME} ALL SOURCES ${opencv_test_java_file_deps})
add_dependencies(opencv_tests ${PROJECT_NAME})
