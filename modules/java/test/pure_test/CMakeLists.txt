if(NOT ANT_EXECUTABLE
    OR NOT BUILD_opencv_imgcodecs
    OR NOT BUILD_opencv_calib3d)
  return()
endif()

project(opencv_test_java)

set(OPENCV_JAR_FILE "${OPENCV_JAR_FILE}")
get_filename_component(JAR_NAME "${OPENCV_JAR_FILE}" NAME)

set(OPENCV_JAVA_TEST_DIR "${OpenCV_BINARY_DIR}/java_test" CACHE INTERNAL "")
file(REMOVE_RECURSE "${OPENCV_JAVA_TEST_DIR}")
file(MAKE_DIRECTORY "${OPENCV_JAVA_TEST_DIR}")
file(REMOVE "${OPENCV_DEPHELPER}/${the_module}_test_source_copy")

set(test_dir ${CMAKE_CURRENT_SOURCE_DIR})

set(depends "")

# 1. gather and copy common test files (resources, utils, etc.)
copy_common_tests("${CMAKE_CURRENT_SOURCE_DIR}/../common_test" "${OPENCV_JAVA_TEST_DIR}" depends)

# 2. gather and copy tests from each module
ocv_copyfiles_append_dir(JAVA_TEST_SRC_COPY "${OPENCV_JAVA_BINDINGS_DIR}/gen/test" "${OPENCV_JAVA_TEST_DIR}/src")

list(APPEND depends gen_opencv_java_source "${OPENCV_DEPHELPER}/gen_opencv_java_source")
ocv_copyfiles_add_target(${the_module}_test_source_copy JAVA_TEST_SRC_COPY "Copy Java(Test) source files" ${depends})
set(depends ${the_module}_test_source_copy "${OPENCV_DEPHELPER}/${the_module}_test_source_copy")

# 3. gather and copy specific files for pure java
file(GLOB_RECURSE test_files RELATIVE "${test_dir}" "${test_dir}/src/*")
file(GLOB_RECURSE test_lib_files RELATIVE "${test_dir}" "${test_dir}/lib/*.jar")
foreach(f ${test_files} ${test_lib_files})
  add_custom_command(OUTPUT "${OPENCV_JAVA_TEST_DIR}/${f}"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${test_dir}/${f}" "${OPENCV_JAVA_TEST_DIR}/${f}"
      DEPENDS "${test_dir}/${f}"
      COMMENT "Copying ${f}"
  )
  list(APPEND depends "${test_dir}/${f}" "${OPENCV_JAVA_TEST_DIR}/${f}")
endforeach()

# Copy the OpenCV jar after it has been generated.
add_custom_command(OUTPUT "${OPENCV_JAVA_TEST_DIR}/bin/${JAR_NAME}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${OPENCV_JAR_FILE}" "${OPENCV_JAVA_TEST_DIR}/bin/${JAR_NAME}"
    DEPENDS "${OPENCV_JAR_FILE}" "${OPENCV_DEPHELPER}/${the_module}_jar"
    COMMENT "Copying the OpenCV jar"
)

add_custom_command(OUTPUT "${OPENCV_JAVA_TEST_DIR}/build.xml"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/build.xml" "${OPENCV_JAVA_TEST_DIR}/build.xml"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/build.xml"
    COMMENT "Copying build.xml"
)

add_custom_command(OUTPUT "${OPENCV_JAVA_TEST_DIR}/build/jar/opencv-test.jar"
    COMMAND "${ANT_EXECUTABLE}" -noinput -k build
    WORKING_DIRECTORY "${OPENCV_JAVA_TEST_DIR}"
    DEPENDS ${depends} "${OPENCV_JAVA_TEST_DIR}/build.xml" "${CMAKE_CURRENT_SOURCE_DIR}/build.xml" "${OPENCV_JAR_FILE}" "${OPENCV_JAVA_TEST_DIR}/bin/${JAR_NAME}"
    COMMENT "Build Java tests"
)

file(GENERATE OUTPUT "${OPENCV_JAVA_TEST_DIR}/ant-$<CONFIGURATION>.properties" CONTENT "opencv.lib.path=$<TARGET_FILE_DIR:${the_module}>")

add_custom_target(${PROJECT_NAME} ALL
    DEPENDS ${the_module} "${OPENCV_JAVA_TEST_DIR}/build/jar/opencv-test.jar"
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/build.xml"
)

if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "tests accuracy")
endif()

add_dependencies(opencv_tests ${PROJECT_NAME})
