if(NOT ANT_EXECUTABLE)
  return()
endif()

project(opencv_test_java)

set(opencv_test_java_bin_dir "${CMAKE_CURRENT_BINARY_DIR}/.build")
set(test_dir ${CMAKE_CURRENT_SOURCE_DIR})
set(test_common_dir "${CMAKE_CURRENT_SOURCE_DIR}/../common_test")

set(opencv_test_java_file_deps "")

# make sure the build directory exists
file(MAKE_DIRECTORY "${opencv_test_java_bin_dir}")

# 1. gather and copy common test files (resources, utils, etc.)
copy_common_tests(test_common_dir opencv_test_java_bin_dir opencv_test_java_file_deps)

# 2. gather and copy tests from each module
copy_modules_tests(OPENCV_JAVA_MODULES opencv_test_java_bin_dir opencv_test_java_file_deps)

# 3. gather and copy specific files for pure java
file(GLOB_RECURSE test_files RELATIVE "${test_dir}" "${test_dir}/src/*")
file(GLOB_RECURSE test_lib_files RELATIVE "${test_dir}" "${test_dir}/lib/*.jar")
foreach(f ${test_files} ${test_lib_files})
  add_custom_command(OUTPUT "${opencv_test_java_bin_dir}/${f}"
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different "${test_dir}/${f}" "${opencv_test_java_bin_dir}/${f}"
                     DEPENDS "${test_dir}/${f}"
                     COMMENT "Copying ${f}"
                    )
  list(APPEND opencv_test_java_file_deps "${test_dir}/${f}" "${opencv_test_java_bin_dir}/${f}")
endforeach()

# Copy the OpenCV jar after it has been generated.
add_custom_command(OUTPUT "${opencv_test_java_bin_dir}/bin/${JAR_NAME}"
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different "${JAR_FILE}" "${opencv_test_java_bin_dir}/bin/${JAR_NAME}"
                   DEPENDS "${JAR_FILE}"
                   COMMENT "Copying the OpenCV jar"
                  )

add_custom_command(OUTPUT "${opencv_test_java_bin_dir}/build.xml"
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/build.xml" "${opencv_test_java_bin_dir}/build.xml"
                   DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/build.xml"
                   COMMENT "Copying build.xml"
                  )

add_custom_command(OUTPUT "${opencv_test_java_bin_dir}/build/jar/opencv-test.jar"
                   COMMAND "${ANT_EXECUTABLE}" build
                   WORKING_DIRECTORY "${opencv_test_java_bin_dir}"
                   DEPENDS ${opencv_test_java_file_deps} "${opencv_test_java_bin_dir}/build.xml" "${CMAKE_CURRENT_SOURCE_DIR}/build.xml" "${JAR_FILE}" "${opencv_test_java_bin_dir}/bin/${JAR_NAME}"
                   COMMENT "Build Java tests"
                  )

# Not add_custom_command because generator expressions aren't supported in
# OUTPUT file names, and we need to generate different files for different
# configurations.
add_custom_target(${PROJECT_NAME}_properties
                  COMMAND "${CMAKE_COMMAND}" -E echo "opencv.lib.path = $<TARGET_FILE_DIR:${the_module}>"
                    > "${opencv_test_java_bin_dir}/ant-$<CONFIGURATION>.properties"
                  VERBATIM
                 )

add_custom_target(${PROJECT_NAME} ALL
                  DEPENDS ${the_module} ${PROJECT_NAME}_properties
                  SOURCES "${opencv_test_java_bin_dir}/build/jar/opencv-test.jar"
                 )

if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "tests accuracy")
endif()
