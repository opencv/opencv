ocv_check_dependencies(opencv_java ${OPENCV_MODULE_opencv_java_OPT_DEPS} ${OPENCV_MODULE_opencv_java_REQ_DEPS})

if(NOT OCV_DEPENDENCIES_FOUND)
  return()
endif()

project(opencv_test_java)

set(opencv_test_java_bin_dir "${CMAKE_CURRENT_BINARY_DIR}/.build")
set(android_source_dir "${CMAKE_CURRENT_SOURCE_DIR}/../android_test")
set(java_source_dir ${CMAKE_CURRENT_SOURCE_DIR})

# make sure the build directory exists
file(MAKE_DIRECTORY "${opencv_test_java_bin_dir}")

# get project sources
file(GLOB_RECURSE opencv_test_java_files RELATIVE "${android_source_dir}" "${android_source_dir}/res/*" "${android_source_dir}/src/*.java")
# These are the files that need to be updated for pure Java.
ocv_list_filterout(opencv_test_java_files "OpenCVTest(Case|Runner).java")
# These files aren't for desktop Java.
ocv_list_filterout(opencv_test_java_files "/android/")

# These are files updated for pure Java.
file(GLOB_RECURSE modified_files RELATIVE "${java_source_dir}" "${java_source_dir}/src/*")

# These are extra jars needed to run the tests.
file(GLOB_RECURSE lib_files RELATIVE "${java_source_dir}" "${java_source_dir}/lib/*.jar")

# copy sources out from the build tree
set(opencv_test_java_file_deps "")
foreach(f ${opencv_test_java_files})
  add_custom_command(OUTPUT "${opencv_test_java_bin_dir}/${f}"
                     COMMAND ${CMAKE_COMMAND} -E copy "${android_source_dir}/${f}" "${opencv_test_java_bin_dir}/${f}"
                     DEPENDS "${android_source_dir}/${f}"
                     COMMENT "Copying ${f}"
                    )
  list(APPEND opencv_test_java_file_deps "${android_source_dir}/${f}" "${opencv_test_java_bin_dir}/${f}")
endforeach()

# Overwrite select Android sources with Java-specific sources.
# Also, copy over the libs we'll need for testing.
foreach(f ${modified_files} ${lib_files})
  add_custom_command(OUTPUT "${opencv_test_java_bin_dir}/${f}"
                     COMMAND ${CMAKE_COMMAND} -E copy "${java_source_dir}/${f}" "${opencv_test_java_bin_dir}/${f}"
                     DEPENDS "${java_source_dir}/${f}"
                     COMMENT "Copying ${f}"
                    )
  list(APPEND opencv_test_java_file_deps "${java_source_dir}/${f}" "${opencv_test_java_bin_dir}/${f}")
endforeach()

# Copy the OpenCV jar after it has been generated.
add_custom_command(OUTPUT "${opencv_test_java_bin_dir}/bin/${JAR_NAME}"
                   COMMAND ${CMAKE_COMMAND} -E copy "${JAR_FILE}" "${opencv_test_java_bin_dir}/bin/${JAR_NAME}"
                   DEPENDS "${JAR_FILE}"
                   COMMENT "Copying the OpenCV jar"
                  )

add_custom_command(OUTPUT "${opencv_test_java_bin_dir}/build.xml"
                   COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/build.xml" "${opencv_test_java_bin_dir}/build.xml"
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
