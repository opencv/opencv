ocv_check_dependencies(opencv_java ${OPENCV_MODULE_opencv_java_OPT_DEPS} ${OPENCV_MODULE_opencv_java_REQ_DEPS})

if(NOT OCV_DEPENDENCIES_FOUND OR NOT ANT_EXECUTABLE OR NOT ANDROID_EXECUTABLE OR NOT ANDROID_TOOLS_Pkg_Revision GREATER 13)
  return()
endif()

project(opencv_test_java)

set(opencv_test_java_bin_dir "${CMAKE_CURRENT_BINARY_DIR}/.build")

# get project sources
file(GLOB_RECURSE opencv_test_java_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/res/*" "${CMAKE_CURRENT_SOURCE_DIR}/src/*")
ocv_list_filterout(opencv_test_java_files ".svn")

# copy sources out from the build tree
set(opencv_test_java_file_deps "")
foreach(f ${opencv_test_java_files} ${ANDROID_MANIFEST_FILE} ".classpath" ".project")
  add_custom_command(
      OUTPUT "${opencv_test_java_bin_dir}/${f}"
      COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/${f}" "${opencv_test_java_bin_dir}/${f}"
      MAIN_DEPENDENCY "${CMAKE_CURRENT_SOURCE_DIR}/${f}"
      COMMENT "Copying ${f}")
  list(APPEND opencv_test_java_file_deps "${CMAKE_CURRENT_SOURCE_DIR}/${f}" "${opencv_test_java_bin_dir}/${f}")
endforeach()

# fix Android project
set(android_proj_target_files ${ANDROID_PROJECT_FILES})
ocv_list_add_prefix(android_proj_target_files "${opencv_test_java_bin_dir}/")
file(RELATIVE_PATH __dep "${opencv_test_java_bin_dir}" "${OpenCV_BINARY_DIR}")

add_custom_command(
    OUTPUT ${android_proj_target_files}
    COMMAND ${CMAKE_COMMAND} -E remove ${android_proj_target_files}
    COMMAND ${ANDROID_EXECUTABLE} --silent update test-project --path "${opencv_test_java_bin_dir}" --main "${OpenCV_BINARY_DIR}"
    COMMAND ${ANDROID_EXECUTABLE} --silent update project --path "${opencv_test_java_bin_dir}" --library "${__dep}"
    MAIN_DEPENDENCY "${opencv_test_java_bin_dir}/${ANDROID_MANIFEST_FILE}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${ANDROID_MANIFEST_FILE}"
    COMMENT "Updating Android Java API test project")

list(APPEND opencv_test_java_file_deps ${android_proj_target_files})

# build java part
add_custom_command(
    OUTPUT "${opencv_test_java_bin_dir}/bin/OpenCVTest-debug.apk"
    COMMAND ${CMAKE_COMMAND} -E copy "${OpenCV_BINARY_DIR}/lib/${ANDROID_NDK_ABI_NAME}/libopencv_java.so" "${opencv_test_java_bin_dir}/libs/${ANDROID_NDK_ABI_NAME}/libopencv_java.so"
    COMMAND ${ANT_EXECUTABLE} -q -noinput -k debug
    COMMAND ${CMAKE_COMMAND} -E touch "${opencv_test_java_bin_dir}/bin/OpenCVTest-debug.apk" # needed because ant does not update the timestamp of updated apk
    WORKING_DIRECTORY "${opencv_test_java_bin_dir}"
    MAIN_DEPENDENCY "${opencv_test_java_bin_dir}/${ANDROID_MANIFEST_FILE}"
    DEPENDS "${JAR_FILE}.dephelper" opencv_java
    DEPENDS ${opencv_test_java_file_deps})

add_custom_target(${PROJECT_NAME} ALL SOURCES "${opencv_test_java_bin_dir}/bin/OpenCVTest-debug.apk" )
add_dependencies(${PROJECT_NAME} opencv_java ${__android_project_chain})
set(__android_project_chain ${PROJECT_NAME} CACHE INTERNAL "auxiliary variable used for Android progects chaining" FORCE)

# put the final .apk to the OpenCV's bin folder
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${opencv_test_java_bin_dir}/bin/OpenCVTest-debug.apk" "${OpenCV_BINARY_DIR}/bin/${PROJECT_NAME}.apk")

add_dependencies(opencv_tests ${PROJECT_NAME})

if(PYTHON_EXECUTABLE)
  set(CHECK_TEST_COVERAGE "${OPENCV_MODULE_opencv_java_LOCATION}/check-tests.py")
  add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
      COMMAND ${PYTHON_EXECUTABLE} ${CHECK_TEST_COVERAGE} "${CMAKE_CURRENT_SOURCE_DIR}/src" "${OpenCV_BINARY_DIR}/src" > "${CMAKE_CURRENT_BINARY_DIR}/tests_coverage.log"
      )
endif()
