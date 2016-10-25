if(NOT ANT_EXECUTABLE OR NOT ANDROID_EXECUTABLE OR NOT ANDROID_TOOLS_Pkg_Revision GREATER 13)
  return()
endif()

project(opencv_test_java)

set(opencv_test_java_bin_dir "${CMAKE_CURRENT_BINARY_DIR}/.build")
set(test_dir "${CMAKE_CURRENT_SOURCE_DIR}")
set(test_common_dir "${CMAKE_CURRENT_SOURCE_DIR}/../common_test")

set(opencv_test_java_file_deps "")

# 1. gather and copy common test files (resources, utils, etc.)
copy_common_tests(test_common_dir opencv_test_java_bin_dir opencv_test_java_file_deps)

# 2. gather and copy tests from each module
copy_modules_tests(OPENCV_JAVA_MODULES opencv_test_java_bin_dir opencv_test_java_file_deps)

# 3. gather and copy specific files for Android
file(GLOB_RECURSE test_files RELATIVE "${test_dir}" "${test_dir}/res/*" "${test_dir}/src/*")
foreach(f ${test_files} ${ANDROID_MANIFEST_FILE} ".classpath" ".project")
  add_custom_command(
      OUTPUT "${opencv_test_java_bin_dir}/${f}"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${test_dir}/${f}" "${opencv_test_java_bin_dir}/${f}"
      MAIN_DEPENDENCY "${test_dir}/${f}"
      COMMENT "Copying ${f}")
  list(APPEND opencv_test_java_file_deps "${test_dir}/${f}" "${opencv_test_java_bin_dir}/${f}")
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

get_target_property(java_location opencv_java LOCATION)
get_filename_component(java_name "${java_location}" NAME)

# build java part
add_custom_command(
    OUTPUT "${opencv_test_java_bin_dir}/bin/OpenCVTest-debug.apk"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${java_location}" "${opencv_test_java_bin_dir}/libs/${ANDROID_NDK_ABI_NAME}/${java_name}"
    COMMAND ${ANT_EXECUTABLE} -q -noinput -k debug -Djava.target=1.6 -Djava.source=1.6
    COMMAND ${CMAKE_COMMAND} -E touch "${opencv_test_java_bin_dir}/bin/OpenCVTest-debug.apk" # needed because ant does not update the timestamp of updated apk
    WORKING_DIRECTORY "${opencv_test_java_bin_dir}"
    MAIN_DEPENDENCY "${opencv_test_java_bin_dir}/${ANDROID_MANIFEST_FILE}"
    DEPENDS "${JAR_FILE}.dephelper" opencv_java
    DEPENDS ${opencv_test_java_file_deps})

add_custom_target(${PROJECT_NAME} ALL SOURCES "${opencv_test_java_bin_dir}/bin/OpenCVTest-debug.apk" )
add_dependencies(${PROJECT_NAME} opencv_java ${__android_project_chain})
set(__android_project_chain ${PROJECT_NAME} CACHE INTERNAL "auxiliary variable used for Android progects chaining" FORCE)

# put the final .apk to the OpenCV's bin folder
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different "${opencv_test_java_bin_dir}/bin/OpenCVTest-debug.apk" "${OpenCV_BINARY_DIR}/bin/${PROJECT_NAME}.apk")

add_dependencies(opencv_tests ${PROJECT_NAME})

if(PYTHON_DEFAULT_AVAILABLE)
  set(CHECK_TEST_COVERAGE "${OPENCV_MODULE_opencv_java_LOCATION}/check-tests.py")
  add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
      COMMAND ${PYTHON_DEFAULT_EXECUTABLE} ${CHECK_TEST_COVERAGE} "${CMAKE_CURRENT_SOURCE_DIR}/src" "${OpenCV_BINARY_DIR}/src" > "${CMAKE_CURRENT_BINARY_DIR}/tests_coverage.log"
      )
endif()
