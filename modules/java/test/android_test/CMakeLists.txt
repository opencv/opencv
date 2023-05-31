if(NOT ANT_EXECUTABLE OR NOT ANDROID_EXECUTABLE OR NOT ANDROID_TOOLS_Pkg_Revision GREATER 13)
  return()
endif()

project(opencv_test_java)

set(OPENCV_JAVA_TEST_DIR "${OpenCV_BINARY_DIR}/android_test" CACHE INTERNAL "")
file(REMOVE_RECURSE "${OPENCV_JAVA_TEST_DIR}")
file(REMOVE "${OPENCV_DEPHELPER}/${the_module}_test_source_copy")

set(test_dir "${CMAKE_CURRENT_SOURCE_DIR}")

set(depends "")

# 1. gather and copy common test files (resources, utils, etc.)
copy_common_tests("${CMAKE_CURRENT_SOURCE_DIR}/../common_test" "${OPENCV_JAVA_TEST_DIR}" depends)

# 2. gather and copy tests from each module
ocv_copyfiles_append_dir(JAVA_TEST_SRC_COPY "${OPENCV_JAVA_BINDINGS_DIR}/gen/test" "${OPENCV_JAVA_TEST_DIR}/src")

list(APPEND depends gen_opencv_java_source "${OPENCV_DEPHELPER}/gen_opencv_java_source")
ocv_copyfiles_add_target(${the_module}_test_source_copy JAVA_TEST_SRC_COPY "Copy Java(Android test) source files" ${depends})
set(depends ${the_module}_test_source_copy "${OPENCV_DEPHELPER}/${the_module}_test_source_copy")

# 3. gather and copy specific files for Android
file(GLOB_RECURSE test_files RELATIVE "${test_dir}" "${test_dir}/res/*" "${test_dir}/src/*")
foreach(f ${test_files} ${ANDROID_MANIFEST_FILE} ".classpath" ".project")
  add_custom_command(
      OUTPUT "${OPENCV_JAVA_TEST_DIR}/${f}"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${test_dir}/${f}" "${OPENCV_JAVA_TEST_DIR}/${f}"
      MAIN_DEPENDENCY "${test_dir}/${f}"
      COMMENT "Copying ${f}")
  list(APPEND depends "${test_dir}/${f}" "${OPENCV_JAVA_TEST_DIR}/${f}")
endforeach()

# fix Android project
set(android_proj_target_files ${ANDROID_PROJECT_FILES})
ocv_list_add_prefix(android_proj_target_files "${OPENCV_JAVA_TEST_DIR}/")
file(RELATIVE_PATH __dep "${OPENCV_JAVA_TEST_DIR}" "${OpenCV_BINARY_DIR}/android_sdk")

add_custom_command(
    OUTPUT ${android_proj_target_files}
    COMMAND ${CMAKE_COMMAND} -E remove ${android_proj_target_files}
    COMMAND ${ANDROID_EXECUTABLE} --silent update test-project --path "${OPENCV_JAVA_TEST_DIR}" --main "${OpenCV_BINARY_DIR}/android_sdk"
    COMMAND ${ANDROID_EXECUTABLE} --silent update project --path "${OPENCV_JAVA_TEST_DIR}" --library "${__dep}"
    MAIN_DEPENDENCY "${OPENCV_JAVA_TEST_DIR}/${ANDROID_MANIFEST_FILE}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${ANDROID_MANIFEST_FILE}"
    COMMENT "Updating Android Java API test project")

list(APPEND depends ${android_proj_target_files})

# build java part
add_custom_command(
    OUTPUT "${OPENCV_JAVA_TEST_DIR}/bin/OpenCVTest-debug.apk"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:opencv_java>" "${OPENCV_JAVA_TEST_DIR}/libs/${ANDROID_NDK_ABI_NAME}/$<TARGET_FILE_NAME:opencv_java>"
    COMMAND ${ANT_EXECUTABLE} -q -noinput -k debug -Djava.target=1.6 -Djava.source=1.6
    COMMAND ${CMAKE_COMMAND} -E touch "${OPENCV_JAVA_TEST_DIR}/bin/OpenCVTest-debug.apk" # needed because ant does not update the timestamp of updated apk
    WORKING_DIRECTORY "${OPENCV_JAVA_TEST_DIR}"
    MAIN_DEPENDENCY "${OPENCV_JAVA_TEST_DIR}/${ANDROID_MANIFEST_FILE}"
    DEPENDS opencv_java_android opencv_java
    DEPENDS ${depends})

add_custom_target(${PROJECT_NAME} ALL SOURCES "${OPENCV_JAVA_TEST_DIR}/bin/OpenCVTest-debug.apk" "${CMAKE_CURRENT_SOURCE_DIR}/${ANDROID_MANIFEST_FILE}")
add_dependencies(${PROJECT_NAME} opencv_java ${__android_project_chain})
set(__android_project_chain ${PROJECT_NAME} CACHE INTERNAL "auxiliary variable used for Android progects chaining" FORCE)

# put the final .apk to the OpenCV's bin folder
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different "${OPENCV_JAVA_TEST_DIR}/bin/OpenCVTest-debug.apk" "${OpenCV_BINARY_DIR}/bin/${PROJECT_NAME}.apk")

add_dependencies(opencv_tests ${PROJECT_NAME})

if(PYTHON_DEFAULT_AVAILABLE)
  set(CHECK_TEST_COVERAGE "${OPENCV_MODULE_opencv_java_LOCATION}/check-tests.py")
  add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
      COMMAND ${PYTHON_DEFAULT_EXECUTABLE} ${CHECK_TEST_COVERAGE} "${OPENCV_JAVA_TEST_DIR}/src" "${OPENCV_ANDROID_LIB_DIR}/src" > "${CMAKE_CURRENT_BINARY_DIR}/tests_coverage.log"
      )
endif()
