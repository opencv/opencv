project(opencv_test_android)

set(OPENCV_ANDROID_TEST_DIR "${OpenCV_BINARY_DIR}/android_test" CACHE INTERNAL "")
file(REMOVE_RECURSE "${OPENCV_ANDROID_TEST_DIR}")

set(ANDROID_TESTS_SRC_DIRS
"'${CMAKE_CURRENT_SOURCE_DIR}/src', \
'${OpenCV_SOURCE_DIR}/modules/java/test/common_test/src', \
'${CMAKE_BINARY_DIR}/modules/java_bindings_generator/gen/test'" CACHE INTERNAL "")

set(ANDROID_TESTS_RES_DIR "'${OpenCV_SOURCE_DIR}/modules/java/test/common_test/res'" CACHE INTERNAL "")

list(APPEND TEST_PROJECT_FILES "CMakeLists.txt" "gradle.properties" "settings.gradle")
foreach(TEST_PROJECT_FILE ${TEST_PROJECT_FILES})
    file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/${TEST_PROJECT_FILE}" DESTINATION "${OPENCV_ANDROID_TEST_DIR}")
endforeach()
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/tests_module/AndroidManifest.xml" DESTINATION "${OPENCV_ANDROID_TEST_DIR}/tests_module")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/tests_module/build.gradle.in" "${OPENCV_ANDROID_TEST_DIR}/tests_module/build.gradle" @ONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/build.gradle.in" "${OPENCV_ANDROID_TEST_DIR}/build.gradle" @ONLY)

file(COPY "${OpenCV_SOURCE_DIR}/platforms/android/gradle-wrapper/gradlew" DESTINATION "${OPENCV_ANDROID_TEST_DIR}")
file(COPY "${OpenCV_SOURCE_DIR}/platforms/android/gradle-wrapper/gradlew.bat" DESTINATION "${OPENCV_ANDROID_TEST_DIR}")
file(COPY "${OpenCV_SOURCE_DIR}/platforms/android/gradle-wrapper/gradle/wrapper/gradle-wrapper.jar" DESTINATION "${OPENCV_ANDROID_TEST_DIR}/gradle/wrapper")

configure_file("${OpenCV_SOURCE_DIR}/platforms/android/gradle-wrapper/gradle/wrapper/gradle-wrapper.properties.in" "${OPENCV_ANDROID_TEST_DIR}/gradle/wrapper/gradle-wrapper.properties" @ONLY)
