project(opencv_test_android)

set(OPENCV_ANDROID_TEST_DIR "${OpenCV_BINARY_DIR}/android_test" CACHE INTERNAL "")
file(REMOVE_RECURSE "${OPENCV_ANDROID_TEST_DIR}")

set(ANDROID_TESTS_SRC_DIRS "'${CMAKE_CURRENT_SOURCE_DIR}/src', '${CMAKE_CURRENT_SOURCE_DIR}/../common_test/src'" CACHE INTERNAL "")

file(COPY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION ${OpenCV_BINARY_DIR})
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/tests_module/build.gradle.in" "${OPENCV_ANDROID_TEST_DIR}/tests_module/build.gradle" @ONLY)

file(COPY "${OpenCV_SOURCE_DIR}/platforms/android/gradle-wrapper/gradle/wrapper/gradle-wrapper.jar" DESTINATION "${OPENCV_ANDROID_TEST_DIR}/gradle/wrapper")
configure_file("${OpenCV_SOURCE_DIR}/platforms/android/gradle-wrapper/gradle/wrapper/gradle-wrapper.properties.in" "${OPENCV_ANDROID_TEST_DIR}/gradle/wrapper/gradle-wrapper.properties" @ONLY)
