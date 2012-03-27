set(sample example-tutorial-3-native)

add_android_project(${sample} "${CMAKE_CURRENT_SOURCE_DIR}" SDK_TARGET 11 ${ANDROID_SDK_TARGET} NATIVE_DEPS opencv_features2d)
if(TARGET ${sample})
  add_dependencies(opencv_android_examples ${sample})
endif()
