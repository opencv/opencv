set(sample example-image-manipulations)

add_android_project(${sample} "${CMAKE_CURRENT_SOURCE_DIR}" LIBRARY_DEPS ${OpenCV_BINARY_DIR} SDK_TARGET 11 ${ANDROID_SDK_TARGET})
if(TARGET ${sample})
  add_dependencies(opencv_android_examples ${sample})
endif()
