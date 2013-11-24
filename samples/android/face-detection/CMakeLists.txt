set(sample example-face-detection)

if(BUILD_FAT_JAVA_LIB)
  set(native_deps opencv_java)
else()
  set(native_deps opencv_contrib)
endif()

add_android_project(${sample} "${CMAKE_CURRENT_SOURCE_DIR}" LIBRARY_DEPS ${OpenCV_BINARY_DIR} SDK_TARGET 11 ${ANDROID_SDK_TARGET} NATIVE_DEPS ${native_deps})
if(TARGET ${sample})
  add_dependencies(opencv_android_examples ${sample})
endif()
