set(sample example-tutorial-4-cuda)

ocv_check_dependencies(opencv_core opencv_java opencv_gpu)

if (OCV_DEPENDENCIES_FOUND)
  if(BUILD_FAT_JAVA_LIB)
    set(native_deps opencv_java opencv_gpu)
  else()
    set(native_deps opencv_gpu)
  endif()

  add_android_project(${sample} "${CMAKE_CURRENT_SOURCE_DIR}" LIBRARY_DEPS ${OpenCV_BINARY_DIR} SDK_TARGET 11 ${ANDROID_SDK_TARGET} NATIVE_DEPS ${native_deps})
  if(TARGET ${sample})
    add_dependencies(opencv_android_examples ${sample})
  endif()
endif()
