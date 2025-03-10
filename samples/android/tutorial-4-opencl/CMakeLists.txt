set(sample example-tutorial-4-opencl)

if(BUILD_FAT_JAVA_LIB)
  set(native_deps opencv_java)
else()
  set(native_deps opencv_imgproc)
endif()

add_android_project(${sample} "${CMAKE_CURRENT_SOURCE_DIR}"
    LIBRARY_DEPS "${OPENCV_ANDROID_LIB_DIR}"
    SDK_TARGET 21 "${ANDROID_SDK_TARGET}"
    NATIVE_DEPS ${native_deps} -lGLESv2 -lEGL
    COPY_LIBS YES
)
if(TARGET ${sample})
  add_dependencies(opencv_android_examples ${sample})
endif()
