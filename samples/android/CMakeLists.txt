# ----------------------------------------------------------------------------
#  CMake file for Android samples. See root CMakeLists.txt
#
# ----------------------------------------------------------------------------
add_custom_target(opencv_android_examples)

add_subdirectory(15-puzzle)
add_subdirectory(face-detection)
add_subdirectory(image-manipulations)
add_subdirectory(color-blob-detection)

add_subdirectory(tutorial-0-androidcamera)
add_subdirectory(tutorial-1-addopencv)
add_subdirectory(tutorial-2-opencvcamera)
add_subdirectory(tutorial-3-native)
add_subdirectory(tutorial-4-mixed)

#hello-android sample
if(HAVE_opencv_highgui)
  ocv_include_modules_recurse(opencv_highgui opencv_core)
  add_executable(hello-android hello-android/main.cpp)
  target_link_libraries(hello-android ${OPENCV_LINKER_LIBS} opencv_highgui opencv_core)
  set_target_properties(hello-android PROPERTIES OUTPUT_NAME hello-android RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}")
  add_dependencies(opencv_android_examples hello-android)
endif()
