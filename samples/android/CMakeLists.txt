# ----------------------------------------------------------------------------
#  CMake file for Android samples. See root CMakeLists.txt
#
# ----------------------------------------------------------------------------
add_custom_target(opencv_android_examples)

ocv_warnings_disable(CMAKE_CXX_FLAGS -Wmissing-declarations)

add_subdirectory(15-puzzle)
add_subdirectory(face-detection)
add_subdirectory(image-manipulations)
add_subdirectory(camera-calibration)
add_subdirectory(color-blob-detection)
add_subdirectory(tutorial-1-camerapreview)
add_subdirectory(tutorial-2-mixedprocessing)
add_subdirectory(tutorial-3-cameracontrol)

if (HAVE_opencv_gpu)
  add_subdirectory(tutorial-4-cuda)
endif()

add_subdirectory(native-activity)

# hello-android sample
if(HAVE_opencv_highgui)
  ocv_include_modules_recurse(opencv_highgui opencv_core)
  add_executable(hello-android hello-android/main.cpp)
  target_link_libraries(hello-android ${OPENCV_LINKER_LIBS} opencv_highgui opencv_core)
  set_target_properties(hello-android PROPERTIES OUTPUT_NAME hello-android RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}")
  add_dependencies(opencv_android_examples hello-android)
endif()
