# ----------------------------------------------------------------------------
#  CMake file for Android samples. See root CMakeLists.txt
#
# ----------------------------------------------------------------------------
add_custom_target(opencv_android_examples)

ocv_warnings_disable(CMAKE_CXX_FLAGS -Wmissing-declarations)

add_subdirectory(15-puzzle)
add_subdirectory(face-detection)
add_subdirectory(qr-detection)
add_subdirectory(image-manipulations)
add_subdirectory(camera-calibration)
add_subdirectory(color-blob-detection)
add_subdirectory(mobilenet-objdetect)
add_subdirectory(video-recorder)
add_subdirectory(tutorial-1-camerapreview)
add_subdirectory(tutorial-2-mixedprocessing)
add_subdirectory(tutorial-3-cameracontrol)
add_subdirectory(tutorial-4-opencl)
