# if(ANDROID AND ANDROID_NATIVE_API_LEVEL GREATER_EQUAL 24)  <-- would be nicer but requires CMake 3.7 or later
if(ANDROID AND ANDROID_NATIVE_API_LEVEL GREATER 23)
  set(HAVE_ANDROID_NATIVE_CAMERA TRUE)
  set(libs "-landroid -llog -lcamera2ndk")
  ocv_add_external_target(android_native_camera "" "${libs}" "HAVE_ANDROID_NATIVE_CAMERA")
endif()
