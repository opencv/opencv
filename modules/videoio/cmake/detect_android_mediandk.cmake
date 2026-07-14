# if(ANDROID AND ANDROID_NATIVE_API_LEVEL GREATER_EQUAL 21)  <-- would be nicer but requires CMake 3.7 or later
if(ANDROID AND ANDROID_NATIVE_API_LEVEL GREATER 20)
  set(HAVE_ANDROID_MEDIANDK TRUE)
  set(libs "-landroid -llog -lmediandk")
  ocv_add_external_target(android_mediandk "" "${libs}" "HAVE_ANDROID_MEDIANDK")
endif()
