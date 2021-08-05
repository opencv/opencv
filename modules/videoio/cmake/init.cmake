if(NOT PROJECT_NAME STREQUAL "OpenCV")
  include(FindPkgConfig)
endif()

macro(add_backend backend_id cond_var)
  if(${cond_var})
    include("${CMAKE_CURRENT_LIST_DIR}/detect_${backend_id}.cmake")
  endif()
endmacro()

add_backend("ffmpeg" WITH_FFMPEG)
add_backend("gstreamer" WITH_GSTREAMER)
add_backend("v4l" WITH_V4L)

add_backend("aravis" WITH_ARAVIS)
add_backend("dc1394" WITH_1394)
add_backend("gphoto" WITH_GPHOTO2)
add_backend("msdk" WITH_MFX)
add_backend("openni2" WITH_OPENNI2)
add_backend("pvapi" WITH_PVAPI)
add_backend("realsense" WITH_LIBREALSENSE)
add_backend("ueye" WITH_UEYE)
add_backend("ximea" WITH_XIMEA)
add_backend("xine" WITH_XINE)

add_backend("avfoundation" WITH_AVFOUNDATION)
add_backend("ios" WITH_CAP_IOS)

add_backend("dshow" WITH_DSHOW)
add_backend("msmf" WITH_MSMF)

add_backend("android_mediandk" WITH_ANDROID_MEDIANDK)
add_backend("android_camera" WITH_ANDROID_NATIVE_CAMERA)
