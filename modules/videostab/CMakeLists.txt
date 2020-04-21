set(the_description "Video stabilization")

if(HAVE_CUDA)
  ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef -Wmissing-declarations -Wshadow -Wunused-parameter)
endif()

if(DEFINED WINRT AND NOT DEFINED ENABLE_WINRT_MODE_NATIVE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /ZW")
endif()

ocv_define_module(videostab opencv_imgproc opencv_features2d opencv_video opencv_photo opencv_calib3d
OPTIONAL opencv_cudawarping opencv_cudaoptflow opencv_videoio WRAP python)
