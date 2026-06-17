set(the_description "SLAM and Visual Odometry")

ocv_define_module(slam
  opencv_geometry
  opencv_features
  opencv_imgproc
  opencv_imgcodecs
  opencv_video
  OPTIONAL
  opencv_dnn
  WRAP python)

ocv_warnings_disable(CMAKE_CXX_FLAGS -Wshadow)
