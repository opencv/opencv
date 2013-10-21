if(BUILD_ANDROID_PACKAGE)
  ocv_module_disable(nonfree)
endif()

set(the_description "Functionality with possible limitations on the use")
ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef)
ocv_define_module(nonfree opencv_imgproc opencv_features2d opencv_calib3d OPTIONAL opencv_cudaarithm opencv_ocl)
