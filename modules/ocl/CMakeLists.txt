if(NOT HAVE_OPENCL)
  ocv_module_disable(ocl)
endif()

set(the_description "OpenCL-accelerated Computer Vision")
ocv_define_module(ocl opencv_core opencv_imgproc opencv_features2d opencv_objdetect opencv_video opencv_calib3d opencv_ml)
ocv_warnings_disable(CMAKE_CXX_FLAGS -Wshadow)
