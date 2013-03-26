if(NOT HAVE_OPENCL)
  ocv_module_disable(ocl)
endif()

set(the_description "OpenCL-accelerated Computer Vision")
ocv_define_module(ocl opencv_core opencv_imgproc opencv_objdetect opencv_video)
ocv_warnings_disable(CMAKE_CXX_FLAGS -Wshadow)
