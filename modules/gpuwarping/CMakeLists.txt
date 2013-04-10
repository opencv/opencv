if(ANDROID OR IOS)
  ocv_module_disable(gpuwarping)
endif()

set(the_description "GPU-accelerated Image Warping")

ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef -Wmissing-declarations)

ocv_define_module(gpuwarping opencv_imgproc opencv_gpuarithm)
