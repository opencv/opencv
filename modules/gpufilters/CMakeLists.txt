if(ANDROID OR IOS)
  ocv_module_disable(gpufilters)
endif()

set(the_description "GPU-accelerated Image Filtering")

ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef -Wmissing-declarations)

ocv_define_module(gpufilters opencv_imgproc OPTIONAL opencv_gpuarithm)
