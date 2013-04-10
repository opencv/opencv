if(ANDROID OR IOS)
  ocv_module_disable(gpuimgproc)
endif()

set(the_description "GPU-accelerated Image Processing")

ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef -Wmissing-declarations -Wshadow -Wunused-parameter)

ocv_define_module(gpuimgproc opencv_imgproc opencv_gpuarithm opencv_gpufilters)
