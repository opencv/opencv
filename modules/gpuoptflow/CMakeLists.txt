if(ANDROID OR IOS)
  ocv_module_disable(gpuoptflow)
endif()

set(the_description "GPU-accelerated Optical Flow")

ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef -Wmissing-declarations)

ocv_define_module(gpuoptflow opencv_video opencv_legacy opencv_gpuarithm opencv_gpuwarping opencv_gpuimgproc OPTIONAL opencv_gpulegacy)
