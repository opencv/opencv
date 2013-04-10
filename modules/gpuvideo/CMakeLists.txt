if(ANDROID OR IOS)
  ocv_module_disable(gpuvideo)
endif()

set(the_description "GPU-accelerated Video Analysis")

ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef -Wmissing-declarations)

ocv_define_module(gpuvideo opencv_video opencv_legacy opencv_gpufilters opencv_gpuimgproc OPTIONAL opencv_gpulegacy)
