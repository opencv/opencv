if(ANDROID OR IOS)
  ocv_module_disable(gpubgsegm)
endif()

set(the_description "GPU-accelerated Background Segmentation")

ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef -Wmissing-declarations)

ocv_define_module(gpubgsegm opencv_video opencv_legacy opencv_gpufilters opencv_gpuimgproc)
