if(ANDROID OR IOS)
  ocv_module_disable(gpubgsegm)
endif()

set(the_description "GPU-accelerated Background Segmentation")

ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4127 /wd4324 /wd4512 -Wundef -Wmissing-declarations)

ocv_define_module(gpubgsegm opencv_video OPTIONAL opencv_legacy opencv_imgproc opencv_gpuarithm opencv_gpufilters opencv_gpuimgproc)
