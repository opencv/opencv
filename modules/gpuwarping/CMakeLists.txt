if(ANDROID OR IOS)
  ocv_module_disable(gpuwarping)
endif()

set(the_description "GPU-accelerated Image Warping")

ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4127 /wd4324 /wd4512 -Wundef -Wmissing-declarations)

ocv_define_module(gpuwarping opencv_imgproc OPTIONAL opencv_gpulegacy)
