if(ANDROID OR IOS)
  ocv_module_disable(gpuimgproc)
endif()

set(the_description "GPU-accelerated Image Processing")

ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4127 /wd4100 /wd4324 /wd4512 -Wundef -Wmissing-declarations -Wshadow -Wunused-parameter)

ocv_define_module(gpuimgproc opencv_imgproc opencv_gpufilters OPTIONAL opencv_gpuarithm)
