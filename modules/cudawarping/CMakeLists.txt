if(ANDROID OR IOS)
  ocv_module_disable(cudawarping)
endif()

set(the_description "CUDA-accelerated Image Warping")

ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4127 /wd4324 /wd4512 -Wundef -Wmissing-declarations)

ocv_define_module(cudawarping opencv_imgproc OPTIONAL opencv_cudalegacy)
