if(ANDROID OR IOS)
  ocv_module_disable(cuda)
endif()

set(the_description "CUDA-accelerated Computer Vision")

ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4127 /wd4100 /wd4324 /wd4512 /wd4515 -Wundef -Wmissing-declarations -Wshadow -Wunused-parameter)

ocv_define_module(cuda opencv_calib3d opencv_objdetect opencv_cudaarithm opencv_cudawarping OPTIONAL opencv_cudalegacy)
