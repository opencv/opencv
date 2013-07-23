if(ANDROID OR IOS)
  ocv_module_disable(cudastereo)
endif()

set(the_description "CUDA-accelerated Stereo Correspondence")

ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4127 /wd4324 /wd4512 -Wundef -Wmissing-declarations)

ocv_define_module(cudastereo opencv_calib3d)
