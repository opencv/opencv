if(ANDROID OR IOS)
  ocv_module_disable(gpustereo)
endif()

set(the_description "GPU-accelerated Stereo Correspondence")

ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef -Wmissing-declarations)

ocv_define_module(gpustereo opencv_calib3d)
