if(ANDROID OR IOS)
  ocv_module_disable(gpucalib3d)
endif()

set(the_description "GPU-accelerated Camera Calibration and 3D Reconstruction")

ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef -Wmissing-declarations)

ocv_define_module(gpucalib3d opencv_calib3d)
