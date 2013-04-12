if(ANDROID OR IOS)
  ocv_module_disable(gpufeatures2d)
endif()

set(the_description "GPU-accelerated Feature Detection and Description")

ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef -Wmissing-declarations -Wshadow -Wunused-parameter)

ocv_define_module(gpufeatures2d opencv_features2d opencv_gpufilters opencv_gpuwarping)
