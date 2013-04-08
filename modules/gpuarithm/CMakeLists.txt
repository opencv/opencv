if(ANDROID OR IOS)
  ocv_module_disable(gpuarithm)
endif()

set(the_description "GPU-accelerated Operations on Matrices")

ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef -Wmissing-declarations)

ocv_define_module(gpuarithm opencv_core)

if(HAVE_CUBLAS)
  CUDA_ADD_CUBLAS_TO_TARGET(${the_module})
endif()
