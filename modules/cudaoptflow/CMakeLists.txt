if(IOS OR (NOT HAVE_CUDA AND NOT BUILD_CUDA_STUBS))
  ocv_module_disable(cudaoptflow)
endif()

set(the_description "CUDA-accelerated Optical Flow")

ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4127 /wd4324 /wd4512 -Wundef -Wmissing-declarations)

ocv_define_module(cudaoptflow opencv_video opencv_cudaarithm opencv_cudawarping opencv_cudaimgproc OPTIONAL opencv_cudalegacy)
