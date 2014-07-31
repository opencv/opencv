if(IOS OR (NOT HAVE_CUDA AND NOT BUILD_CUDA_STUBS))
  ocv_module_disable(cudafilters)
endif()

set(the_description "CUDA-accelerated Image Filtering")

ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4127 /wd4324 /wd4512 -Wundef -Wmissing-declarations)

ocv_define_module(cudafilters opencv_imgproc opencv_cudaarithm)
