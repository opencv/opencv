if(IOS OR WINRT OR (NOT HAVE_CUDA AND NOT BUILD_CUDA_STUBS))
  ocv_module_disable(cudafeatures2d)
endif()

set(the_description "CUDA-accelerated Feature Detection and Description")

ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4127 /wd4100 /wd4324 /wd4512 /wd4515 -Wundef -Wmissing-declarations -Wshadow -Wunused-parameter -Wshadow)

ocv_define_module(cudafeatures2d opencv_features2d opencv_cudafilters opencv_cudawarping)
