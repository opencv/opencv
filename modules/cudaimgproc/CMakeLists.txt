if(IOS)
  ocv_module_disable(cudaimgproc)
endif()

set(the_description "CUDA-accelerated Image Processing")

ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4127 /wd4100 /wd4324 /wd4512 /wd4515 -Wundef -Wmissing-declarations -Wshadow -Wunused-parameter)

ocv_define_module(cudaimgproc opencv_imgproc OPTIONAL opencv_cudaarithm opencv_cudafilters)
