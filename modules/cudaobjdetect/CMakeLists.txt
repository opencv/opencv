if(IOS OR (NOT HAVE_CUDA AND NOT BUILD_CUDA_STUBS))
  ocv_module_disable(cudaobjdetect)
endif()

set(the_description "CUDA-accelerated Object Detection")

ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4127 /wd4324 /wd4512 -Wundef -Wmissing-declarations -Wshadow)

ocv_define_module(cudaobjdetect opencv_objdetect opencv_cudaarithm opencv_cudawarping OPTIONAL opencv_cudalegacy)
