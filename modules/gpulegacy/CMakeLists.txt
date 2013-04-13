if(NOT HAVE_CUDA)
  ocv_module_disable(gpulegacy)
endif()

set(the_description "GPU-accelerated Computer Vision (legacy)")

ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4127 /wd4130 /wd4324 /wd4512 -Wundef -Wmissing-declarations -Wuninitialized)

ocv_define_module(gpulegacy opencv_core OPTIONAL opencv_objdetect)
