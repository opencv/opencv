if(NOT HAVE_CUDA)
  ocv_module_disable(gpunvidia)
endif()

set(the_description "GPU-accelerated Computer Vision (HAL module)")

ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef -Wmissing-declarations)

ocv_define_module(gpunvidia opencv_core opencv_objdetect)
