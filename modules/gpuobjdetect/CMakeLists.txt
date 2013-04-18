if(ANDROID OR IOS)
  ocv_module_disable(gpuobjdetect)
endif()

set(the_description "GPU-accelerated Object Detection")

ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef -Wmissing-declarations)

ocv_define_module(gpuobjdetect opencv_objdetect opencv_gpuimgproc OPTIONAL opencv_gpulegacy)
