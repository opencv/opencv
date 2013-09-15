if(ANDROID OR IOS)
  ocv_module_disable(superres)
endif()

set(the_description "Super Resolution")
ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4127 -Wundef)
ocv_define_module(superres opencv_imgproc opencv_video
                  OPTIONAL opencv_highgui opencv_ocl
                  opencv_gpuarithm opencv_gpufilters opencv_gpuwarping opencv_gpuimgproc opencv_gpuoptflow opencv_gpucodec)
