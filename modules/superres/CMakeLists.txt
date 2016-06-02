if(IOS OR WINRT)
  ocv_module_disable(superres)
endif()

set(the_description "Super Resolution")
ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4127 -Wundef -Wshadow)
ocv_define_module(superres opencv_imgproc opencv_video
                  OPTIONAL opencv_videoio opencv_cudaarithm opencv_cudafilters opencv_cudawarping opencv_cudaimgproc opencv_cudaoptflow opencv_cudacodec
                  WRAP python)
