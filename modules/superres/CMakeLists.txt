if(IOS OR WINRT)
  ocv_module_disable(superres)
endif()

set(the_description "Super Resolution")
if(HAVE_CUDA)
  ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef -Wshadow)
endif()
ocv_define_module(superres opencv_imgproc opencv_video
                  OPTIONAL opencv_videoio opencv_cudaarithm opencv_cudafilters opencv_cudawarping opencv_cudaimgproc opencv_cudaoptflow opencv_cudacodec
                  WRAP python)
