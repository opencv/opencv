if(NOT HAVE_CUDA)
  ocv_module_disable(cudalegacy)
endif()

set(the_description "CUDA-accelerated Computer Vision (legacy)")

ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4127 /wd4130 /wd4324 /wd4512 /wd4310 -Wundef -Wmissing-declarations -Wuninitialized -Wshadow -Wdeprecated-declarations -Wstrict-aliasing -Wtautological-compare)

ocv_define_module(cudalegacy opencv_core opencv_video
  OPTIONAL opencv_objdetect opencv_imgproc opencv_calib3d opencv_cudaarithm opencv_cudafilters opencv_cudaimgproc)
