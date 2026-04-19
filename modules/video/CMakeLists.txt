set(the_description "Video Analysis")
ocv_add_dispatched_file(lkpyramid SSE2 SSE4_1 AVX2 AVX512_SKX AVX512_ICL)
ocv_define_module(video
    opencv_imgproc
    OPTIONAL
      opencv_3d
      opencv_features
      opencv_dnn
    WRAP
      java
      objc
      python
      js
)

if(HAVE_OPENMP AND DEFINED OpenMP_CXX_LIBRARIES AND OpenMP_CXX_LIBRARIES)
  ocv_target_link_libraries(${the_module} LINK_PRIVATE "${OpenMP_CXX_LIBRARIES}")
endif()
