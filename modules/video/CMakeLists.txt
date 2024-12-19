set(the_description "Video Analysis")
ocv_define_module(video
    opencv_imgproc
    OPTIONAL
      opencv_calib3d
      opencv_dnn
    WRAP
      java
      objc
      python
      js
)

ocv_target_link_libraries(${the_module} PRIVATE lapjv)

if(HAVE_OPENMP AND DEFINED OpenMP_CXX_LIBRARIES AND OpenMP_CXX_LIBRARIES)
  ocv_target_link_libraries(${the_module} LINK_PRIVATE "${OpenMP_CXX_LIBRARIES}")
endif()
