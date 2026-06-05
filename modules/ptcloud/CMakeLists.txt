set(the_description "High level point cloud and mesh operations")

set(debug_modules "")
if(DEBUG_opencv_ptcloud)
  list(APPEND debug_modules opencv_highgui)
endif()
ocv_define_module(ptcloud opencv_geometry opencv_imgproc opencv_flann ${debug_modules}
    WRAP java objc python js
)
ocv_target_link_libraries(${the_module} ${LAPACK_LIBRARIES})

if(NOT HAVE_EIGEN)
  message(STATUS "Geometry: Eigen support is disabled. Eigen is Required for Posegraph optimization")
endif()
