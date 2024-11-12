set(the_description "Camera Calibration and 3D Reconstruction")

ocv_add_dispatched_file(undistort SSE2 AVX2)

set(debug_modules "")
if(DEBUG_opencv_3d)
  list(APPEND debug_modules opencv_highgui)
endif()
ocv_define_module(3d opencv_imgproc opencv_features opencv_flann ${debug_modules}
    WRAP java objc python js
)
ocv_target_link_libraries(${the_module} ${LAPACK_LIBRARIES})

if(NOT HAVE_EIGEN)
  message(STATUS "3d: Eigen support is disabled. Eigen is Required for Posegraph optimization")
endif()
