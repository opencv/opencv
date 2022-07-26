set(the_description "Camera Calibration and 3D Reconstruction")

ocv_add_dispatched_file(undistort SSE2 AVX2)

if(ARM AND CV_GCC AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
  # suppress warnings from GCC only on 7.1 and later
  ocv_warnings_disable(CMAKE_CXX_FLAGS -Wno-psabi)
endif()

set(debug_modules "")
if(DEBUG_opencv_calib3d)
  list(APPEND debug_modules opencv_highgui)
endif()
ocv_define_module(calib3d opencv_imgproc opencv_features2d opencv_flann ${debug_modules}
    WRAP java objc python js
)
ocv_target_link_libraries(${the_module} ${LAPACK_LIBRARIES})
