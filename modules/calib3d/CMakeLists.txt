set(the_description "Camera Calibration and 3D Reconstruction")

ocv_add_dispatched_file(undistort SSE2 AVX2)

set(debug_modules "")
if(DEBUG_opencv_calib3d)
  list(APPEND debug_modules opencv_highgui)
endif()
ocv_define_module(calib3d opencv_imgproc opencv_features2d opencv_flann ${debug_modules}
    WRAP java objc python js
)
