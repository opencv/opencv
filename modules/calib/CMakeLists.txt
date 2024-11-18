set(the_description "Camera Calibration and 3D Reconstruction")

set(debug_modules "")
if(DEBUG_opencv_calib)
  list(APPEND debug_modules opencv_highgui)
endif()
ocv_define_module(calib opencv_imgproc opencv_features opencv_flann opencv_3d opencv_stereo ${debug_modules}
    WRAP java objc python js
)
