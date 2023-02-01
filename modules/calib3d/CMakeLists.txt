set(the_description "Camera Calibration and 3D Reconstruction")
set(debug_modules "")
if(DEBUG_opencv_calib3d)
  list(APPEND debug_modules opencv_highgui)
endif()
ocv_define_module(calib3d opencv_imgproc opencv_features2d ${debug_modules}
    WRAP java python js
)
