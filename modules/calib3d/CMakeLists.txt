set(the_description "Camera Calibration and 3D Reconstruction")
set(debug_modules "")
if(DEBUG_opencv_calib3d)
  list(APPEND debug_modules opencv_highgui)
endif()
if(ARM AND CV_GCC AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
  # suppress warnings from GCC only on 7.1 and later
  ocv_warnings_disable(CMAKE_CXX_FLAGS -Wno-psabi)
endif()
ocv_define_module(calib3d opencv_imgproc opencv_features2d ${debug_modules}
    WRAP java python js
)
