set(the_description "Depth from Stereo")

set(debug_modules "")
if(DEBUG_opencv_stereo)
  list(APPEND debug_modules opencv_highgui)
endif()
ocv_define_module(stereo opencv_imgproc opencv_3d ${debug_modules}
    WRAP java objc python js
)
