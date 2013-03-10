# ----------------------------------------------------------------------------
#  CMake file for Matlab/Octave support
# ----------------------------------------------------------------------------

# make sure we're on a supported architecture with Matlab installed
if (IOS OR ANDROID OR NOT MATLAB_FOUND)
    ocv_module_disable(matlab)
endif()

set(the_description "The Matlab/Octave bindings")
ocv_add_module(matlab   BINDINGS opencv_core opencv_imgproc 
                        OPTIONAL opencv_objdetect opencv_features2d opencv_video
                                 opencv_highgui opencv_ml opencv_calib3d opencv_photo
                                 opencv_nonfree opencv_calib)

add_subdirectory(test)
