# ----------------------------------------------------------------------------
#  CMake file for Matlab/Octave support
# ----------------------------------------------------------------------------

# make sure we're on a supported architecture with Matlab and python installed
if (IOS OR ANDROID OR NOT MATLAB_FOUND OR NOT PYTHONLIBS_FOUND)
    ocv_module_disable(matlab)
endif()

set(the_description "The Matlab/Octave bindings")
ocv_add_module(matlab   BINDINGS opencv_core opencv_imgproc 
                        OPTIONAL opencv_objdetect opencv_features2d opencv_video
                                 opencv_highgui opencv_ml opencv_calib3d opencv_photo
                                 opencv_nonfree opencv_calib)

# Add all of the headers we wish to parse
set(opencv_hdrs
    "${OPENCV_MODULE_opencv_core_LOCATION}/include/opencv2/core/core.hpp"
    "${OPENCV_MODULE_opencv_flann_LOCATION}/include/opencv2/flann/miniflann.hpp"
    "${OPENCV_MODULE_opencv_imgproc_LOCATION}/include/opencv2/imgproc/imgproc.hpp"
    "${OPENCV_MODULE_opencv_video_LOCATION}/include/opencv2/video/background_segm.hpp"
    "${OPENCV_MODULE_opencv_video_LOCATION}/include/opencv2/video/tracking.hpp"
    "${OPENCV_MODULE_opencv_photo_LOCATION}/include/opencv2/photo/photo.hpp"
    "${OPENCV_MODULE_opencv_highgui_LOCATION}/include/opencv2/highgui/highgui.hpp"
    "${OPENCV_MODULE_opencv_ml_LOCATION}/include/opencv2/ml/ml.hpp"
    "${OPENCV_MODULE_opencv_features2d_LOCATION}/include/opencv2/features2d/features2d.hpp"
    "${OPENCV_MODULE_opencv_calib3d_LOCATION}/include/opencv2/calib3d/calib3d.hpp"
    "${OPENCV_MODULE_opencv_objdetect_LOCATION}/include/opencv2/objdetect/objdetect.hpp"
    "${OPENCV_MODULE_opencv_softcascade_LOCATION}/include/opencv2/softcascade/softcascade.hpp"
    "${OPENCV_MODULE_opencv_contrib_LOCATION}/include/opencv2/contrib/contrib.hpp")

if(HAVE_opencv_nonfree)
    list(APPEND opencv_hdrs     "${OPENCV_MODULE_opencv_nonfree_LOCATION}/include/opencv2/nonfree/features2d.hpp"
                                "${OPENCV_MODULE_opencv_nonfree_LOCATION}/include/opencv2/nonfree/nonfree.hpp")
    endif()

# add the python generator to the python path
set(PYPATH_CACHE $ENV{PYTHONPATH})
set(ENV{PYTHONPATH} ${OPENCV_MODULE_opencv_python_LOCATION}/src2 $ENV{PYTHONPATH})

# synthesise the matlab sources
execute_process(
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generator/gen_matlab_caller.py 
            ${opencv_hdrs} ${CMAKE_CURRENT_BINARY_DIR}/src)

# compile the matlab sources
file(GLOB SOURCE_FILES ${CMAKE_CURRENT_BINARY_DIR}/src)
foreach(SOURCE_FILE ${SOURCE_FILES})
  # compile the source file using mex
endforeach()



# restore the pythonpath
set(ENV{PYTHONPATH} ${PYPATH_CACHE})
