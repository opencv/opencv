# ----------------------------------------------------------------------------
#  CMake file for python support
# ----------------------------------------------------------------------------

if((WIN32 AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    OR BUILD_opencv_world
    )
<<<<<<< HEAD
  ocv_module_disable(python2)
  ocv_module_disable(python3)
=======

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
    "${OPENCV_MODULE_opencv_contrib_LOCATION}/include/opencv2/contrib/contrib.hpp")

if(HAVE_opencv_nonfree)
  list(APPEND opencv_hdrs     "${OPENCV_MODULE_opencv_nonfree_LOCATION}/include/opencv2/nonfree/features2d.hpp"
                              "${OPENCV_MODULE_opencv_nonfree_LOCATION}/include/opencv2/nonfree/nonfree.hpp")
>>>>>>> a28cde9c3bf69e7839971c29900fbbd4963998bd
endif()

if(ANDROID OR IOS)
  ocv_module_disable(python2)
  ocv_module_disable(python3)
endif()

add_subdirectory(python2)
add_subdirectory(python3)
