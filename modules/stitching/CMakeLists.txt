if(IOS)
    return()
endif()

set(DEPS opencv_core opencv_imgproc opencv_features2d opencv_calib3d opencv_flann opencv_objdetect)
if(NOT ANDROID)
    set(DEPS ${DEPS} opencv_gpu)
endif()

define_opencv_module(stitching ${DEPS})

