file(GLOB SRCS *.cpp)
ocv_add_application(opencv_createsamples
    MODULES opencv_core opencv_imgproc opencv_objdetect opencv_imgcodecs opencv_highgui opencv_calib3d opencv_features2d opencv_videoio
    SRCS ${SRCS})
