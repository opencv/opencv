set(DEPS opencv_core opencv_imgproc opencv_features2d opencv_highgui opencv_calib3d opencv_videoio opencv_objdetect)
file(GLOB SRCS *.cpp)
ocv_add_application(opencv_interactive-calibration MODULES ${DEPS} SRCS ${SRCS})
