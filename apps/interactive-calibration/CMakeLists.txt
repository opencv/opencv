set(DEPS opencv_core opencv_imgproc opencv_features opencv_highgui opencv_geometry opencv_calib opencv_videoio opencv_objdetect)
file(GLOB SRCS *.cpp)
ocv_add_application(opencv_interactive-calibration MODULES ${DEPS} SRCS ${SRCS})
