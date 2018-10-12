ocv_warnings_disable(CMAKE_CXX_FLAGS -Woverloaded-virtual -Winconsistent-missing-override -Wsuggest-override)
file(GLOB SRCS *.cpp)
ocv_add_application(opencv_traincascade
    MODULES opencv_core opencv_imgproc opencv_objdetect opencv_imgcodecs opencv_highgui opencv_calib3d opencv_features2d
    SRCS ${SRCS})
