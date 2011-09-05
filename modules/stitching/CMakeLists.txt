include_directories("${OpenCV_SOURCE_DIR}/modules/imgproc/src") # For gcgraph.hpp
define_opencv_module(stitching opencv_core opencv_imgproc opencv_features2d opencv_calib3d opencv_gpu opencv_flann opencv_objdetect)
