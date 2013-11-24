CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

set(name "facerec")
project(facerec_cpp_samples)

#SET(OpenCV_DIR /path/to/your/opencv/installation)

# packages
find_package(OpenCV REQUIRED) # http://opencv.org

# probably you should loop through the sample files here
add_executable(facerec_demo facerec_demo.cpp)
target_link_libraries(facerec_demo opencv_core opencv_contrib opencv_imgproc opencv_highgui)

add_executable(facerec_video facerec_video.cpp)
target_link_libraries(facerec_video opencv_contrib opencv_core opencv_imgproc opencv_highgui opencv_objdetect opencv_imgproc)

add_executable(facerec_eigenfaces facerec_eigenfaces.cpp)
target_link_libraries(facerec_eigenfaces opencv_contrib opencv_core opencv_imgproc opencv_highgui)

add_executable(facerec_fisherfaces facerec_fisherfaces.cpp)
target_link_libraries(facerec_fisherfaces opencv_contrib opencv_core opencv_imgproc opencv_highgui)

add_executable(facerec_lbph facerec_lbph.cpp)
target_link_libraries(facerec_lbph opencv_contrib opencv_core opencv_imgproc opencv_highgui)
