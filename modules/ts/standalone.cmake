find_package(OpenCV REQUIRED opencv_core opencv_imgproc opencv_imgcodecs opencv_videoio opencv_highgui)
if(CMAKE_VERSION VERSION_LESS "2.8.11")
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()

message(STATUS "OpenCV INCLUDES: ${OpenCV_INCLUDE_DIRS}")

file(GLOB SRCS src/*.cpp)
add_library(opencv_ts SHARED ${SRCS})
target_include_directories(opencv_ts PRIVATE include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(opencv_ts PRIVATE ${OpenCV_LIBS})

install(TARGETS opencv_ts EXPORT opencv_ts LIBRARY DESTINATION lib INCLUDES DESTINATION include)
install(DIRECTORY include/opencv2 DESTINATION include)
install(EXPORT opencv_ts DESTINATION share/OpenCV)
