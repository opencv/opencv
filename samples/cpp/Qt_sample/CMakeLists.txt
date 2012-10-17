#http://www.cmake.org/cmake/help/cmake2.6docs.html
cmake_minimum_required (VERSION 2.6)
project (OpenGL_Qt_Binding)


FIND_PACKAGE( OpenCV REQUIRED )
find_package (OpenGL REQUIRED)


ADD_EXECUTABLE(OpenGL_Qt_Binding main.cpp)
TARGET_LINK_LIBRARIES(OpenGL_Qt_Binding ${OpenCV_LIBS} ${OPENGL_LIBRARIES} )
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube4.avi ${CMAKE_CURRENT_BINARY_DIR}/cube4.avi COPYONLY)
