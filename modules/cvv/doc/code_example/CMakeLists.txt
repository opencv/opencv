cmake_minimum_required(VERSION 2.8)

project(cvvisual_test)

SET(CMAKE_PREFIX_PATH ~/software/opencv/install)

SET(CMAKE_CXX_COMPILER "g++-4.8")
SET(CMAKE_CXX_FLAGS "-std=c++11 -O2 -pthread -Wall -Werror")

OPTION(CVV_DEBUG_MODE "cvvisual-debug-mode" ON)
if(CVV_DEBUG_MODE MATCHES ON)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCVVISUAL_DEBUGMODE")
endif()


FIND_PACKAGE(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

FIND_PACKAGE(Qt5Core REQUIRED)
include_directories(${Qt5Core_INCLUDE_DIRS})
FIND_PACKAGE(Qt5Widgets REQUIRED)
include_directories(${Qt5Widgets_INCLUDE_DIRS})
add_definitions(${QT_DEFINITIONS})

SET(OpenCVVisual_DIR "$ENV{HOME}/<<<SET ME>>>")
include_directories("${OpenCVVisual_DIR}/include")
link_directories("${OpenCVVisual_DIR}/build/release")

add_executable(cvvt main.cpp)
target_link_libraries(cvvt
  opencv_core opencv_highgui opencv_imgproc opencv_features2d
  cvvisual
  Qt5Core Qt5Widgets Qt5Gui
)
