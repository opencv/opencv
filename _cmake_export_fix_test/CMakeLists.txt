cmake_minimum_required(VERSION 3.10)
project(OpenCVConfigTest LANGUAGES CXX)

# Let user override this from the command line if needed
set(OpenCV_DIR "${OpenCV_DIR}" CACHE PATH "Path to OpenCVConfig.cmake")

find_package(OpenCV REQUIRED)

add_executable(opencv_config_test main.cpp)
target_link_libraries(opencv_config_test PRIVATE ${OpenCV_LIBS})
target_include_directories(opencv_config_test PRIVATE ${OpenCV_INCLUDE_DIRS})
