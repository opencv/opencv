cmake_minimum_required(VERSION 3.5)

# Define the test project
project(DNNFuncTest)
set(SOURCES test.cpp)
add_executable(${PROJECT_NAME} ${SOURCES})

# Use opencv debug version
set(OpenCV_DIR "../../build")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Add opencv lib
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
# message(${OpenCV_LIBS})

# Set compiler flags
set(CPP_FLAGS -g)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CPP_FLAGS}")