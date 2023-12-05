cmake_minimum_required(VERSION 3.5)

# Define the test project
project(DNNFuncTest)
set(SOURCES test.cpp)
add_executable(${PROJECT_NAME} ${SOURCES})

# Use opencv debug version
set(OpenCV_DIR "../../build")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

if (NOT RENDERDOC_PATH)
  message(WARNING "RENDERDOC_PATH is not defined or is empty. Set this variable to enable renderdoc support.")
  set(RENDERDOC_PATH "" CACHE PATH "Path to renderdoc header file")
else()
  message("RENDERDOC_PATH is defined: ${RENDERDOC_PATH}")
  include_directories(${RENDERDOC_PATH})
endif()

if (NOT OpenCV_CUDA_VERSION)
    message(STATUS "OpenCV was not built with CUDA.")
else()
    message(STATUS "OpenCV was built with CUDA.")
    add_definitions(-DWITH_CUDA)
endif()

# Add opencv lib
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
# message(${OpenCV_LIBS})

# Set compiler flags
set(CPP_FLAGS -g)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CPP_FLAGS}")
