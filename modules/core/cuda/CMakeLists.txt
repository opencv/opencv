project(opencv_core_cuda)
add_definitions(-DUSE_CUDA)
include_directories(${CUDA_INCLUDE_DIRS}
                    "../src/"
                    "../include/opencv2/core/"
                    "${OpenCV_SOURCE_DIR}/modules/gpu/include"
                   )
ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef)
cuda_add_library(opencv_core_cuda SHARED main.cpp ../src/cuda/matrix_operations.cu)
target_link_libraries(opencv_core_cuda ${CUDA_LIBRARIES})