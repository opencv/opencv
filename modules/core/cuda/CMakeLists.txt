project(opencv_core_cuda)
add_definitions(-DUSE_CUDA)
include_directories(${CUDA_INCLUDE_DIRS}
                    "../src/"
                    "../include/opencv2/core/"
                    "${OpenCV_SOURCE_DIR}/modules/gpu/include"
                   )
ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef)
cuda_add_library(opencv_core_cuda SHARED main.cpp ../src/cuda/matrix_operations.cu)
if(BUILD_FAT_JAVA_LIB)
  target_link_libraries(opencv_core_cuda ${OPENCV_BUILD_DIR}/${LIBRARY_OUTPUT_PATH}/libopencv_java.so ${CUDA_LIBRARIES} ${CUDA_npp_LIBRARY})
else()
  target_link_libraries(opencv_core_cuda ${OPENCV_BUILD_DIR}/${LIBRARY_OUTPUT_PATH}/libopencv_core.so ${CUDA_LIBRARIES} ${CUDA_npp_LIBRARY})
endif()