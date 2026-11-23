# Non-CPU HAL: the GPU backends of cv::accel, one source compiled for each runtime. opencv_core
# links them and creates them in its registry (modules/core/src/accel/registry.cpp).

set(gpu_hal_libs "")

# core's headers, also for nvcc, which takes the directory's include paths
ocv_include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/src"
                        "${OpenCV_SOURCE_DIR}/modules/core/include")

macro(ocv_gpu_hal_library target)
  # compiled as part of opencv_core
  target_compile_definitions(${target} PRIVATE __OPENCV_BUILD=1)
  if(BUILD_SHARED_LIBS)
    target_compile_definitions(${target} PRIVATE CVAPI_EXPORTS)
  endif()
  set_target_properties(${target} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH}
                                             DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}")
  if(NOT BUILD_SHARED_LIBS)
    ocv_install_target(${target} EXPORT OpenCVModules ARCHIVE DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev)
  endif()
  if(ENABLE_SOLUTION_FOLDERS)
    set_target_properties(${target} PROPERTIES FOLDER "3rdparty")
  endif()
  list(APPEND gpu_hal_libs ${target})
endmacro()

if(HAVE_CUDA)
  ocv_add_library(gpu_hal_cuda STATIC src/backend_cuda.cpp src/kernels.cu)
  ocv_gpu_hal_library(gpu_hal_cuda)
  if(ENABLE_CUDA_FIRST_CLASS_LANGUAGE)
    target_link_libraries(gpu_hal_cuda PRIVATE CUDA::cudart${CUDA_LIB_EXT})
  else()
    target_link_libraries(gpu_hal_cuda PRIVATE ${CUDA_LIBRARIES})
  endif()
endif()

set(OPENCV_ACCEL_GPU_LIBRARIES ${gpu_hal_libs} PARENT_SCOPE)
