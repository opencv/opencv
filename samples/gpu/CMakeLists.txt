ocv_install_example_src(gpu *.cpp *.hpp CMakeLists.txt)

set(OPENCV_CUDA_SAMPLES_REQUIRED_DEPS
  opencv_core
  opencv_flann
  opencv_imgproc
  opencv_imgcodecs
  opencv_videoio
  opencv_highgui
  opencv_ml
  opencv_video
  opencv_objdetect
  opencv_features2d
  opencv_calib3d
  opencv_superres
  opencv_cudaarithm
  opencv_cudafilters
  opencv_cudawarping
  opencv_cudaimgproc
  opencv_cudafeatures2d
  opencv_cudaoptflow
  opencv_cudabgsegm
  opencv_cudastereo
  opencv_cudaobjdetect)
ocv_check_dependencies(${OPENCV_CUDA_SAMPLES_REQUIRED_DEPS})

if(NOT BUILD_EXAMPLES OR NOT OCV_DEPENDENCIES_FOUND)
  return()
endif()

project(gpu_samples)
if(HAVE_CUDA OR CUDA_FOUND)
  add_definitions(-DHAVE_CUDA=1)
endif()
if(COMMAND ocv_warnings_disable)
  ocv_warnings_disable(CMAKE_CXX_FLAGS -Wsuggest-override -Winconsistent-missing-override)
endif()
ocv_include_modules_recurse(${OPENCV_CUDA_SAMPLES_REQUIRED_DEPS})
if(HAVE_opencv_xfeatures2d)
  ocv_include_modules_recurse(opencv_xfeatures2d)
endif()
if(HAVE_opencv_cudacodec)
  ocv_include_modules_recurse(opencv_cudacodec)
endif()
if(HAVE_CUDA)
  ocv_include_directories(${CUDA_INCLUDE_DIRS})
endif()
if((CV_GCC OR CV_CLANG) AND NOT ENABLE_NOISY_WARNINGS)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-function")
endif()
file(GLOB all_samples RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)
foreach(sample_filename ${all_samples})
  ocv_define_sample(tgt ${sample_filename} gpu)
  ocv_target_link_libraries(${tgt} ${OPENCV_LINKER_LIBS} ${OPENCV_CUDA_SAMPLES_REQUIRED_DEPS})
  if(HAVE_opencv_xfeatures2d)
    ocv_target_link_libraries(${tgt} opencv_xfeatures2d)
  endif()
  if(HAVE_opencv_cudacodec)
    ocv_target_link_libraries(${tgt} opencv_cudacodec)
  endif()
endforeach()
