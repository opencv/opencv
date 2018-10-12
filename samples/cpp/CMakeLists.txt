ocv_install_example_src(cpp *.cpp *.hpp CMakeLists.txt)

set(OPENCV_CPP_SAMPLES_REQUIRED_DEPS
  opencv_core
  opencv_imgproc
  opencv_flann
  opencv_imgcodecs
  opencv_videoio
  opencv_highgui
  opencv_ml
  opencv_video
  opencv_objdetect
  opencv_photo
  opencv_features2d
  opencv_calib3d
  opencv_stitching
  opencv_videostab
  opencv_shape
  ${OPENCV_MODULES_PUBLIC}
  ${OpenCV_LIB_COMPONENTS})
ocv_check_dependencies(${OPENCV_CPP_SAMPLES_REQUIRED_DEPS})

if(NOT BUILD_EXAMPLES OR NOT OCV_DEPENDENCIES_FOUND)
  return()
endif()

project(cpp_samples)
ocv_include_modules_recurse(${OPENCV_CPP_SAMPLES_REQUIRED_DEPS})
file(GLOB_RECURSE cpp_samples RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)
if(NOT HAVE_opencv_cudaarithm OR NOT HAVE_opencv_cudafilters)
  ocv_list_filterout(cpp_samples "/gpu/")
endif()
if(NOT BUILD_opencv_viz OR NOT VTK_USE_FILE)
  ocv_list_filterout(cpp_samples "/viz/")
endif()
ocv_list_filterout(cpp_samples "real_time_pose_estimation/")
foreach(sample_filename ${cpp_samples})
  if(sample_filename MATCHES "/viz/")
    include(${VTK_USE_FILE})
    if(COMMAND ocv_warnings_disable)  # eliminate warnings from VTK headers (include directory is not "SYSTEM")
      ocv_warnings_disable(CMAKE_CXX_FLAGS -Winconsistent-missing-override -Wsuggest-override)
    endif()
  endif()
  set(package "cpp")
  if(sample_filename MATCHES "tutorial_code")
    set(package "tutorial")
  endif()
  ocv_define_sample(tgt ${sample_filename} ${package})
  ocv_target_link_libraries(${tgt} ${OPENCV_LINKER_LIBS} ${OPENCV_CPP_SAMPLES_REQUIRED_DEPS})
  if(sample_filename MATCHES "/gpu/" AND HAVE_opencv_cudaarithm AND HAVE_opencv_cuda_filters)
    ocv_target_link_libraries(${tgt} opencv_cudaarithm opencv_cudafilters)
  endif()
  if(sample_filename MATCHES "/viz/")
    ocv_target_link_libraries(${tgt} ${VTK_LIBRARIES})
    target_compile_definitions(${tgt} PRIVATE -DUSE_VTK)
  endif()
  if(HAVE_OPENGL AND sample_filename MATCHES "detect_mser")
    target_compile_definitions(${tgt} PRIVATE HAVE_OPENGL)
  endif()
endforeach()

include("tutorial_code/calib3d/real_time_pose_estimation/CMakeLists.txt" OPTIONAL)
