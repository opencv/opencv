ocv_install_example_src(tapi *.cpp *.hpp CMakeLists.txt)

set(OPENCV_TAPI_SAMPLES_REQUIRED_DEPS
  opencv_core
  opencv_imgproc
  opencv_video
  opencv_imgcodecs
  opencv_videoio
  opencv_highgui
  opencv_objdetect
  opencv_features2d
  opencv_calib3d
  opencv_flann)
ocv_check_dependencies(${OPENCV_TAPI_SAMPLES_REQUIRED_DEPS})

if(NOT BUILD_EXAMPLES OR NOT OCV_DEPENDENCIES_FOUND)
  return()
endif()

project(tapi_samples)
ocv_include_modules_recurse(${OPENCV_TAPI_SAMPLES_REQUIRED_DEPS})
file(GLOB all_samples RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)
foreach(sample_filename ${all_samples})
  ocv_define_sample(tgt ${sample_filename} tapi)
  ocv_target_link_libraries(${tgt} ${OPENCV_LINKER_LIBS} ${OPENCV_TAPI_SAMPLES_REQUIRED_DEPS})
endforeach()
