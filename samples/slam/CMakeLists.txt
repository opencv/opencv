ocv_install_example_src(slam *.cpp *.hpp CMakeLists.txt)

set(OPENCV_SLAM_SAMPLES_REQUIRED_DEPS
  opencv_core
  opencv_dnn
  opencv_features
  opencv_ptcloud)
ocv_check_dependencies(${OPENCV_SLAM_SAMPLES_REQUIRED_DEPS})

if(NOT BUILD_EXAMPLES OR NOT OCV_DEPENDENCIES_FOUND)
  return()
endif()

project(slam_samples)
ocv_include_modules_recurse(${OPENCV_SLAM_SAMPLES_REQUIRED_DEPS})
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../dnn")

file(GLOB_RECURSE slam_samples RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)
foreach(sample_filename ${slam_samples})
  ocv_define_sample(tgt ${sample_filename} slam)
  ocv_target_link_libraries(${tgt} PRIVATE ${OPENCV_LINKER_LIBS} ${OPENCV_SLAM_SAMPLES_REQUIRED_DEPS})
endforeach()
