SET(OPENCV_PTCLOUD_SAMPLES_REQUIRED_DEPS
  opencv_core
  opencv_imgproc
  opencv_imgcodecs
  opencv_highgui
  opencv_ptcloud)
ocv_check_dependencies(${OPENCV_PTCLOUD_SAMPLES_REQUIRED_DEPS})

if(BUILD_EXAMPLES AND OCV_DEPENDENCIES_FOUND)
  project(ptcloud_samples)
  ocv_include_modules_recurse(${OPENCV_PTCLOUD_SAMPLES_REQUIRED_DEPS})
  file(GLOB all_samples RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)
  foreach(sample_filename ${all_samples})
    ocv_define_sample(tgt ${sample_filename} ptcloud)
    ocv_target_link_libraries(${tgt} PRIVATE ${OPENCV_PTCLOUD_SAMPLES_REQUIRED_DEPS})
  endforeach()
endif()

ocv_install_example_src(ptcloud *.cpp CMakeLists.txt)
