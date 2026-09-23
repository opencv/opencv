ocv_install_example_src(vlm *.cpp *.py CMakeLists.txt)

set(OPENCV_VLM_SAMPLES_REQUIRED_DEPS
  opencv_core
  opencv_dnn
  opencv_imgcodecs
  opencv_vlm)
ocv_check_dependencies(${OPENCV_VLM_SAMPLES_REQUIRED_DEPS})

if(NOT BUILD_EXAMPLES OR NOT OCV_DEPENDENCIES_FOUND)
  return()
endif()

project(vlm_samples)
ocv_include_modules_recurse(${OPENCV_VLM_SAMPLES_REQUIRED_DEPS})

file(GLOB_RECURSE vlm_samples RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)
foreach(sample_filename ${vlm_samples})
  ocv_define_sample(tgt ${sample_filename} vlm)
  ocv_target_link_libraries(${tgt} PRIVATE ${OPENCV_LINKER_LIBS} ${OPENCV_VLM_SAMPLES_REQUIRED_DEPS})
endforeach()
