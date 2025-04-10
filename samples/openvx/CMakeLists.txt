ocv_install_example_src(cpp *.cpp *.hpp CMakeLists.txt)

cmake_minimum_required(VERSION 3.5)

set(OPENCV_OPENVX_SAMPLE_REQUIRED_DEPS
  opencv_core
  opencv_imgproc
  opencv_imgcodecs
  opencv_videoio
  opencv_highgui)
ocv_check_dependencies(${OPENCV_OPENVX_SAMPLE_REQUIRED_DEPS})

if(NOT BUILD_EXAMPLES OR NOT OCV_DEPENDENCIES_FOUND)
  return()
endif()

project(openvx_samples)
ocv_include_modules_recurse(${OPENCV_OPENVX_SAMPLE_REQUIRED_DEPS})
add_definitions(-DIVX_USE_OPENCV)
add_definitions(-DIVX_HIDE_INFO_WARNINGS)
file(GLOB_RECURSE cpp_samples RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)
foreach(sample_filename ${cpp_samples})
  ocv_define_sample(tgt ${sample_filename} openvx)
  ocv_target_link_libraries(${tgt} PRIVATE ${OPENCV_LINKER_LIBS} ${OPENCV_OPENVX_SAMPLE_REQUIRED_DEPS})
endforeach()
