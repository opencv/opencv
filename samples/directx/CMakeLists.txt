ocv_install_example_src(directx *.cpp *.hpp CMakeLists.txt)

set(OPENCV_DIRECTX_SAMPLES_REQUIRED_DEPS
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
    opencv_videoio
    opencv_highgui)
ocv_check_dependencies(${OPENCV_DIRECTX_SAMPLES_REQUIRED_DEPS})

if(NOT BUILD_EXAMPLES OR NOT OCV_DEPENDENCIES_FOUND)
  return()
endif()

project("directx_samples")
ocv_include_modules_recurse(${tgt} ${OPENCV_DIRECTX_SAMPLES_REQUIRED_DEPS})
file(GLOB all_samples RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)
foreach(sample_filename ${all_samples})
  ocv_define_sample(tgt ${sample_filename} directx)
  ocv_target_link_libraries(${tgt} ${OPENCV_LINKER_LIBS} ${OPENCV_DIRECTX_SAMPLES_REQUIRED_DEPS})
endforeach()
