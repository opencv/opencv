set(OPENCV_REQUIRED_DEPS
  opencv_gapi
  opencv_core
  opencv_imgproc

  opencv_imgcodecs
  opencv_videoio
  opencv_highgui
)
ocv_check_dependencies(${OPENCV_REQUIRED_DEPS})

if(NOT OCV_DEPENDENCIES_FOUND)
  return()
endif()

project(gapi_documentation_code)
ocv_include_modules_recurse(${OPENCV_REQUIRED_DEPS})

file(GLOB code_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)
foreach(code_filename ${code_files})
  ocv_define_sample(tgt ${code_filename} "tutorial")
  ocv_target_link_libraries(${tgt} ${OPENCV_LINKER_LIBS} ${OPENCV_REQUIRED_DEPS})
endforeach()
