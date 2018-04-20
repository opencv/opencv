if(APPLE)
    return()
endif()

if(UNIX)
  find_package(X11 QUIET)
  if(NOT X11_FOUND)
    message(STATUS "OpenGL samples require development files for libX11")
    return()
  endif()
  include_directories(${X11_INCLUDE_DIR})
  set(SAMPLE_LINKER_DEPS "${X11_LIBRARIES}")
endif()

SET(OPENCV_OPENGL_SAMPLES_REQUIRED_DEPS
  opencv_core
  opencv_imgproc
  opencv_imgcodecs
  opencv_videoio
  opencv_highgui)
ocv_check_dependencies(${OPENCV_OPENGL_SAMPLES_REQUIRED_DEPS})

if(BUILD_EXAMPLES AND OCV_DEPENDENCIES_FOUND)
  project(opengl_samples)
  ocv_include_modules_recurse(${OPENCV_OPENGL_SAMPLES_REQUIRED_DEPS})
  file(GLOB all_samples RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)
  foreach(sample_filename ${all_samples})
    ocv_define_sample(tgt ${sample_filename} opengl)
    ocv_target_link_libraries(${tgt}
      ${OPENCV_LINKER_LIBS} ${OPENCV_OPENGL_SAMPLES_REQUIRED_DEPS} ${SAMPLE_LINKER_DEPS})
  endforeach()
endif()

ocv_install_example_src(opengl *.cpp *.hpp CMakeLists.txt)
