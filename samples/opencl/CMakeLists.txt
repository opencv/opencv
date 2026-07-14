ocv_install_example_src(opencl *.cpp *.hpp CMakeLists.txt)

# cmake 3.1 needed for find_package(OpenCL)
if(CMAKE_VERSION VERSION_LESS "3.1")
  message(STATUS "OpenCL samples require CMakes 3.1+")
  return()
endif()

set(OPENCV_OPENCL_SAMPLES_REQUIRED_DEPS
  opencv_core
  opencv_imgproc
  opencv_video
  opencv_imgcodecs
  opencv_videoio
  opencv_highgui)
ocv_check_dependencies(${OPENCV_OPENCL_SAMPLES_REQUIRED_DEPS})

if(NOT BUILD_EXAMPLES OR NOT OCV_DEPENDENCIES_FOUND)
  return()
endif()

find_package(OpenCL 1.2 QUIET)
if(NOT OpenCL_FOUND)
  message(STATUS "OpenCL samples are skipped: OpenCL SDK is required")
  return()
endif()

project(opencl_samples)
ocv_include_modules_recurse(${OPENCV_OPENCL_SAMPLES_REQUIRED_DEPS})
ocv_include_directories(${OpenCL_INCLUDE_DIR})
file(GLOB all_samples RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)
foreach(sample_filename ${all_samples})
  ocv_define_sample(tgt ${sample_filename} opencl)
  ocv_target_link_libraries(${tgt} PRIVATE
    ${OPENCV_LINKER_LIBS}
    ${OPENCV_OPENCL_SAMPLES_REQUIRED_DEPS}
    ${OpenCL_LIBRARY})
endforeach()
