# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html

set(PROJECT_NAME norm)
project(${PROJECT_NAME})

ocv_install_example_src(norm *.cpp *.hpp CMakeLists.txt)

set(LOCAL_DEPS
  opencv_core
  ${OPENCV_MODULES_PUBLIC}
  ${OpenCV_LIB_COMPONENTS})
ocv_check_dependencies(${LOCAL_DEPS})

if(NOT OCV_DEPENDENCIES_FOUND)
  return()
endif()

ocv_define_sample(norm norm.cpp ${SEMIHOSTING_SUFFIX})
ocv_include_modules_recurse(${LOCAL_DEPS})
target_include_directories(${norm} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(${norm} PRIVATE ${RAW_PIXEL_INCLUDE})
ocv_target_link_libraries(${norm} PRIVATE ${OPENCV_LINKER_LIBS}
  ${LOCAL_DEPS})
