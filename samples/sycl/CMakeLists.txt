if(OPENCV_SKIP_SAMPLES_SYCL)
  return()
endif()

ocv_install_example_src(sycl *.cpp *.hpp CMakeLists.txt)

set(OPENCV_SYCL_SAMPLES_REQUIRED_DEPS
  opencv_core
  opencv_imgproc
  opencv_imgcodecs
  opencv_videoio
  opencv_highgui)
ocv_check_dependencies(${OPENCV_SYCL_SAMPLES_REQUIRED_DEPS})

if(NOT BUILD_EXAMPLES OR NOT OCV_DEPENDENCIES_FOUND OR OPENCV_SKIP_SAMPLES_BUILD_SYCL)
  return()
endif()

if(CMAKE_VERSION VERSION_LESS "3.5")
  message(STATUS "SYCL samples require CMake 3.5+")
  return()
endif()

cmake_policy(VERSION 3.5)

find_package(SYCL QUIET)  # will oneAPI support this straightforward way?

if(NOT SYCL_FOUND AND NOT OPENCV_SKIP_SAMPLES_SYCL_ONEDNN)
  # lets try scripts from oneAPI:oneDNN component
  if(NOT DEFINED DNNLROOT AND DEFINED ENV{DNNLROOT})
    set(DNNLROOT "$ENV{DNNLROOT}")
  endif()
  # Some versions of called script violate CMake policy and may emit unrecoverable CMake errors
  # Use OPENCV_SKIP_SAMPLES_SYCL=1 / OPENCV_SKIP_SAMPLES_SYCL_ONEDNN to bypass this
  find_package(dnnl CONFIG QUIET HINTS "${DNNLROOT}")
endif()

if(NOT SYCL_FOUND AND NOT OPENCV_SKIP_SAMPLES_SYCL_COMPUTECPP)
  # lets try this SYCL SDK too: https://github.com/codeplaysoftware/computecpp-sdk
  find_package(ComputeCpp QUIET)
  if(ComputeCpp_FOUND)
    set(SYCL_TARGET ComputeCpp::ComputeCpp)
    set(SYCL_FLAGS ${ComputeCpp_FLAGS})
    set(SYCL_INCLUDE_DIRS ${ComputeCpp_INCLUDE_DIRS})
    set(SYCL_LIBRARIES ${ComputeCpp_LIBRARIES})
  endif()
endif()

if(OPENCV_CMAKE_DEBUG_SYCL)
  ocv_cmake_dump_vars("SYCL")  # OpenCV source tree is required
endif()

if(NOT SYCL_TARGET)
  message(STATUS "SYCL/OpenCL samples are skipped: SYCL SDK is required")
  message(STATUS "   - check configuration of SYCL_DIR/SYCL_ROOT/CMAKE_MODULE_PATH")
  message(STATUS "   - ensure that right compiler is selected from SYCL SDK (e.g, clang++): CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}")
  return()
endif()

project(sycl_samples)

if(SYCL_FLAGS)  # "target_link_libraries(... ${SYCL_TARGET})" is not enough. Hacking...
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SYCL_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${SYCL_FLAGS}")
endif()

ocv_include_modules_recurse(${OPENCV_SYCL_SAMPLES_REQUIRED_DEPS})
ocv_include_directories(${OpenCL_INCLUDE_DIR})
file(GLOB all_samples RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)
foreach(sample_filename ${all_samples})
  ocv_define_sample(tgt ${sample_filename} sycl)
  ocv_target_link_libraries(${tgt} PRIVATE
    ${OPENCV_LINKER_LIBS}
    ${OPENCV_SYCL_SAMPLES_REQUIRED_DEPS}
    ${SYCL_TARGET})

  if(COMMAND add_sycl_to_target)  # ComputeCpp
    add_sycl_to_target(TARGET ${tgt} SOURCES ${sample_filename})
  endif()
endforeach()
