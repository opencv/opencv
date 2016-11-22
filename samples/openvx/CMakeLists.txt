cmake_minimum_required(VERSION 2.8.9)

set(OPENCV_OPENVX_SAMPLE_REQUIRED_DEPS opencv_core opencv_imgproc opencv_imgcodecs opencv_videoio opencv_highgui)

ocv_check_dependencies(${OPENCV_OPENVX_SAMPLE_REQUIRED_DEPS})

if(BUILD_EXAMPLES AND OCV_DEPENDENCIES_FOUND)
  set(group "openvx")
  set(name_wrapped "interop")
  set(name_orig "interop_orig")
  set(name_video "interop_video")

  project("${group}_sample")

  ocv_include_modules_recurse(${OPENCV_OPENVX_SAMPLE_REQUIRED_DEPS})

  add_definitions(-DIVX_USE_OPENCV)
  add_definitions(-DIVX_HIDE_INFO_WARNINGS)

  file(GLOB srcs_wrapped wrappers.cpp *.hpp)
  file(GLOB srcs_orig  no_wrappers.cpp *.hpp)
  file(GLOB srcs_video wrappers_video.cpp *.hpp)

  MACRO(OPENVX_DEFINE_SAMPLE name srcs)
    set(target "example_${group}_${name}")
	add_executable(${target} ${srcs})
    ocv_target_link_libraries(${target} ${OPENCV_LINKER_LIBS} ${OPENCV_OPENVX_SAMPLE_REQUIRED_DEPS} ${OPENVX_LIBRARIES})
    if(ENABLE_SOLUTION_FOLDERS)
        set_target_properties(${target} PROPERTIES FOLDER "samples//${group}")
    endif()
  ENDMACRO()

  OPENVX_DEFINE_SAMPLE(${name_wrapped} ${srcs_wrapped})
  OPENVX_DEFINE_SAMPLE(${name_orig} ${srcs_orig})
  OPENVX_DEFINE_SAMPLE(${name_video} ${srcs_video})

endif()
