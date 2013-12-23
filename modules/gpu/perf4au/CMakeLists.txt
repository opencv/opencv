set(PERF4AU_REQUIRED_DEPS opencv_core opencv_imgproc opencv_highgui opencv_video opencv_legacy opencv_gpu opencv_ts)

ocv_check_dependencies(${PERF4AU_REQUIRED_DEPS})

if (OCV_DEPENDENCIES_FOUND)
  set(the_target gpu_perf4au)
  project(${the_target})

  ocv_include_modules(${PERF4AU_REQUIRED_DEPS})

  if(CMAKE_COMPILER_IS_GNUCXX AND NOT ENABLE_NOISY_WARNINGS)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-function")
  endif()

  file(GLOB srcs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp *.h *.hpp)
  add_executable(${the_target} ${srcs})

  target_link_libraries(${the_target} ${OPENCV_LINKER_LIBS} ${PERF4AU_REQUIRED_DEPS})

  if(ENABLE_SOLUTION_FOLDERS)
    set_target_properties(${the_target} PROPERTIES FOLDER "tests performance")
  endif()

  if(WIN32)
    if(MSVC AND NOT BUILD_SHARED_LIBS)
      set_target_properties(${the_target} PROPERTIES LINK_FLAGS "/NODEFAULTLIB:atlthunk.lib /NODEFAULTLIB:atlsd.lib /DEBUG")
    endif()
  endif()
endif()