SET(OPENCV_DIRECTX_SAMPLES_REQUIRED_DEPS opencv_core opencv_imgproc opencv_imgcodecs opencv_videoio opencv_highgui)

ocv_check_dependencies(${OPENCV_DIRECTX_SAMPLES_REQUIRED_DEPS})

if(BUILD_EXAMPLES AND OCV_DEPENDENCIES_FOUND)
  set(project "directx")
  string(TOUPPER "${project}" project_upper)

  project("${project}_samples")

  ocv_include_modules(${OPENCV_DIRECTX_SAMPLES_REQUIRED_DEPS})

  # ---------------------------------------------
  #      Define executable targets
  # ---------------------------------------------
  MACRO(OPENCV_DEFINE_DIRECTX_EXAMPLE name srcs)
    set(the_target "example_${project}_${name}")
    add_executable(${the_target} ${srcs})

    ocv_target_link_libraries(${the_target} ${OPENCV_LINKER_LIBS} ${OPENCV_DIRECTX_SAMPLES_REQUIRED_DEPS})

    set_target_properties(${the_target} PROPERTIES
      OUTPUT_NAME "${project}-example-${name}"
      PROJECT_LABEL "(EXAMPLE_${project_upper}) ${name}")

    if(ENABLE_SOLUTION_FOLDERS)
      set_target_properties(${the_target} PROPERTIES FOLDER "samples//${project}")
    endif()

    if(WIN32)
      if(MSVC AND NOT BUILD_SHARED_LIBS)
        set_target_properties(${the_target} PROPERTIES LINK_FLAGS "/NODEFAULTLIB:atlthunk.lib /NODEFAULTLIB:atlsd.lib /DEBUG")
      endif()
      install(TARGETS ${the_target} RUNTIME DESTINATION "${OPENCV_SAMPLES_BIN_INSTALL_PATH}/${project}" COMPONENT main)
    endif()
  ENDMACRO()

  file(GLOB all_samples RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

  foreach(sample_filename ${all_samples})
    get_filename_component(sample ${sample_filename} NAME_WE)
    file(GLOB sample_srcs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${sample}.*)
    OPENCV_DEFINE_DIRECTX_EXAMPLE(${sample} ${sample_srcs})
  endforeach()
endif()
