SET(OPENCV_TAPI_SAMPLES_REQUIRED_DEPS opencv_core opencv_imgproc opencv_video opencv_highgui opencv_objdetect opencv_features2d opencv_calib3d opencv_nonfree opencv_flann)

ocv_check_dependencies(${OPENCV_TAPI_SAMPLES_REQUIRED_DEPS})

if(BUILD_EXAMPLES AND OCV_DEPENDENCIES_FOUND)
  set(project "tapi")
  string(TOUPPER "${project}" project_upper)

  project("${project}_samples")

  ocv_include_modules(${OPENCV_TAPI_SAMPLES_REQUIRED_DEPS})

  # ---------------------------------------------
  #      Define executable targets
  # ---------------------------------------------
  MACRO(OPENCV_DEFINE_TAPI_EXAMPLE name srcs)
    set(the_target "example_${project}_${name}")
    add_executable(${the_target} ${srcs})

    target_link_libraries(${the_target} ${OPENCV_LINKER_LIBS} ${OPENCV_TAPI_SAMPLES_REQUIRED_DEPS})

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
    OPENCV_DEFINE_TAPI_EXAMPLE(${sample} ${sample_srcs})
  endforeach()
endif()

if(INSTALL_C_EXAMPLES AND NOT WIN32)
  file(GLOB install_list *.c *.cpp *.jpg *.png *.data makefile.* build_all.sh *.dsp *.cmd )
  install(FILES ${install_list}
          DESTINATION ${OPENCV_SAMPLES_SRC_INSTALL_PATH}/tapi
          PERMISSIONS OWNER_READ GROUP_READ WORLD_READ COMPONENT samples)
endif()
