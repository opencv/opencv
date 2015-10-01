SET(OPENCV_VA_INTEL_SAMPLES_REQUIRED_DEPS opencv_core opencv_imgproc opencv_imgcodecs opencv_videoio opencv_highgui)

ocv_check_dependencies(${OPENCV_VA_INTEL_SAMPLES_REQUIRED_DEPS})

if(BUILD_EXAMPLES AND OCV_DEPENDENCIES_FOUND)
  set(project "va_intel")
  string(TOUPPER "${project}" project_upper)

  project("${project}_samples")

  ocv_include_modules_recurse(${OPENCV_VA_INTEL_SAMPLES_REQUIRED_DEPS})

  # ---------------------------------------------
  #      Define executable targets
  # ---------------------------------------------
  MACRO(OPENCV_DEFINE_VA_INTEL_EXAMPLE name srcs)
    set(the_target "example_${project}_${name}")
    add_executable(${the_target} ${srcs})

    ocv_target_link_libraries(${the_target} ${OPENCV_LINKER_LIBS} ${OPENCV_VA_INTEL_SAMPLES_REQUIRED_DEPS} ${VA_LIBRARIES} ${VA_INTEL_LIBRARIES})

    set_target_properties(${the_target} PROPERTIES
      OUTPUT_NAME "${project}-example-${name}"
      PROJECT_LABEL "(EXAMPLE_${project_upper}) ${name}")

    if(ENABLE_SOLUTION_FOLDERS)
      set_target_properties(${the_target} PROPERTIES FOLDER "samples//${project}")
    endif()
  ENDMACRO()

  file(GLOB all_samples RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

  foreach(sample_filename ${all_samples})
    get_filename_component(sample ${sample_filename} NAME_WE)
    file(GLOB sample_srcs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${sample}.*)
    OPENCV_DEFINE_VA_INTEL_EXAMPLE(${sample} ${sample_srcs})
  endforeach()
endif()
