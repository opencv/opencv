SET(OPENCV_GPU_SAMPLES_REQUIRED_DEPS opencv_core opencv_flann opencv_imgproc opencv_highgui
                                     opencv_ml opencv_video opencv_objdetect opencv_features2d
                                     opencv_calib3d opencv_legacy opencv_contrib opencv_gpu
                                     opencv_nonfree)

ocv_check_dependencies(${OPENCV_GPU_SAMPLES_REQUIRED_DEPS})

if(BUILD_EXAMPLES AND OCV_DEPENDENCIES_FOUND)
  set(project "gpu")
  string(TOUPPER "${project}" project_upper)
    
  project("${project}_samples")

  ocv_include_modules(${OPENCV_GPU_SAMPLES_REQUIRED_DEPS})
  ocv_include_directories(
    "${OpenCV_SOURCE_DIR}/modules/gpu/src/nvidia"
    "${OpenCV_SOURCE_DIR}/modules/gpu/src/nvidia/core"
    )

  if(HAVE_CUDA)
    ocv_include_directories(${CUDA_INCLUDE_DIRS})
  endif()

  if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-function")
  endif()

  # ---------------------------------------------
  #      Define executable targets
  # ---------------------------------------------
  MACRO(OPENCV_DEFINE_GPU_EXAMPLE name srcs)
    set(the_target "example_${project}_${name}")
    add_executable(${the_target} ${srcs})

    target_link_libraries(${the_target} ${OPENCV_LINKER_LIBS} ${OPENCV_GPU_SAMPLES_REQUIRED_DEPS})

    set_target_properties(${the_target} PROPERTIES
      OUTPUT_NAME "${name}_${project}"                                       
      PROJECT_LABEL "(EXAMPLE_${project_upper}) ${name}")
        
    if(ENABLE_SOLUTION_FOLDERS)
      set_target_properties(${the_target} PROPERTIES FOLDER "samples//${project}")
    endif()

    if(WIN32)
      if(MSVC AND NOT BUILD_SHARED_LIBS)
        set_target_properties(${the_target} PROPERTIES LINK_FLAGS "/NODEFAULTLIB:atlthunk.lib /NODEFAULTLIB:atlsd.lib /DEBUG")
      endif()
      install(TARGETS ${the_target} RUNTIME DESTINATION "samples/${project}" COMPONENT main)
    endif()
  ENDMACRO()

  file(GLOB all_samples RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

  foreach(sample_filename ${all_samples})
    get_filename_component(sample ${sample_filename} NAME_WE)
    file(GLOB sample_srcs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${sample}.*)
    OPENCV_DEFINE_GPU_EXAMPLE(${sample} ${sample_srcs})
  endforeach()

  include("performance/CMakeLists.txt")
endif()

if (NOT WIN32)
  file(GLOB install_list *.c *.cpp *.jpg *.png *.data makefile.* build_all.sh *.dsp *.cmd )
  install(FILES ${install_list}
          DESTINATION share/opencv/samples/${project}
          PERMISSIONS OWNER_READ GROUP_READ WORLD_READ)
endif()

