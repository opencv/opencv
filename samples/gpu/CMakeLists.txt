SET(OPENCV_CUDA_SAMPLES_REQUIRED_DEPS opencv_core opencv_flann opencv_imgproc opencv_imgcodecs opencv_videoio opencv_highgui
                                     opencv_ml opencv_video opencv_objdetect opencv_features2d
                                     opencv_calib3d opencv_cuda opencv_superres
                                     opencv_cudaarithm opencv_cudafilters opencv_cudawarping opencv_cudaimgproc
                                     opencv_cudafeatures2d opencv_cudaoptflow opencv_cudabgsegm
                                     opencv_cudastereo opencv_cudalegacy)

ocv_check_dependencies(${OPENCV_CUDA_SAMPLES_REQUIRED_DEPS})

if(BUILD_EXAMPLES AND OCV_DEPENDENCIES_FOUND)
  set(project "gpu")
  string(TOUPPER "${project}" project_upper)

  project("${project}_samples")

  ocv_include_modules(${OPENCV_CUDA_SAMPLES_REQUIRED_DEPS})
  ocv_include_directories(
    "${OpenCV_SOURCE_DIR}/modules/gpu/src/nvidia"
    "${OpenCV_SOURCE_DIR}/modules/gpu/src/nvidia/core"
    )

  if(HAVE_opencv_nonfree)
    ocv_include_directories("${OpenCV_SOURCE_DIR}/modules/nonfree/include")
  endif()

  if(HAVE_opencv_cudacodec)
    ocv_include_directories("${OpenCV_SOURCE_DIR}/modules/cudacodec/include")
  endif()

  if(HAVE_CUDA)
    ocv_include_directories(${CUDA_INCLUDE_DIRS})
  endif()

  if(HAVE_OPENCL)
    ocv_include_directories("${OpenCV_SOURCE_DIR}/modules/ocl/include")
  endif()

  if(CMAKE_COMPILER_IS_GNUCXX AND NOT ENABLE_NOISY_WARNINGS)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-function")
  endif()

  # ---------------------------------------------
  #      Define executable targets
  # ---------------------------------------------
  MACRO(OPENCV_DEFINE_CUDA_EXAMPLE name srcs)
    set(the_target "example_${project}_${name}")
    add_executable(${the_target} ${srcs})

    ocv_target_link_libraries(${the_target} ${OPENCV_LINKER_LIBS} ${OPENCV_CUDA_SAMPLES_REQUIRED_DEPS})

    if(HAVE_CUDA AND NOT ANDROID)
      ocv_target_link_libraries(${the_target} ${CUDA_CUDA_LIBRARY})
    endif()

    if(HAVE_opencv_nonfree)
      ocv_target_link_libraries(${the_target} opencv_nonfree)
    endif()
    if(HAVE_opencv_cudacodec)
      ocv_target_link_libraries(${the_target} opencv_cudacodec)
    endif()

    if(HAVE_opencv_ocl)
      ocv_target_link_libraries(${the_target} opencv_ocl)
    endif()

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
      install(TARGETS ${the_target} RUNTIME DESTINATION "${OPENCV_SAMPLES_BIN_INSTALL_PATH}/${project}" COMPONENT samples)
    endif()
  ENDMACRO()

  file(GLOB all_samples RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

  foreach(sample_filename ${all_samples})
    get_filename_component(sample ${sample_filename} NAME_WE)
    file(GLOB sample_srcs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${sample}.*)
    OPENCV_DEFINE_CUDA_EXAMPLE(${sample} ${sample_srcs})
  endforeach()

  include("performance/CMakeLists.txt")
endif()

if(INSTALL_C_EXAMPLES AND NOT WIN32)
  file(GLOB install_list *.c *.cpp *.jpg *.png *.data makefile.* build_all.sh *.dsp *.cmd )
  install(FILES ${install_list}
          DESTINATION ${OPENCV_SAMPLES_SRC_INSTALL_PATH}/gpu
          PERMISSIONS OWNER_READ GROUP_READ WORLD_READ COMPONENT samples)
endif()
