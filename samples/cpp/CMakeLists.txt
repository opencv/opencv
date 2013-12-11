# ----------------------------------------------------------------------------
#  CMake file for C samples. See root CMakeLists.txt
#
# ----------------------------------------------------------------------------

SET(OPENCV_CPP_SAMPLES_REQUIRED_DEPS opencv_core opencv_flann opencv_imgproc
    opencv_highgui opencv_ml opencv_video opencv_objdetect opencv_photo opencv_nonfree opencv_softcascade
    opencv_features2d opencv_calib3d opencv_legacy opencv_contrib opencv_stitching opencv_videostab opencv_bioinspired opencv_shape)

ocv_check_dependencies(${OPENCV_CPP_SAMPLES_REQUIRED_DEPS})


if(BUILD_EXAMPLES AND OCV_DEPENDENCIES_FOUND)
  project(cpp_samples)

  ocv_include_directories("${OpenCV_SOURCE_DIR}/include")#for opencv.hpp
  ocv_include_modules(${OPENCV_CPP_SAMPLES_REQUIRED_DEPS})

  if(HAVE_opencv_cudaoptflow)
    ocv_include_directories("${OpenCV_SOURCE_DIR}/modules/cudaoptflow/include")
  endif()
  if(HAVE_opencv_cudaimgproc)
    ocv_include_directories("${OpenCV_SOURCE_DIR}/modules/cudaimgproc/include")
  endif()
  if(HAVE_opencv_cudaarithm)
    ocv_include_directories("${OpenCV_SOURCE_DIR}/modules/cudaarithm/include")
  endif()
  if(HAVE_opencv_cudafilters)
    ocv_include_directories("${OpenCV_SOURCE_DIR}/modules/cudafilters/include")
  endif()

  if(HAVE_opencv_ocl)
    ocv_include_directories("${OpenCV_SOURCE_DIR}/modules/ocl/include")
  endif()

  if(CMAKE_COMPILER_IS_GNUCXX AND NOT ENABLE_NOISY_WARNINGS)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-function")
  endif()

  # ---------------------------------------------
  #      Define executable targets
  # ---------------------------------------------
  MACRO(OPENCV_DEFINE_CPP_EXAMPLE name srcs)

    if("${srcs}" MATCHES "tutorial_code")
      set(sample_kind tutorial)
      set(sample_KIND TUTORIAL)
      set(sample_subfolder "tutorials")
    else()
      set(sample_kind example)
      set(sample_KIND EXAMPLE)
      set(sample_subfolder "cpp")
    endif()

    set(the_target "${sample_kind}_${name}")
    add_executable(${the_target} ${srcs})
    target_link_libraries(${the_target} ${OPENCV_LINKER_LIBS} ${OPENCV_CPP_SAMPLES_REQUIRED_DEPS})

    if("${srcs}" MATCHES "gpu/")
      target_link_libraries(${the_target} opencv_cudaarithm opencv_cudafilters)
    endif()

    if(HAVE_opencv_ocl)
      target_link_libraries(${the_target} opencv_ocl)
    endif()

    set_target_properties(${the_target} PROPERTIES
      OUTPUT_NAME "cpp-${sample_kind}-${name}"
      PROJECT_LABEL "(${sample_KIND}) ${name}")

    if(ENABLE_SOLUTION_FOLDERS)
      set_target_properties(${the_target} PROPERTIES FOLDER "samples/${sample_subfolder}")
    endif()

    if(WIN32)
      if (MSVC AND NOT BUILD_SHARED_LIBS)
        set_target_properties(${the_target} PROPERTIES LINK_FLAGS "/NODEFAULTLIB:atlthunk.lib /NODEFAULTLIB:atlsd.lib /DEBUG")
      endif()
      install(TARGETS ${the_target}
              RUNTIME DESTINATION "${OPENCV_SAMPLES_BIN_INSTALL_PATH}/${sample_subfolder}" COMPONENT main)
    endif()
  ENDMACRO()

  file(GLOB_RECURSE cpp_samples RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

  if(NOT HAVE_OPENGL)
    ocv_list_filterout(cpp_samples Qt_sample)
  endif()

  if(NOT HAVE_opencv_cudaarithm OR NOT HAVE_opencv_cudafilters)
    ocv_list_filterout(cpp_samples "/gpu/")
  endif()

  ocv_list_filterout(cpp_samples "viz")

  foreach(sample_filename ${cpp_samples})
    get_filename_component(sample ${sample_filename} NAME_WE)
    OPENCV_DEFINE_CPP_EXAMPLE(${sample}  ${sample_filename})
  endforeach()
endif()

if (INSTALL_C_EXAMPLES AND NOT WIN32)
  file(GLOB C_SAMPLES *.c *.cpp *.jpg *.png *.data makefile.* build_all.sh *.dsp *.cmd )
  install(FILES ${C_SAMPLES}
          DESTINATION share/OpenCV/samples/cpp
          PERMISSIONS OWNER_READ GROUP_READ WORLD_READ)
endif()
