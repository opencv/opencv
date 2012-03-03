# ----------------------------------------------------------------------------
#  CMake file for C samples. See root CMakeLists.txt
#
# ----------------------------------------------------------------------------

SET(OPENCV_CPP_SAMPLES_REQUIRED_DEPS opencv_core opencv_flann opencv_imgproc opencv_highgui opencv_ml opencv_video opencv_objdetect
                                     opencv_features2d opencv_calib3d opencv_legacy opencv_contrib opencv_stitching)

ocv_check_dependencies(${OPENCV_CPP_SAMPLES_REQUIRED_DEPS})


if(BUILD_EXAMPLES AND OCV_DEPENDENCIES_FOUND)
  project(cpp_samples)

  ocv_include_directories("${OpenCV_SOURCE_DIR}/include")#for opencv.hpp
  ocv_include_modules(${OPENCV_CPP_SAMPLES_REQUIRED_DEPS})

  if (HAVE_opencv_gpu)
    ocv_include_directories("${OpenCV_SOURCE_DIR}/modules/gpu/include")
  endif()

  if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-function")
  endif()
    
  # ---------------------------------------------
  #      Define executable targets
  # ---------------------------------------------
  MACRO(OPENCV_DEFINE_CPP_EXAMPLE name srcs)
    set(the_target "example_${name}")
    add_executable(${the_target} ${srcs})
    target_link_libraries(${the_target} ${OPENCV_LINKER_LIBS} ${OPENCV_CPP_SAMPLES_REQUIRED_DEPS})

    if (HAVE_opencv_gpu)
      target_link_libraries(${the_target} opencv_gpu)
    endif()

    set_target_properties(${the_target} PROPERTIES
      OUTPUT_NAME "${name}"
      PROJECT_LABEL "(EXAMPLE) ${name}")
            
    if(ENABLE_SOLUTION_FOLDERS)
      set_target_properties(${the_target} PROPERTIES FOLDER "samples//cpp")
    endif()

    if(WIN32)
      if (MSVC AND NOT BUILD_SHARED_LIBS)
        set_target_properties(${the_target} PROPERTIES LINK_FLAGS "/NODEFAULTLIB:atlthunk.lib /NODEFAULTLIB:atlsd.lib /DEBUG")
      endif()
      install(TARGETS ${the_target}
              RUNTIME DESTINATION "samples/cpp" COMPONENT main)
    endif()
  ENDMACRO()
    
  file(GLOB cpp_samples RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)
    
  foreach(sample_filename ${cpp_samples})
    get_filename_component(sample ${sample_filename} NAME_WE)
    OPENCV_DEFINE_CPP_EXAMPLE(${sample}  ${sample_filename})
  endforeach()
endif()

if (INSTALL_C_EXAMPLES AND NOT WIN32)
  file(GLOB C_SAMPLES *.c *.cpp *.jpg *.png *.data makefile.* build_all.sh *.dsp *.cmd )
  install(FILES ${C_SAMPLES}
          DESTINATION share/opencv/samples/cpp
          PERMISSIONS OWNER_READ GROUP_READ WORLD_READ)
endif()

