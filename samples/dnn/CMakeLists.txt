SET(OPENCV_DNN_SAMPLES_REQUIRED_DEPS opencv_core opencv_imgproc opencv_dnn
    opencv_imgcodecs opencv_videoio opencv_highgui
    ${OpenCV_LIB_COMPONENTS})

ocv_check_dependencies(${OPENCV_DNN_SAMPLES_REQUIRED_DEPS})


if(BUILD_EXAMPLES AND OCV_DEPENDENCIES_FOUND)
  project(dnn_samples)

  # Model branch name: dnn_samples_face_detector_20170830
  set(DNN_FACE_DETECTOR_MODEL_COMMIT "b2bfc75f6aea5b1f834ff0f0b865a7c18ff1459f")
  set(DNN_FACE_DETECTOR_MODEL_HASH "afbb6037fd180e8d2acb3b58ca737b9e")

  set(DNN_FACE_DETECTOR_MODEL_NAME "res10_300x300_ssd_iter_140000.caffemodel")

  set(DNN_FACE_DETECTOR_MODEL_DOWNLOAD_DIR "${CMAKE_CURRENT_LIST_DIR}/face_detector")

  if(COMMAND ocv_download)
    ocv_download(FILENAME ${DNN_FACE_DETECTOR_MODEL_NAME}
               HASH ${DNN_FACE_DETECTOR_MODEL_HASH}
               URL
                 "$ENV{OPENCV_DNN_MODELS_URL}"
                 "${OPENCV_DNN_MODELS_URL}"
                 "https://raw.githubusercontent.com/opencv/opencv_3rdparty/${DNN_FACE_DETECTOR_MODEL_COMMIT}/"
               DESTINATION_DIR ${DNN_FACE_DETECTOR_MODEL_DOWNLOAD_DIR}
               ID DNN_FACE_DETECTOR
               RELATIVE_URL
               STATUS res)
  endif()

  ocv_include_directories("${OpenCV_SOURCE_DIR}/include")
  ocv_include_modules_recurse(${OPENCV_DNN_SAMPLES_REQUIRED_DEPS})

  # ---------------------------------------------
  #      Define executable targets
  # ---------------------------------------------
  MACRO(OPENCV_DEFINE_DNN_EXAMPLE name srcs)
    set(sample_kind example_dnn)
    set(sample_subfolder "dnn")

    set(the_target "${sample_kind}_${name}")
    add_executable(${the_target} ${srcs})
    ocv_target_link_libraries(${the_target} ${OPENCV_LINKER_LIBS} ${OPENCV_DNN_SAMPLES_REQUIRED_DEPS})

    set_target_properties(${the_target} PROPERTIES
      OUTPUT_NAME "${sample_kind}-${name}"
      PROJECT_LABEL "(${sample_KIND}) ${name}")

    if(ENABLE_SOLUTION_FOLDERS)
      set_target_properties(${the_target} PROPERTIES FOLDER "samples/${sample_subfolder}")
    endif()

    if(WIN32)
      if (MSVC AND NOT BUILD_SHARED_LIBS)
        set_target_properties(${the_target} PROPERTIES LINK_FLAGS "/NODEFAULTLIB:atlthunk.lib /NODEFAULTLIB:atlsd.lib /DEBUG")
      endif()
      install(TARGETS ${the_target}
              RUNTIME DESTINATION "${OPENCV_SAMPLES_BIN_INSTALL_PATH}/${sample_subfolder}" COMPONENT samples)
    endif()
  ENDMACRO()

  file(GLOB_RECURSE dnn_samples RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

  foreach(sample_filename ${dnn_samples})
    get_filename_component(sample ${sample_filename} NAME_WE)
    OPENCV_DEFINE_DNN_EXAMPLE(${sample}  ${sample_filename})
  endforeach()
endif()

if(INSTALL_C_EXAMPLES AND NOT WIN32)
  file(GLOB C_SAMPLES *.c *.cpp *.jpg *.png *.data makefile.* build_all.sh *.dsp *.cmd )
  install(FILES ${C_SAMPLES}
          DESTINATION ${OPENCV_SAMPLES_SRC_INSTALL_PATH}/dnn
          PERMISSIONS OWNER_READ GROUP_READ WORLD_READ COMPONENT samples)
endif()
