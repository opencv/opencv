set(OPENCV_TRAINCASCADE_DEPS opencv_core opencv_ml opencv_imgproc opencv_photo opencv_objdetect opencv_highgui opencv_calib3d opencv_video opencv_features2d opencv_flann opencv_legacy)
ocv_check_dependencies(${OPENCV_TRAINCASCADE_DEPS})

if(NOT OCV_DEPENDENCIES_FOUND)
  return()
endif()

project(traincascade)

ocv_include_directories("${CMAKE_CURRENT_SOURCE_DIR}" "${OpenCV_SOURCE_DIR}/include/opencv")
ocv_include_modules(${OPENCV_TRAINCASCADE_DEPS})

set(traincascade_files traincascade.cpp
  cascadeclassifier.cpp cascadeclassifier.h
  boost.cpp boost.h features.cpp traincascade_features.h
  haarfeatures.cpp haarfeatures.h
  lbpfeatures.cpp lbpfeatures.h
  HOGfeatures.cpp HOGfeatures.h
  imagestorage.cpp imagestorage.h)

set(the_target opencv_traincascade)
add_executable(${the_target} ${traincascade_files})
target_link_libraries(${the_target} ${OPENCV_TRAINCASCADE_DEPS} opencv_haartraining_engine)

set_target_properties(${the_target} PROPERTIES
                      DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
                      ARCHIVE_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}
                      RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
                      INSTALL_NAME_DIR lib
                      OUTPUT_NAME "opencv_traincascade")

if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(${the_target} PROPERTIES FOLDER "applications")
endif()

if(INSTALL_CREATE_DISTRIB)
  if(BUILD_SHARED_LIBS)
    install(TARGETS ${the_target} RUNTIME DESTINATION ${OPENCV_BIN_INSTALL_PATH} CONFIGURATIONS Release COMPONENT main)
  endif()
else()
  install(TARGETS ${the_target} RUNTIME DESTINATION ${OPENCV_BIN_INSTALL_PATH} COMPONENT main)
endif()
