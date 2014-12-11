SET(OPENCV_HAARTRAINING_DEPS opencv_core opencv_imgproc opencv_highgui opencv_objdetect opencv_calib3d opencv_video opencv_features2d opencv_flann opencv_legacy)
ocv_check_dependencies(${OPENCV_HAARTRAINING_DEPS})

if(NOT OCV_DEPENDENCIES_FOUND)
  return()
endif()

project(haartraining)

ocv_include_directories("${CMAKE_CURRENT_SOURCE_DIR}" "${OpenCV_SOURCE_DIR}/include/opencv")
ocv_include_modules(${OPENCV_HAARTRAINING_DEPS})

if(WIN32)
  link_directories(${CMAKE_CURRENT_BINARY_DIR})
endif()

# -----------------------------------------------------------
#  Library
# -----------------------------------------------------------
set(cvhaartraining_lib_src
  _cvcommon.h
  cvclassifier.h
  _cvhaartraining.h
  cvhaartraining.h
  cvboost.cpp
  cvcommon.cpp
  cvhaarclassifier.cpp
  cvhaartraining.cpp
  cvsamples.cpp
  cvsamplesoutput.cpp
  cvsamplesoutput.h
  ioutput.h
  )

add_library(opencv_haartraining_engine STATIC ${cvhaartraining_lib_src})
set_target_properties(opencv_haartraining_engine PROPERTIES
                      DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
                      ARCHIVE_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}
                      RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
                      INSTALL_NAME_DIR lib
                     )
target_link_libraries(opencv_haartraining_engine ${OPENCV_HAARTRAINING_DEPS})

if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(opencv_haartraining_engine PROPERTIES FOLDER "applications")
endif()

# -----------------------------------------------------------
#  apps
# -----------------------------------------------------------

foreach(app haartraining createsamples performance)
  # Executable
  add_executable(opencv_${app} cvhaartraining.h ${app}.cpp)
  target_link_libraries(opencv_${app} opencv_haartraining_engine ${OPENCV_HAARTRAINING_DEPS})
  set_target_properties(opencv_${app} PROPERTIES
                        DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
                        OUTPUT_NAME "opencv_${app}")

  # Install
  if(INSTALL_CREATE_DISTRIB)
    if(BUILD_SHARED_LIBS)
      install(TARGETS opencv_${app} RUNTIME DESTINATION ${OPENCV_BIN_INSTALL_PATH} CONFIGURATIONS Release COMPONENT dev)
    endif()
  else()
    install(TARGETS opencv_${app} RUNTIME DESTINATION ${OPENCV_BIN_INSTALL_PATH} COMPONENT dev)
  endif()

  # IDE
  if(ENABLE_SOLUTION_FOLDERS)
    set_target_properties(opencv_${app} PROPERTIES FOLDER "applications")
  endif()
endforeach()
