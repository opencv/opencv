set(OPENCV_CREATESAMPLES_DEPS opencv_core opencv_imgproc opencv_objdetect opencv_imgcodecs opencv_highgui opencv_calib3d opencv_features2d opencv_videoio)
ocv_check_dependencies(${OPENCV_CREATESAMPLES_DEPS})

if(NOT OCV_DEPENDENCIES_FOUND)
  return()
endif()

project(createsamples)
set(the_target opencv_createsamples)

ocv_target_include_directories(${the_target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${OpenCV_SOURCE_DIR}/include/opencv")
ocv_target_include_modules_recurse(${the_target} ${OPENCV_CREATESAMPLES_DEPS})

file(GLOB SRCS *.cpp)
file(GLOB HDRS *.h*)

set(createsamples_files ${SRCS} ${HDRS})

ocv_add_executable(${the_target} ${createsamples_files})
ocv_target_link_libraries(${the_target} ${OPENCV_CREATESAMPLES_DEPS})

set_target_properties(${the_target} PROPERTIES
                      DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
                      ARCHIVE_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}
                      RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
                      INSTALL_NAME_DIR lib
                      OUTPUT_NAME "opencv_createsamples")

if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(${the_target} PROPERTIES FOLDER "applications")
endif()

if(INSTALL_CREATE_DISTRIB)
  if(BUILD_SHARED_LIBS)
    install(TARGETS ${the_target} RUNTIME DESTINATION ${OPENCV_BIN_INSTALL_PATH} CONFIGURATIONS Release COMPONENT dev)
  endif()
else()
  install(TARGETS ${the_target} OPTIONAL RUNTIME DESTINATION ${OPENCV_BIN_INSTALL_PATH} COMPONENT dev)
endif()
