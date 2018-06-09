set(OPENCV_APPLICATION_DEPS opencv_core)
ocv_check_dependencies(${OPENCV_APPLICATION_DEPS})
if(NOT OCV_DEPENDENCIES_FOUND)
  return()
endif()

project(opencv_version)
set(the_target opencv_version)
ocv_target_include_modules_recurse(${the_target} ${OPENCV_APPLICATION_DEPS})
ocv_add_executable(${the_target} opencv_version.cpp)
ocv_target_link_libraries(${the_target} ${OPENCV_APPLICATION_DEPS})

set_target_properties(${the_target} PROPERTIES
                      DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
                      RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
                      OUTPUT_NAME "opencv_version")

set_target_properties(${the_target} PROPERTIES FOLDER "applications")

if(INSTALL_CREATE_DISTRIB)
  if(BUILD_SHARED_LIBS)
    install(TARGETS ${the_target} RUNTIME DESTINATION ${OPENCV_BIN_INSTALL_PATH} CONFIGURATIONS Release COMPONENT libs)
  endif()
else()
  install(TARGETS ${the_target} RUNTIME DESTINATION ${OPENCV_BIN_INSTALL_PATH} COMPONENT libs)
endif()

if(WIN32)
  project(opencv_version_win32)
  set(the_target opencv_version_win32)
  ocv_target_include_modules_recurse(${the_target} ${OPENCV_APPLICATION_DEPS})
  ocv_add_executable(${the_target} opencv_version.cpp)
  ocv_target_link_libraries(${the_target} ${OPENCV_APPLICATION_DEPS})
  target_compile_definitions(${the_target} PRIVATE "OPENCV_WIN32_API=1")
  set_target_properties(${the_target} PROPERTIES
                        DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
                        RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
                        OUTPUT_NAME "opencv_version_win32")

  set_target_properties(${the_target} PROPERTIES FOLDER "applications")

  if(INSTALL_CREATE_DISTRIB)
    if(BUILD_SHARED_LIBS)
      install(TARGETS ${the_target} RUNTIME DESTINATION ${OPENCV_BIN_INSTALL_PATH} CONFIGURATIONS Release COMPONENT libs)
    endif()
  else()
    install(TARGETS ${the_target} RUNTIME DESTINATION ${OPENCV_BIN_INSTALL_PATH} COMPONENT libs)
  endif()
endif()
