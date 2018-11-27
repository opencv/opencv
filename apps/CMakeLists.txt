add_definitions(-D__OPENCV_BUILD=1)
add_definitions(-D__OPENCV_APPS=1)

# Unified function for creating OpenCV applications:
#   ocv_add_application(tgt [MODULES <m1> [<m2> ...]] SRCS <src1> [<src2> ...])
function(ocv_add_application the_target)
  cmake_parse_arguments(APP "" "" "MODULES;SRCS" ${ARGN})
  ocv_check_dependencies(${APP_MODULES})
  if(NOT OCV_DEPENDENCIES_FOUND)
     return()
  endif()

  project(${the_target})
  ocv_target_include_modules_recurse(${the_target} ${APP_MODULES})
  ocv_target_include_directories(${the_target} PRIVATE "${OpenCV_SOURCE_DIR}/include/opencv")
  ocv_add_executable(${the_target} ${APP_SRCS})
  ocv_target_link_libraries(${the_target} ${APP_MODULES})
  set_target_properties(${the_target} PROPERTIES
                        DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
                        ARCHIVE_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}
                        RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
                        OUTPUT_NAME "${the_target}")

  if(ENABLE_SOLUTION_FOLDERS)
    set_target_properties(${the_target} PROPERTIES FOLDER "applications")
  endif()

  if(INSTALL_CREATE_DISTRIB)
    if(BUILD_SHARED_LIBS)
      install(TARGETS ${the_target} RUNTIME DESTINATION ${OPENCV_BIN_INSTALL_PATH} CONFIGURATIONS Release COMPONENT dev)
    endif()
  else()
    install(TARGETS ${the_target} RUNTIME DESTINATION ${OPENCV_BIN_INSTALL_PATH} COMPONENT dev)
  endif()
endfunction()

link_libraries(${OPENCV_LINKER_LIBS})

macro(ocv_add_app directory)
  if(DEFINED BUILD_APPS_LIST)
    list(FIND BUILD_APPS_LIST ${directory} _index)
    if (${_index} GREATER -1)
      add_subdirectory(${directory})
    else()
      message(STATUS "Skip OpenCV app: ${directory}")
    endif()
  else()
    add_subdirectory(${directory})
  endif()
endmacro()

#ocv_add_app(traincascade)
#ocv_add_app(createsamples)
ocv_add_app(annotation)
ocv_add_app(visualisation)
ocv_add_app(interactive-calibration)
ocv_add_app(version)
