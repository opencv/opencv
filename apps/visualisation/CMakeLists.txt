SET(OPENCV_VISUALISATION_DEPS opencv_core opencv_highgui opencv_imgproc)
ocv_check_dependencies(${OPENCV_VISUALISATION_DEPS})

if(NOT OCV_DEPENDENCIES_FOUND)
   return()
endif()

project(visualisation)

ocv_include_directories("${CMAKE_CURRENT_SOURCE_DIR}" "${OpenCV_SOURCE_DIR}/include/opencv")
ocv_include_modules(${OPENCV_VISUALISATION_DEPS})

set(visualisation_files opencv_visualisation.cpp)
set(the_target opencv_visualisation)

add_executable(${the_target} ${visualisation_files})
target_link_libraries(${the_target} ${OPENCV_VISUALISATION_DEPS})

set_target_properties(${the_target} PROPERTIES
                      DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
                      ARCHIVE_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}
                      RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
                      OUTPUT_NAME "opencv_visualisation")

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
