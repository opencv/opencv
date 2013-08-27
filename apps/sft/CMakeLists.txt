set(name sft)
set(the_target opencv_${name})

set(OPENCV_${the_target}_DEPS opencv_core opencv_softcascade opencv_highgui opencv_imgproc opencv_ml)
ocv_check_dependencies(${OPENCV_${the_target}_DEPS})

if(NOT OCV_DEPENDENCIES_FOUND)
  return()
endif()

project(${the_target})

ocv_include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include" "${OpenCV_SOURCE_DIR}/include/opencv")
ocv_include_modules(${OPENCV_${the_target}_DEPS})

file(GLOB ${the_target}_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_executable(${the_target} ${${the_target}_SOURCES})

target_link_libraries(${the_target} ${OPENCV_${the_target}_DEPS})

set_target_properties(${the_target} PROPERTIES
                      DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
                      ARCHIVE_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}
                      RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
                      INSTALL_NAME_DIR lib
                      OUTPUT_NAME "opencv_trainsoftcascade")

if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(${the_target} PROPERTIES FOLDER "applications")
endif()

install(TARGETS ${the_target} RUNTIME DESTINATION bin COMPONENT main)
