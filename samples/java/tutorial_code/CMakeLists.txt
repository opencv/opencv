# ----------------------------------------------------------------------------
#  CMake file for Java tutorials compilation.
#
# ----------------------------------------------------------------------------
if(NOT ANT_EXECUTABLE OR NOT TARGET opencv_java)
  return()
endif()

project(compile_java_tutorials)

set(curdir "${CMAKE_CURRENT_SOURCE_DIR}")
set(opencv_tutorial_java_bin_dir "${CMAKE_CURRENT_BINARY_DIR}/.compiled")
set(TUTORIALS_DIRS "")

file(GLOB children RELATIVE ${curdir} ${curdir}/*/*)
foreach(child ${children})
  if(IS_DIRECTORY ${curdir}/${child})
    file(GLOB contains_java_files "${child}/*.java")
    if(contains_java_files)
      list(APPEND TUTORIALS_DIRS ${child})
    endif()
  endif()
endforeach()

add_custom_target("${PROJECT_NAME}"
                  DEPENDS opencv_java
                 )

foreach(TUTORIAL_DIR ${TUTORIALS_DIRS})
  get_filename_component(TUTORIAL_NAME ${TUTORIAL_DIR} NAME_WE)
  add_custom_command(TARGET "${PROJECT_NAME}"
                     COMMAND ${ANT_EXECUTABLE} -q
                          -DocvJarDir="${OpenCV_BINARY_DIR}/bin"
                          -DsrcDir="${TUTORIAL_DIR}"
                          -DdstDir="${opencv_tutorial_java_bin_dir}/${TUTORIAL_NAME}"
                     WORKING_DIRECTORY "${curdir}"
                     COMMENT "Compile the tutorial: ${TUTORIAL_NAME}"
                    )
endforeach()
