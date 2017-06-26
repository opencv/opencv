# ----------------------------------------------------------------------------
#  CMake file for Java tutorials compilation.
#
# ----------------------------------------------------------------------------
if(NOT ANT_EXECUTABLE)
  return()
endif()

project(compile_java_tutorials)

set(curdir "${CMAKE_CURRENT_SOURCE_DIR}")
set(opencv_tutorial_java_bin_dir "${CMAKE_CURRENT_BINARY_DIR}/.compiled")
set(target_name "compile_java_tutorials")
set(TUTORIALS_DIRS "")

FILE(GLOB children RELATIVE ${curdir} ${curdir}/*/*)
FOREACH(child ${children})
  if(IS_DIRECTORY ${curdir}/${child})
    FILE(GLOB contains_java_files "${child}/*.java")
    if(contains_java_files)
      LIST(APPEND TUTORIALS_DIRS ${child})
    endif()
  endif()
ENDFOREACH()

add_custom_target("${target_name}" ALL
                  DEPENDS opencv_java
                 )

FOREACH(tutorial_dir ${TUTORIALS_DIRS})
  get_filename_component(tutorial_name ${tutorial_dir} NAME_WE)
  add_custom_command(TARGET "${target_name}"
                     COMMAND ${ANT_EXECUTABLE} -q
                          -DocvJarDir="${OpenCV_BINARY_DIR}/bin"
                          -DsrcDir="${tutorial_dir}"
                          -DdstDir="${opencv_tutorial_java_bin_dir}/${tutorial_name}"
                     WORKING_DIRECTORY "${curdir}"
                     COMMENT "Compile the tutorial: ${tutorial_name}"
                    )
ENDFOREACH()
