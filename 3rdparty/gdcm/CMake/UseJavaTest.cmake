# Add a java test from a java file
#
# Usage:
# set_source_files_properties(test.py PROPERTIES PYTHONPATH
#   "${LIBRARY_OUTPUT_PATH}:${VTK_DIR}")
# ADD_PYTHON_TEST(PYTHON-TEST test.py)
#
#  Copyright (c) 2006-2011 Mathieu Malaterre <mathieu.malaterre@gmail.com>
#
#  Redistribution and use is allowed according to the terms of the New
#  BSD license.
#  For details see the accompanying COPYING-CMAKE-SCRIPTS file.
#

# Need python interpreter:
#find_package(PythonInterp REQUIRED)
#mark_as_advanced(PYTHON_EXECUTABLE)
# UseCSharp.cmake

macro(ADD_JAVA_TEST TESTNAME FILENAME)
  get_source_file_property(loc ${FILENAME}.class LOCATION)
  get_source_file_property(pyenv ${FILENAME}.class RUNTIMEPATH)
  get_source_file_property(theclasspath ${FILENAME}.class CLASSPATH)
  get_filename_component(loc2 ${loc} NAME_WE)


  if(CMAKE_CONFIGURATION_TYPES)
    # I cannot use CMAKE_CFG_INTDIR since it expand to "$(OutDir)"
    if(pyenv)
      set(pyenv "${pyenv};${LIBRARY_OUTPUT_PATH}/${CMAKE_BUILD_TYPE}")
    else()
      set(pyenv ${LIBRARY_OUTPUT_PATH}/${CMAKE_BUILD_TYPE})
      #set(pyenv ${LIBRARY_OUTPUT_PATH}/${CMAKE_CFG_INTDIR})
      #set(pyenv ${LIBRARY_OUTPUT_PATH}/${CMAKE_CONFIG_TYPE})
      #set(pyenv ${LIBRARY_OUTPUT_PATH}/\${CMAKE_CONFIG_TYPE})
    endif()
  else()
    if(pyenv)
      set(pyenv ${pyenv}:${LIBRARY_OUTPUT_PATH})
    else()
      set(pyenv ${LIBRARY_OUTPUT_PATH})
    endif()
   endif()
  string(REGEX REPLACE ";" " " wo_semicolumn "${ARGN}")

  set(classpath)
  if(theclasspath)
    set(classpath "${theclasspath}${JavaProp_PATH_SEPARATOR}.")
  else()
    set(classpath ".")
  endif()
  set(theld_library_path $ENV{LD_LIBRARY_PATH})
  set(ld_library_path)
  if(theld_library_path)
    set(ld_library_path ${theld_library_path})
  endif()
  if(pyenv)
    set(ld_library_path ${ld_library_path}:${pyenv})
  endif()

  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${TESTNAME}.cmake
"
  if(UNIX)
  set(ENV{LD_LIBRARY_PATH} ${ld_library_path})
  set(ENV{DYLD_LIBRARY_PATH} ${ld_library_path})
  #set(ENV{CLASSPATH} ${pyenv}/gdcm.jar:.)
  message(\"pyenv: ${pyenv}\")
  else()
  #set(the_path $ENV{PATH})
  set(ENV{PATH} "${ld_library_path}")
  endif()
  message(\"loc: ${loc}\")
  message(\"loc2: ${loc2}\")
  message(\"classpath: ${classpath}\")
  message(\"java runtime: ${Java_JAVA_EXECUTABLE}\")
  #message( \"wo_semicolumn: ${wo_semicolumn}\" )
  execute_process(
    COMMAND ${Java_JAVA_EXECUTABLE} -classpath \"${classpath}\" ${loc2} ${wo_semicolumn}
    WORKING_DIRECTORY \"${EXECUTABLE_OUTPUT_PATH}\"
    RESULT_VARIABLE import_res
    OUTPUT_VARIABLE import_output
    ERROR_VARIABLE  import_output
  )

  # Pass the output back to ctest
  if(import_output)
    message("\${import_output}")
  endif()
  if(import_res)
    message(SEND_ERROR "\${import_res}")
  endif()
"
)
  add_test(NAME ${TESTNAME} COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/${TESTNAME}.cmake)
endmacro()

# Byte compile recursively a directory (DIRNAME)
#macro(ADD_PYTHON_COMPILEALL_TEST DIRNAME)
#  # First get the path:
#  get_filename_component(temp_path "${PYTHON_LIBRARIES}" PATH)
#  # Find the python script:
#  get_filename_component(PYTHON_COMPILE_ALL_PY "${temp_path}/../compileall.py" ABSOLUTE)
#  # add test, use DIRNAME to create uniq name for the test:
#  add_test(COMPILE_ALL-${DIRNAME} ${PYTHON_EXECUTABLE} "${PYTHON_COMPILE_ALL_PY}" -q ${DIRNAME})
#endmacro()
#
