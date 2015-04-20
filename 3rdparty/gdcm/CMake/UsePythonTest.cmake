# Add a python test from a python file
# One cannot simply do:
# set(ENV{PYTHONPATH} ${LIBRARY_OUTPUT_PATH})
# set(my_test "from test_mymodule import *\;test_mymodule()")
# add_test(PYTHON-TEST-MYMODULE  python -c ${my_test})
# Since cmake is only transmitting the ADD_TEST line to ctest thus you are loosing
# the env var. The only way to store the env var is to physically write in the cmake script
# whatever PYTHONPATH you want and then add the test as 'cmake -P python_test.cmake'
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
find_package(PythonInterp REQUIRED)
mark_as_advanced(PYTHON_EXECUTABLE)

macro(ADD_PYTHON_TEST TESTNAME FILENAME)
  get_source_file_property(loc ${FILENAME} LOCATION)
  get_source_file_property(pyenv ${FILENAME} PYTHONPATH)
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
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${TESTNAME}.cmake
"
  set(ENV{PYTHONPATH} ${pyenv}:\$ENV{PYTHONPATH})
  set(ENV{LD_LIBRARY_PATH} ${pyenv}:\$ENV{LD_LIBRARY_PATH})
  message(\"${pyenv}\")
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} ${loc} ${wo_semicolumn}
    #WORKING_DIRECTORY @LIBRARY_OUTPUT_PATH@
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
macro(ADD_PYTHON_COMPILEALL_TEST DIRNAME)
  # First get the path:
  get_filename_component(temp_path "${PYTHON_LIBRARIES}" PATH)
  # Find the python script:
  get_filename_component(PYTHON_COMPILE_ALL_PY "${temp_path}/../compileall.py" ABSOLUTE)
  # add test, use DIRNAME to create uniq name for the test:
  add_test(COMPILE_ALL-${DIRNAME} ${PYTHON_EXECUTABLE} "${PYTHON_COMPILE_ALL_PY}" -q ${DIRNAME})
endmacro()
