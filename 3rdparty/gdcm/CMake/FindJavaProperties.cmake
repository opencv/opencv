#  Copyright (c) 2006-2011 Mathieu Malaterre <mathieu.malaterre@gmail.com>
#
#  Redistribution and use is allowed according to the terms of the New
#  BSD license.
#  For details see the accompanying COPYING-CMAKE-SCRIPTS file.
#

# This module will populate the following cmake variables:
# JavaProp_JAVA_LIBRARY_PATH
# JavaProp_OS_ARCH
# JavaProp_OS_NAME
# JavaProp_JAVA_HOME
# JavaProp_SUN_BOOT_LIBRARY_PATH
# JavaProp_PATH_SEPARATOR
# JavaProp_SUN_ARCH_DATA_MODEL

# I can't get FindJNI.cmake to work, so instead re-write one more robust
# which only requires javac and java being in the PATH

get_filename_component(current_list_path ${CMAKE_CURRENT_LIST_FILE} PATH)
find_package(Java 1.5 REQUIRED)

# need to re-run everytime the setting for Java has changed:
# There is technically one caveat still, when one only modify
# Java_JAVA_EXECUTABLE from cmake-gui, everything is re-run properly except the
# FIND_PATH for jar and javac
if(JavaProp_JAVA_HOME)
  get_filename_component(javarealpath
    ${Java_JAVA_EXECUTABLE}
    REALPATH
    )
  get_filename_component(javahomesubdir
    ${JavaProp_JAVA_HOME}
    PATH
    )
  #string(FIND "${javarealpath}" "${javahomesubdir}" res)
  #if(-1 EQUAL ${res})
  #  message(STATUS "Need to re-execute JavaProp")
  #  file(REMOVE
  #    ${CMAKE_BINARY_DIR}/GetSystemProperty.class
  #    )
  #endif()
  string(REGEX MATCH "${javahomesubdir}"
    outputvar
    "${javarealpath}"
    )
  if(NOT outputvar)
    message(STATUS "Need to re-execute JavaProp: ${outputvar}")
    file(REMOVE
      ${CMAKE_BINARY_DIR}/GetSystemProperty.class
      )
  endif()
endif()

# For some reason I have to use two execute_process instead of a chained one...
if(${current_list_path}/GetSystemProperty.java IS_NEWER_THAN ${CMAKE_BINARY_DIR}/GetSystemProperty.class)
  #message("${current_list_path}/GetSystemProperty.java")
  #message("${CMAKE_CURRENT_BINARY_DIR}/GetSystemProperty.class")
  execute_process(
    COMMAND ${Java_JAVAC_EXECUTABLE} -source 1.5 -target 1.5
    ${current_list_path}/GetSystemProperty.java -d ${CMAKE_BINARY_DIR}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )

  # populate the following list of java properties into CMake properties:
  set(JAVA_PROPERTY_LIST
    java.library.path
    os.arch
    os.name
    java.home
    sun.boot.library.path
    path.separator # : / ;
    sun.arch.data.model # 32 / 64
    )
  foreach(property ${JAVA_PROPERTY_LIST})
    string(TOUPPER ${property} property_upper)
    string(REPLACE "." "_" property_cmake_name ${property_upper})
    execute_process(
      COMMAND ${Java_JAVA_EXECUTABLE} GetSystemProperty ${property}
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      OUTPUT_VARIABLE ${property_cmake_name}
      OUTPUT_STRIP_TRAILING_WHITESPACE
      )
    #message("${property} : ${property_cmake_name} : ${${property_cmake_name}}")
    set(JavaProp_${property_cmake_name} ${${property_cmake_name}}
      CACHE STRING "Java Prop Value for: ${property}" FORCE
      )
    mark_as_advanced(
      JavaProp_${property_cmake_name}
      )
  endforeach()
endif()
