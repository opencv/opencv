# - C# module for CMake
# Defines the following macros:
#   CSHARP_ADD_EXECUTABLE(name [ files ])
#     - Define C# executable with given name
#   CSHARP_ADD_LIBRARY(name [ files ])
#     - Define C# library with given name
#   CSHARP_LINK_LIBRARIES(name [ libraries ])
#     - Link libraries to csharp library
#
#  Copyright (c) 2006-2011 Mathieu Malaterre <mathieu.malaterre@gmail.com>
#
#  Redistribution and use is allowed according to the terms of the New
#  BSD license.
#  For details see the accompanying COPYING-CMAKE-SCRIPTS file.
#

# TODO:
# http://www.cs.nuim.ie/~jpower/Research/csharp/Index.html

if(WIN32)
  include(${DotNETFrameworkSDK_USE_FILE})
  # remap
  set(CMAKE_CSHARP1_COMPILER ${CSC_v1_EXECUTABLE})
  set(CMAKE_CSHARP2_COMPILER ${CSC_v2_EXECUTABLE})
  set(CMAKE_CSHARP3_COMPILER ${CSC_v3_EXECUTABLE})
  set(CMAKE_CSHARP4_COMPILER ${CSC_v4_EXECUTABLE})

  #set(CMAKE_CSHARP3_INTERPRETER ${MONO_EXECUTABLE})
else()
  include(${MONO_USE_FILE})
  set(CMAKE_CSHARP1_COMPILER ${MCS_EXECUTABLE})
  set(CMAKE_CSHARP2_COMPILER ${GMCS_EXECUTABLE})
  set(CMAKE_CSHARP3_COMPILER ${SMCS_EXECUTABLE})
  set(CMAKE_CSHARP4_COMPILER ${SMCS_EXECUTABLE})

  set(CMAKE_CSHARP_INTERPRETER ${MONO_EXECUTABLE})
endif()

set(DESIRED_CSHARP_COMPILER_VERSION 2 CACHE STRING "Pick a version for C# compiler to use: 1, 2, 3 or 4")
mark_as_advanced(DESIRED_CSHARP_COMPILER_VERSION)

# default to v1:
if(DESIRED_CSHARP_COMPILER_VERSION MATCHES 1)
  set(CMAKE_CSHARP_COMPILER ${CMAKE_CSHARP1_COMPILER})
elseif(DESIRED_CSHARP_COMPILER_VERSION MATCHES 2)
  set(CMAKE_CSHARP_COMPILER ${CMAKE_CSHARP2_COMPILER})
elseif(DESIRED_CSHARP_COMPILER_VERSION MATCHES 3)
  set(CMAKE_CSHARP_COMPILER ${CMAKE_CSHARP3_COMPILER})
elseif(DESIRED_CSHARP_COMPILER_VERSION MATCHES 4)
  set(CMAKE_CSHARP_COMPILER ${CMAKE_CSHARP4_COMPILER})
else()
  message(FATAL_ERROR "Do not know this version")
endif()

# CMAKE_CSHARP_COMPILER /platform and anycpu
if(WIN32)
# There is a subttle issue when compiling on 64bits platform using a 32bits compiler
# See bug ID: 3510023 (BadImageFormatException: An attempt was made to load a progr)

set(CSC_ACCEPTS_PLATFORM_FLAG 0)

if(CMAKE_CSHARP_COMPILER)
  execute_process(COMMAND "${CMAKE_CSHARP_COMPILER}" "/?" OUTPUT_VARIABLE CSC_HELP)
  # when cmd locale is in French it displays: "/platform:<chaine>" in english: "/platform:<string>"
  # so only regex match in /platform:
  if("${CSC_HELP}" MATCHES "/platform:")
    set(CSC_ACCEPTS_PLATFORM_FLAG 1)
  endif()
endif()

if(NOT DEFINED CSC_PLATFORM_FLAG)
  set(CSC_PLATFORM_FLAG "")
  if(CSC_ACCEPTS_PLATFORM_FLAG)
    set(CSC_PLATFORM_FLAG "/platform:x86")
    if("${CMAKE_SIZEOF_VOID_P}" GREATER 4)
      set(CSC_PLATFORM_FLAG "/platform:x64")
    endif()
  endif()
endif()
endif()


# Check something is found:
if(NOT CMAKE_CSHARP_COMPILER)
  # status message only for now:
  message("Sorry C# v${DESIRED_CSHARP_COMPILER_VERSION} was not found on your system")
else()
  #if (NOT CSHARP_FIND_QUIETLY)
  message(STATUS "Will be using C# v${DESIRED_CSHARP_COMPILER_VERSION}: ${CMAKE_CSHARP_COMPILER}")
  #endif ()
endif()

macro(CSHARP_ADD_LIBRARY name)
  set(csharp_cs_sources)
  set(csharp_cs_sources_dep)
  foreach(it ${ARGN})
    if(EXISTS ${it})
      set(csharp_cs_sources "${csharp_cs_sources} ${it}")
      set(csharp_cs_sources_dep ${csharp_cs_sources_dep} ${it})
    else()
      if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${it})
        set(csharp_cs_sources "${csharp_cs_sources} ${CMAKE_CURRENT_SOURCE_DIR}/${it}")
        set(csharp_cs_sources_dep ${csharp_cs_sources_dep} ${CMAKE_CURRENT_SOURCE_DIR}/${it})
      else()
        #message("Could not find: ${it}")
        set(csharp_cs_sources "${csharp_cs_sources} ${it}")
      endif()
    endif()
  endforeach()

  #set(SHARP #)
  separate_arguments(csharp_cs_sources)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${name}.dll
    COMMAND ${CMAKE_CSHARP_COMPILER}
    ARGS "/t:library" "/out:${name}.dll" ${csharp_cs_sources}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS "${csharp_cs_sources_dep}"
    COMMENT "Creating Csharp library ${name}.cs"
  )
  add_custom_target(CSharp_${name} ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${name}.dll
  )
endmacro()

macro(CSHARP_ADD_EXECUTABLE name)
  set(csharp_cs_sources)
  foreach(it ${ARGN})
    if(EXISTS ${it})
      set(csharp_cs_sources "${csharp_cs_sources} ${it}")
    else()
      if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${it})
        set(csharp_cs_sources "${csharp_cs_sources} ${CMAKE_CURRENT_SOURCE_DIR}/${it}")
      else()
        #message("Could not find: ${it}")
        set(csharp_cs_sources "${csharp_cs_sources} ${it}")
      endif()
    endif()
  endforeach()

  set(CSHARP_EXECUTABLE_${name}_ARGS
    #"/out:${name}.dll" ${csharp_cs_sources}
    #"/r:gdcm_csharp.dll"
    "/out:${name}.exe ${csharp_cs_sources}"
  )

endmacro()

macro(CSHARP_LINK_LIBRARIES name)
  set(csharp_libraries)
  set(csharp_libraries_depends)
  foreach(it ${ARGN})
    #if(EXISTS ${it}.dll)
      set(csharp_libraries "${csharp_libraries} /r:${it}.dll")
    #  set(csharp_libraries_depends ${it}.dll)
    #else()
    #  if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/${it}.dll)
    #    set(csharp_libraries "${csharp_libraries} /r:${it}.dll")
    #    set(csharp_libraries_depends ${CMAKE_CURRENT_BINARY_DIR}/${it}.dll)
    #  else()
    #    message("Could not find: ${it}")
    #  endif()
    #endif()
  endforeach()
  set(CSHARP_EXECUTABLE_${name}_ARGS " ${csharp_libraries} ${CSHARP_EXECUTABLE_${name}_ARGS}")
  #message( "DEBUG: ${CSHARP_EXECUTABLE_${name}_ARGS}" )

  # BAD DESIGN !
  # This should be in the _ADD_EXECUTABLE...
  separate_arguments(CSHARP_EXECUTABLE_${name}_ARGS)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${name}.exe
    COMMAND ${CMAKE_CSHARP_COMPILER}
    #ARGS "/r:gdcm_csharp.dll" "/out:${name}.exe" ${csharp_cs_sources}
    ARGS ${CSHARP_EXECUTABLE_${name}_ARGS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    #DEPENDS ${csharp_cs_sources}
    COMMENT "Create HelloWorld.exe"
  )

  #message("DEBUG2:${csharp_libraries_depends}")
  add_custom_target(CSHARP_EXECUTABLE_${name} ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${name}.exe
            ${csharp_libraries_depends}
  )

endmacro()
