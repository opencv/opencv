cmake_minimum_required(VERSION 2.8.9)

if(NOT EXPAT_NAMESPACE)
  set(EXPAT_NAMESPACE "EXPAT")
  set(EXPAT_STANDALONE 1)
endif()
# In all cases:
string(TOLOWER ${EXPAT_NAMESPACE} EXPAT_LIBRARY_NAME)

project(${EXPAT_NAMESPACE} C)

set(EXPAT_MAJOR_VERSION 2)
set(EXPAT_MINOR_VERSION 0)
set(EXPAT_BUILD_VERSION 0)
set(EXPAT_VERSION
  "${EXPAT_MAJOR_VERSION}.${EXPAT_MINOR_VERSION}.${EXPAT_BUILD_VERSION}")

set(EXPAT_LIBRARY_PROPERTIES ${EXPAT_LIBRARY_PROPERTIES}
  VERSION "${EXPAT_VERSION}"
  SOVERSION "${EXPAT_MAJOR_VERSION}.${EXPAT_MINOR_VERSION}"
)

# Do full dependency headers.
include_regular_expression("^.*$")

#if(WIN32)
#  if(BUILD_SHARED_LIBS)
#    set(EXPATDLL 1)
#  else()
#    set (EXPATSTATIC 1)
#  endif()
#endif()

#-----------------------------------------------------------------------------
include (${CMAKE_ROOT}/Modules/CheckIncludeFile.cmake)
# FIXME not the right way of doing it:
CHECK_INCLUDE_FILE("string.h"       HAVE_MEMMOVE)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/expat_config.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/expat_config.h)
set(MANGLE_PREFIX ${EXPAT_LIBRARY_NAME})
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/expat_mangle.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/expat_mangle.h
               @ONLY )

# for expat_config.h
include_directories(BEFORE ${CMAKE_CURRENT_BINARY_DIR})
subdirs(lib)

