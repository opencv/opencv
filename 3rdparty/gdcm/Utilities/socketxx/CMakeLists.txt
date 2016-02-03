cmake_minimum_required(VERSION 2.8.9)
if(POLICY CMP0022)
  cmake_policy(SET CMP0022 OLD)
endif()

# http://www.linuxhacker.at/socketxx
if(NOT SOCKETXX_NAMESPACE)
  set(SOCKETXX_NAMESPACE "SOCKETXX")
  set(SOCKETXX_STANDALONE 1)
endif()
# In all cases:
string(TOLOWER ${SOCKETXX_NAMESPACE} SOCKETXX_LIBRARY_NAME)

project(${SOCKETXX_NAMESPACE} CXX)

if(NOT SOCKETXX_INSTALL_BIN_DIR)
  set(SOCKETXX_INSTALL_BIN_DIR "bin")
endif()
if(NOT SOCKETXX_INSTALL_LIB_DIR)
  set(SOCKETXX_INSTALL_LIB_DIR "lib")
endif()
if(NOT SOCKETXX_INSTALL_INCLUDE_DIR)
  set(SOCKETXX_INSTALL_INCLUDE_DIR "include")
endif()

# configure.in:LIBSOCKET_SO_VERSION
# LIBSOCKET_SO_VERSION=1:2:0
# SOCKET_VERSION=1.12.12

#-----------------------------------------------------------------------------
# SOCKETXX version number
set(SOCKETXX_MAJOR_VERSION 1)
set(SOCKETXX_MINOR_VERSION 2)
set(SOCKETXX_BUILD_VERSION 0)
set(SOCKETXX_VERSION
  "${SOCKETXX_MAJOR_VERSION}.${SOCKETXX_MINOR_VERSION}.${SOCKETXX_BUILD_VERSION}"
  )
set(SOCKETXX_API_VERSION
  "${SOCKETXX_MAJOR_VERSION}.${SOCKETXX_MINOR_VERSION}"
  )
set(SOCKETXX_LIBRARY_PROPERTIES
  VERSION "${SOCKETXX_VERSION}"
  SOVERSION "${SOCKETXX_API_VERSION}"
  )

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  )

#option(BUILD_SHARED_LIBS "Build with shared libraries." ON)

add_subdirectory(socket++)
#add_subdirectory(test)  #these just don't work on windows


