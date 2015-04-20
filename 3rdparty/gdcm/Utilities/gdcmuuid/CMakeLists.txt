cmake_minimum_required(VERSION 2.8.9)

if(NOT UUID_NAMESPACE)
  set(UUID_NAMESPACE "UUID")
  set(UUID_STANDALONE 1)
endif()
# In all cases:
string(TOLOWER ${UUID_NAMESPACE} UUID_LIBRARY_NAME)

project(${UUID_NAMESPACE} C)

# Do full dependency headers.
include_regular_expression("^.*$")

set(UUID_SRCS
compare.c
gen_uuid.c
pack.c
parse.c
unpack.c
unparse.c
uuid_time.c
uuid.h
uuidP.h
)

#include (${CMAKE_ROOT}/Modules/CheckIncludeFile.cmake)

if(WIN32)
  if (BUILD_SHARED_LIBS)
    add_definitions(-DUUID_DLL)
  endif ()
endif()

if(COMMAND CHECK_INCLUDE_FILE_CONCAT)
else()
include (${CMAKE_ROOT}/Modules/CheckIncludeFile.cmake)
include (${CMAKE_ROOT}/Modules/CheckIncludeFiles.cmake)

macro(CHECK_INCLUDE_FILE_CONCAT FILE VARIABLE)
  CHECK_INCLUDE_FILES("${UUID_INCLUDES};${FILE}" ${VARIABLE})
  if(${VARIABLE})
    set(UUID_INCLUDES ${UUID_INCLUDES} ${FILE})
  endif()
endmacro()
endif()

CHECK_INCLUDE_FILE_CONCAT("stdlib.h"       HAVE_STDLIB_H)
if(UNIX) #Avoid polluting Win32 cmakecache
  CHECK_INCLUDE_FILE_CONCAT("unistd.h"       HAVE_UNISTD_H)
  CHECK_INCLUDE_FILE_CONCAT("inttypes.h"     HAVE_INTTYPES_H)
  CHECK_INCLUDE_FILE_CONCAT("sys/ioctl.h"    HAVE_SYS_IOCTL_H)
  CHECK_INCLUDE_FILE_CONCAT("sys/time.h"     HAVE_SYS_TIME_H)
  CHECK_INCLUDE_FILE_CONCAT("sys/file.h"     HAVE_SYS_FILE_H)
  CHECK_INCLUDE_FILE_CONCAT("sys/socket.h"   HAVE_SYS_SOCKET_H)
  CHECK_INCLUDE_FILE_CONCAT("sys/sockio.h"   HAVE_SYS_SOCKIO_H)
  CHECK_INCLUDE_FILE_CONCAT("net/if.h"       HAVE_NET_IF_H)
  CHECK_INCLUDE_FILE_CONCAT("netinet/in.h"   HAVE_NETINET_IN_H)
  CHECK_INCLUDE_FILE_CONCAT("net/if_dl.h"    HAVE_NET_IF_DL_H)
  CHECK_INCLUDE_FILE_CONCAT("net/if_arp.h"   HAVE_NET_IF_ARP_H)
endif()
if(WIN32) #Avoid polluting UNIX cmakecache
  CHECK_INCLUDE_FILE_CONCAT("winsock.h"       HAVE_WINSOCK_H)
endif()

set(UUID_HAVES_ALL
  HAVE_STDLIB_H
  HAVE_UNISTD_H
  HAVE_INTTYPES_H
  HAVE_SYS_IOCTL_H
  #HAVE_SYS_TIME_H
  HAVE_SYS_FILE_H
  HAVE_SYS_SOCKET_H
  HAVE_SYS_SOCKIO_H
  HAVE_NET_IF_H
  HAVE_NETINET_IN_H
  HAVE_NET_IF_DL_H
  HAVE_NET_IF_ARP_H
  HAVE_WINSOCK_H
)

set(UUID_HAVES)
foreach(have ${UUID_HAVES_ALL})
  if(${${have}})
    set(UUID_HAVES "${UUID_HAVES} -D${have}")
  endif()
endforeach()
# Pass all -D :
if( UUID_HAVES )
set_source_files_properties( ${UUID_SRCS}
  PROPERTIES COMPILE_FLAGS ${UUID_HAVES})
endif()

set(MANGLE_PREFIX ${UUID_LIBRARY_NAME})
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/uuid_mangle.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/uuid_mangle.h
               @ONLY )

# for uuid_mangle.h
include_directories(BEFORE ${CMAKE_CURRENT_BINARY_DIR})

add_library(${UUID_LIBRARY_NAME} ${UUID_SRCS})
if(GDCM_LIBRARY_PROPERTIES)
set_target_properties(${UUID_LIBRARY_NAME} PROPERTIES ${GDCM_LIBRARY_PROPERTIES})
endif()
if(WIN32 AND NOT CYGWIN)
  target_link_libraries(${UUID_LIBRARY_NAME} iphlpapi)
endif()

if(NOT UUID_INSTALL_BIN_DIR)
  set(UUID_INSTALL_BIN_DIR "bin")
endif()
if(NOT UUID_INSTALL_LIB_DIR)
  set(UUID_INSTALL_LIB_DIR "lib")
endif()

if(NOT UUID_INSTALL_NO_LIBRARIES)
  install(TARGETS ${UUID_LIBRARY_NAME}
    EXPORT ${GDCM_TARGETS_NAME}
    RUNTIME DESTINATION ${UUID_INSTALL_BIN_DIR} COMPONENT Applications
    LIBRARY DESTINATION ${UUID_INSTALL_LIB_DIR} COMPONENT Libraries
    ARCHIVE DESTINATION ${UUID_INSTALL_LIB_DIR} COMPONENT DebugDevel
  ${CPACK_NAMELINK_TYPE}
  )
endif()
