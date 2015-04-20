# common
set(VERSION "1.12.12")
set(PACKAGE "socket++")

#set(CMAKE_REQUIRED_INCLUDES "string.h")
include(CheckFunctionExists)
CHECK_FUNCTION_EXISTS(strsignal SOCKETXX_HAVE_STRSIGNAL)

# specific
if(WIN32)
else()
  # FIXME this should be try_compile:
  set(EXTERN_C_BEGIN)
  set(SYS_SIGLIST sys_siglist)
  set(SIGHND_ARGTYPE int)
  set(SYS_ERRLIST sys_errlist)
  set(SYS_ERRLIST_DECLARED 1)
  set(_S_LIBGXX 0)
  set(HAVE_DLFCN_H 1)
  set(HAVE_FORK 1)
  set(HAVE_MEMORY_H 1)
  set(HAVE_SELECT 1)
  set(HAVE_SSTREAM 1)
  set(HAVE_STDINT_H 1)
  set(HAVE_STDLIB_H 1)
  set(HAVE_STRING 1)
  set(HAVE_STRINGS_H 1)
  set(HAVE_STRING_H 1)
  set(HAVE_SYS_STAT_H 1)
  set(HAVE_SYS_TYPES_H 1)
  set(HAVE_SYS_WAIT_H 1)
  set(HAVE_UNISTD_H 1)
  set(HAVE_VFORK 1)
  set(HAVE_VFORK_H 0)
  set(HAVE_WORKING_FORK 1)
  set(HAVE_WORKING_VFORK 1)
  set(RETSIGTYPE void)
  set(STDC_HEADERS 1)
endif()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/config.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/config.h
  )

include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_BINARY_DIR}
	)

set(libsocket___la_SOURCES
  sockstream.cpp sockstream.h
  sockinet.cpp sockinet.h
  #pipestream.cpp pipestream.h
  fork.cpp fork.h
  protocol.cpp protocol.h
  echo.cpp echo.h
  #smtp.cpp smtp.h
  #ftp.cpp ftp.h
  #sig.cpp sig.h
  )

if(UNIX)
  set(libsocket___la_SOURCES
    ${libsocket___la_SOURCES}
    #sockunix.cpp sockunix.h
    )
endif()

add_library(${SOCKETXX_LIBRARY_NAME} ${libsocket___la_SOURCES})
set_target_properties(${SOCKETXX_LIBRARY_NAME} PROPERTIES ${SOCKETXX_LIBRARY_PROPERTIES} LINK_INTERFACE_LIBRARIES "")
if(WIN32)
  target_link_libraries(${SOCKETXX_LIBRARY_NAME} ws2_32.lib)
endif()
if(CMAKE_SYSTEM MATCHES "SunOS.*")
  target_link_libraries(${SOCKETXX_LIBRARY_NAME} socket nsl)
endif()

if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(${SOCKETXX_LIBRARY_NAME} PROPERTIES FOLDER "3rdparty")
endif()

# Install library
if(NOT SOCKETXX_INSTALL_NO_LIBRARIES)
  # Runtime
  install(TARGETS ${SOCKETXX_LIBRARY_NAME}
    EXPORT ${GDCM_TARGETS_NAME}
    RUNTIME DESTINATION ${SOCKETXX_INSTALL_BIN_DIR} COMPONENT Applications
    LIBRARY DESTINATION ${SOCKETXX_INSTALL_LIB_DIR} COMPONENT Libraries ${NAMELINK_SKIP}
    ARCHIVE DESTINATION ${SOCKETXX_INSTALL_LIB_DIR} COMPONENT DebugDevel
  )
  #Development
  if(NAMELINK_ONLY)
    install(TARGETS ${SOCKETXX_LIBRARY_NAME}
    EXPORT ${GDCM_TARGETS_NAME}
      LIBRARY DESTINATION ${SOCKETXX_INSTALL_LIB_DIR} COMPONENT DebugDevel ${NAMELINK_ONLY}
    )
  endif()
endif()

if(NOT SOCKETXX_INSTALL_NO_DEVELOPMENT)
  file(GLOB header_files "*.h")
  install(FILES ${header_files}
    DESTINATION ${SOCKETXX_INSTALL_INCLUDE_DIR} COMPONENT Headers
  )
endif()

