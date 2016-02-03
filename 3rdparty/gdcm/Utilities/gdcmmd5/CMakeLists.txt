cmake_minimum_required(VERSION 2.8.9)

if(NOT MD5_NAMESPACE)
  set(MD5_NAMESPACE "MD5")
  set(MD5_STANDALONE 1)
endif()
# In all cases:
string(TOLOWER ${MD5_NAMESPACE} MD5_LIBRARY_NAME)

project(${MD5_NAMESPACE} C)

# Do full dependency headers.
include_regular_expression("^.*$")

set(MD5_SOURCES
  md5.c
  )

if (WIN32)
  if (BUILD_SHARED_LIBS)
    set(MD5DLL 1)
  else ()
    set(MD5STATIC 1)
  endif ()
endif ()

# TODO
# Optimized function needs to define:
# ARCH_IS_BIG_ENDIAN
# Big endian thing:
#include (${CMAKE_ROOT}/Modules/TestBigEndian.cmake)
#TEST_BIG_ENDIAN(MD5_WORDS_BIGENDIAN)
#set_source_files_properties(${MD5_SOURCES}
#  PROPERTIES COMPILE_FLAGS -DARCH_IS_BIG_ENDIAN
#  )
set(MANGLE_PREFIX ${MD5_LIBRARY_NAME})
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/md5_mangle.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/md5_mangle.h
               @ONLY )

# for md5_mangle.h
include_directories(BEFORE ${CMAKE_CURRENT_BINARY_DIR})

add_library(${MD5_LIBRARY_NAME} ${MD5_SOURCES})

if(GDCM_LIBRARY_PROPERTIES)
  set_target_properties(${MD5_LIBRARY_NAME} PROPERTIES ${GDCM_LIBRARY_PROPERTIES})
endif()

if(NOT MD5_INSTALL_BIN_DIR)
  set(MD5_INSTALL_BIN_DIR "bin")
endif()
if(NOT MD5_INSTALL_LIB_DIR)
  set(MD5_INSTALL_LIB_DIR "lib")
endif()

if(NOT MD5_INSTALL_NO_LIBRARIES)
  install(TARGETS ${MD5_LIBRARY_NAME}
    EXPORT ${GDCM_TARGETS_NAME}
    RUNTIME DESTINATION ${MD5_INSTALL_BIN_DIR} COMPONENT Applications
    LIBRARY DESTINATION ${MD5_INSTALL_LIB_DIR} COMPONENT Libraries
    ARCHIVE DESTINATION ${MD5_INSTALL_LIB_DIR} COMPONENT DebugDevel
  ${CPACK_NAMELINK_TYPE}
  )
endif()

if(MD5_STANDALONE)
  # md5main
  add_executable(md5main md5main.c)
  target_link_libraries(md5main ${MD5_LIBRARY_NAME})
  if(UNIX)
    target_link_libraries(md5main -lm)
  endif()
  # md5cmp
  add_executable(md5cmp md5cmp.c)
  target_link_libraries(md5cmp ${MD5_LIBRARY_NAME})
  # tst2md5
  add_executable(tst2md5 tst2md5.c)
  target_link_libraries(tst2md5 ${MD5_LIBRARY_NAME})
  install(TARGETS md5main md5cmp tst2md5
    RUNTIME DESTINATION ${MD5_INSTALL_BIN_DIR} COMPONENT Applications
    LIBRARY DESTINATION ${MD5_INSTALL_LIB_DIR} COMPONENT Libraries
    ARCHIVE DESTINATION ${MD5_INSTALL_LIB_DIR} COMPONENT DebugDevel
  ${CPACK_NAMELINK_TYPE}
  )

endif()

