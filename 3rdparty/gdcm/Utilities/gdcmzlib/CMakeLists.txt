cmake_minimum_required(VERSION 2.8.9)

if(NOT ZLIB_NAMESPACE)
  set(ZLIB_NAMESPACE "ZLIB")
  set(ZLIB_STANDALONE 1)
endif()
# In all cases:
string(TOLOWER ${ZLIB_NAMESPACE} ZLIB_LIBRARY_NAME)

project(${ZLIB_NAMESPACE} C)

# Do full dependency headers.
include_regular_expression("^.*$")

# source files for zlib
set(ZLIB_SRCS
  adler32.c
  compress.c
  crc32.c
  deflate.c
  #example.c
  gzio.c
  infback.c
  inffast.c
  inflate.c
  inftrees.c
  #minigzip.c
  trees.c
  uncompr.c
  zutil.c
  )


# for windows add the .def and .rc files to the source list if building shared libs
if(WIN32)
  if(BUILD_SHARED_LIBS)
    set(ZLIB_DLL 1)
    if(NOT UNIX)
      if(NOT BORLAND)
        if(NOT MINGW)
          set(ZLIB_SRCS ${ZLIB_SRCS} zlib.def zlib.rc  )
        endif()
      endif() 
    endif()   
  endif()
endif()

if(UNIX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()

  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/zconf.in.h
    ${CMAKE_CURRENT_BINARY_DIR}/zconf.h
  )
include_directories(
 ${CMAKE_CURRENT_BINARY_DIR}
)


add_library(${ZLIB_LIBRARY_NAME} ${ZLIB_SRCS})
set_target_properties(${ZLIB_LIBRARY_NAME} PROPERTIES ${GDCM_LIBRARY_PROPERTIES})
set_target_properties(${ZLIB_LIBRARY_NAME} PROPERTIES LINKER_LANGUAGE C)

if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(${ZLIB_LIBRARY_NAME} PROPERTIES FOLDER "3rdparty")
endif()

if(NOT BUILD_SHARED_LIBS)
  ocv_install_target(${ZLIB_LIBRARY_NAME} EXPORT OpenCVModules ARCHIVE DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev)
endif()

if(NOT ZLIB_INSTALL_NO_LIBRARIES)
  install(TARGETS ${ZLIB_LIBRARY_NAME}
    EXPORT ${GDCM_TARGETS_NAME}
    RUNTIME DESTINATION ${ZLIB_INSTALL_BIN_DIR} COMPONENT Applications
    LIBRARY DESTINATION ${ZLIB_INSTALL_LIB_DIR} COMPONENT Libraries
    ARCHIVE DESTINATION ${ZLIB_INSTALL_LIB_DIR} COMPONENT DebugDevel
  ${CPACK_NAMELINK_TYPE}
  )
endif()
