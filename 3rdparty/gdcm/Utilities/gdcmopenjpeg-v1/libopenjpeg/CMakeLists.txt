include_regular_expression("^.*$")
# Defines the source code for the library
set(OPENJPEG_SRCS
  bio.c
  cio.c
  dwt.c
  event.c
  image.c
  j2k.c
  j2k_lib.c
  jp2.c
  jpt.c
  mct.c
  mqc.c
  openjpeg.c
  pi.c
  raw.c
  t1.c
  t2.c
  tcd.c
  tgt.c
)

# Build the library
if(WIN32)
  if(BUILD_SHARED_LIBS)
    add_definitions(-DOPJ_EXPORTS)
  else()
    add_definitions(-DOPJ_STATIC)
  endif()
endif()

if(UNIX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()

add_library(${OPENJPEG_LIBRARY_NAME} ${OPENJPEG_SRCS})
set_target_properties(${OPENJPEG_LIBRARY_NAME} PROPERTIES
  ${OPENJPEG_LIBRARY_PROPERTIES} LINK_INTERFACE_LIBRARIES "")
if(UNIX)
  target_link_libraries(${OPENJPEG_LIBRARY_NAME} m)
endif()

if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(${OPENJPEG_LIBRARY_NAME} PROPERTIES FOLDER "3rdparty")
endif()

if(NOT BUILD_SHARED_LIBS)
  ocv_install_target(${OPENJPEG_LIBRARY_NAME} EXPORT OpenCVModules ARCHIVE DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev)
endif()

# Install library
if(NOT OPENJPEG_INSTALL_NO_LIBRARIES)
  install(TARGETS ${OPENJPEG_LIBRARY_NAME}
    EXPORT ${GDCM_TARGETS_NAME}
    RUNTIME DESTINATION ${OPENJPEG_INSTALL_BIN_DIR} COMPONENT Applications
    LIBRARY DESTINATION ${OPENJPEG_INSTALL_LIB_DIR} COMPONENT Libraries
    ARCHIVE DESTINATION ${OPENJPEG_INSTALL_LIB_DIR} COMPONENT DebugDevel
    )
endif()

# Install includes files
if(NOT OPENJPEG_INSTALL_NO_DEVELOPMENT)
install(FILES openjpeg.h
  #DESTINATION ${OPENJPEG_INSTALL_INCLUDE_DIR}/${subdir} COMPONENT Headers
  DESTINATION ${OPENJPEG_INSTALL_INCLUDE_DIR} COMPONENT Headers
)
#install(CODE
#  "execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${OPENJPEG_INSTALL_INCLUDE_DIR}/${subdir}/openjpeg.h \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${OPENJPEG_INSTALL_INCLUDE_DIR}/openjpeg.h)")

# install man page of the library
#install(
#  FILES       ../doc/man/man3/libopenjpeg.3
#  DESTINATION ${OPENJPEG_INSTALL_MAN_DIR}/man3)
endif()
