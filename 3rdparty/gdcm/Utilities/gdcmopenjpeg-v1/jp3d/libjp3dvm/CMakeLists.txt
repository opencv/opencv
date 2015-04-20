if(UNIX OR CYGWIN)
 add_definitions(-O3)
elseif(MINGW)
 add_definitions(-Os)
elseif(WIN32)
  add_definitions(/Os)
endif()
#
include_regular_expression("^.*$")
# Defines the source code for the library
set(JP3DVM_SRCS
bio.c  cio.c  dwt.c  event.c  jp3d.c  jp3d_lib.c  mct.c  mqc.c  openjpeg.c  pi.c  raw.c  t1.c  t1_3d.c  t2.c  tcd.c  tgt.c  volume.c
)

# Build the library
if(WIN32)
  if(BUILD_SHARED_LIBS)
    add_definitions(-DOPJ_EXPORTS)
  else()
    add_definitions(-DOPJ_STATIC)
  endif()
endif()
add_library(${OPENJPEG_LIBRARY_NAME}_JP3D ${JP3DVM_SRCS})
set_target_properties(${OPENJPEG_LIBRARY_NAME}_JP3D
  PROPERTIES
    VERSION   1.3.0
    SOVERSION 1)

# Install library
install(TARGETS ${OPENJPEG_LIBRARY_NAME}_JP3D
DESTINATION ${OPENJPEG_INSTALL_LIB_DIR} COMPONENT Libraries
)

# Install includes files
install(FILES openjpeg3d.h
  DESTINATION ${OPENJPEG_INSTALL_INCLUDE_DIR}/openjpeg3d-1.3
)
install(CODE
  "execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${OPENJPEG_INSTALL_INCLUDE_DIR}/openjpeg3d-1.3/openjpeg3d.h \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${OPENJPEG_INSTALL_INCLUDE_DIR}/openjpeg3d.h)")
