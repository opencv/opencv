include_regular_expression("^.*$")
# Defines the source code for the library
set(OPENJPEG_SRCS
  tcd.c
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
  tgt.c
  profile.c
  invert.c
  function_list.c
  t2.c
  bio.h
  cio.h
  dwt.h
  event.h
  image.h
  j2k.h
  j2k_lib.h
  jp2.h
  jpt.h
  mct.h
  mqc.h
  openjpeg.h
  pi.h
  int.h
  raw.h
  t1.h
  t2.h
  tcd.h
  tgt.h
  profile.h
  invert.h
  function_list.h
)

# Pass proper definition to preprocessor to generate shared lib
if(WIN32)
  if(BUILD_SHARED_LIBS)
    add_definitions(-DOPJ_EXPORTS)
  else()
    add_definitions(-DOPJ_STATIC)
  endif()
endif()

if(ENABLE_PROFILING)
    add_definitions(-D_PROFILE)
endif()

# Create the library
add_library(${OPENJPEG_LIBRARY_NAME} ${OPENJPEG_SRCS})
set_target_properties(${OPENJPEG_LIBRARY_NAME} PROPERTIES
  ${OPENJPEG_LIBRARY_PROPERTIES} LINK_INTERFACE_LIBRARIES "")
if(UNIX)
  target_link_libraries(${OPENJPEG_LIBRARY_NAME} m)
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
  file(GLOB header_files "*.h")
  install(FILES ${header_files}
    DESTINATION ${OPENJPEG_INSTALL_INCLUDE_DIR} COMPONENT Headers
    )
endif()
#install(FILES openjpeg.h
#	DESTINATION ${OPENJPEG_INSTALL_INCLUDE_DIR} COMPONENT Headers
#)
