cmake_minimum_required(VERSION 2.8.9)

if(NOT JPEG_NAMESPACE)
  set(JPEG_NAMESPACE "JPEG")
  set(JPEG_STANDALONE 1)
  set(JPEG_INSTALL_BIN_DIR bin)
  set(JPEG_INSTALL_LIB_DIR lib)
  set(JPEG_INSTALL_INCLUDE_DIR include/ljpeg-62)
  option(JPEG_BUILD_SHARED_LIBS "Build IJG/JPEG with shared libraries." OFF)
  set(BUILD_SHARED_LIBS ${JPEG_BUILD_SHARED_LIBS})
endif()
# In all cases:
string(TOLOWER ${JPEG_NAMESPACE} JPEG_LIBRARY_NAME)


project(${JPEG_NAMESPACE} C)

# Do full dependency headers.
include_regular_expression("^.*$")

# let advanced user the option to define JPEG_API_VERSION:
if(NOT DEFINED JPEG_API_VERSION)
  set(JPEG_API_VERSION "${GDCM_MAJOR_VERSION}.${GDCM_MINOR_VERSION}")
endif()
set(JPEG_LIBRARY_PROPERTIES ${JPEG_LIBRARY_PROPERTIES}
  VERSION "${GDCM_VERSION}"
  SOVERSION "${JPEG_API_VERSION}"
)


# memmgr back ends: compile only one of these into a working library
# (For now, let's use the mode that requires the image fit into memory.
# This is the recommended mode for Win32 anyway.)
set(systemdependent_SRCS jmemnobs.c)

# library object files common to compression and decompression
set(common_SRCS
jcomapi.c jutils.c jerror.c jmemmgr.c
#jmemsrc.c
)

# compression library object files
set(compression_SRCS
jcapimin.c jcapistd.c jctrans.c jcparam.c  jcinit.c
jcmaster.c jcmarker.c jcmainct.c jcprepct.c jccoefct.c jccolor.c
jcsample.c jchuff.c jcphuff.c jcdctmgr.c jfdctfst.c jfdctflt.c
jdatadst.c
)

# Lossy (DCT) codec:
set(comp_lossy_SRCS
jfdctint.c
jclossy.c

# lossless
jclossls.c

#jdarith.c
jcodec.c
jcscale.c
jcshuff.c
jclhuff.c
jcpred.c
jcdiffct.c
)

# decompression library object files
set(decompression_SRCS
jdapimin.c jdapistd.c jdtrans.c  jdmaster.c
jdinput.c jdmarker.c jdhuff.c jdphuff.c jdmainct.c jdcoefct.c
jdpostct.c jddctmgr.c jidctfst.c jidctflt.c jidctint.c jidctred.c
jdsample.c jdcolor.c jquant1.c jquant2.c jdmerge.c
jdatasrc.c
)

set(decomp_lossy_SRCS
 jdlossls.c jdlossy.c
 #jcarith.c
jdlhuff.c
jddiffct.c
jdpred.c
jdscale.c
jdshuff.c
)

set(JPEG_HEADER_FILES
  jchuff.h
  jdct.h
  jdhuff.h
  jerror.h
  jinclude.h
  jmemsys.h
  jmorecfg.h
  jpegint.h
  jpeglib.h
  jversion.h
  jlossy.h
  jlossls.h
  )

set(JPEG_SOURCES)
foreach(src
  ${systemdependent_SRCS}
  ${common_SRCS}
  ${compression_SRCS}
  ${decompression_SRCS}
  ${comp_lossy_SRCS}
  ${decomp_lossy_SRCS}
  )
  set(JPEG_SOURCES ${JPEG_SOURCES}
    "${CMAKE_CURRENT_SOURCE_DIR}/${src}")
endforeach()

if (WIN32)
  if (BUILD_SHARED_LIBS)
    set(JPEGDLL 1)
  else ()
    set(JPEGSTATIC 1)
  endif ()
endif ()

# By default support DICOMObjects bug, if this is a speed penalty for you
# you should consider disabling it
set(SUPPORT_DICOMOBJECTS_BUG ON)

# Generate three libraries: jpeg 8bits, 12bits and 16bits
# using only one source directory

#-----------------------------------------------------------------------------
include (${CMAKE_ROOT}/Modules/CheckIncludeFile.cmake)
CHECK_INCLUDE_FILE("stddef.h"       HAVE_STDDEF_H)
CHECK_INCLUDE_FILE("stdlib.h"       HAVE_STDLIB_H)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
# Particular case jpegcmake.h needs to be specified either 8, 12 or 16 bits:
# also configure the name mangling
foreach(bits 8 12 16)
  set(CMAKE_BITS_IN_JSAMPLE ${bits})
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/jpegcmake.h.in
                 ${CMAKE_CURRENT_BINARY_DIR}/${bits}/jpegcmake.h
                 @ONLY)
  set(MANGLE_PREFIX ${JPEG_LIBRARY_NAME}${bits})
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mangle_jpeg.h.in
                 ${CMAKE_CURRENT_BINARY_DIR}/${bits}/mangle_jpeg${bits}bits.h
                 @ONLY)


  # FIXME FIXME FIXME
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/jmorecfg.h
                 ${CMAKE_CURRENT_BINARY_DIR}/${bits}/jmorecfg.h
                 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/jconfig.h
                 ${CMAKE_CURRENT_BINARY_DIR}/${bits}/jconfig.h
                 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/jpeglib.h
                 ${CMAKE_CURRENT_BINARY_DIR}/${bits}/jpeglib.h
                 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/jerror.h
                 ${CMAKE_CURRENT_BINARY_DIR}/${bits}/jerror.h
                 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/jinclude.h
                 ${CMAKE_CURRENT_BINARY_DIR}/${bits}/jinclude.h
                 COPYONLY)
endforeach()

# I could not get the depencencies in CMake to work properly so instead
# have multiple directory for each target that will ensure proper recompilation
# of the objects files
subdirs(8 12 16)

if(NOT JPEG_INSTALL_NO_DEVELOPMENT)
  file(GLOB header_files "*.h")
  install(FILES ${header_files}
    DESTINATION ${JPEG_INSTALL_INCLUDE_DIR} COMPONENT Headers
  )
endif()
