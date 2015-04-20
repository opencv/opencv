# Makefile for the main JPWL OpenJPEG codecs: JPWL_ j2k_to_image and JPWL_image_to_j2k

add_definitions(-DUSE_JPWL)

set(OPJ_SRCS
../libopenjpeg/bio.c
../libopenjpeg/cio.c
../libopenjpeg/dwt.c
../libopenjpeg/event.c
../libopenjpeg/image.c
../libopenjpeg/j2k.c
../libopenjpeg/j2k_lib.c
../libopenjpeg/jp2.c
../libopenjpeg/jpt.c
../libopenjpeg/mct.c
../libopenjpeg/mqc.c
../libopenjpeg/openjpeg.c
../libopenjpeg/pi.c
../libopenjpeg/raw.c
../libopenjpeg/t1.c
../libopenjpeg/t2.c
../libopenjpeg/tcd.c
../libopenjpeg/tgt.c
)
set(JPWL_SRCS crc.c jpwl.c jpwl_lib.c rs.c)

set(common_SRCS "")
# If not getopt was found then add it to the lib:
if(DONT_HAVE_GETOPT)
  set(common_SRCS ${OPENJPEG_SOURCE_DIR}/common/getopt.c)
endif()

# Build the library
if(WIN32)
  if(BUILD_SHARED_LIBS)
    add_definitions(-DOPJ_EXPORTS)
  else()
    add_definitions(-DOPJ_STATIC)
  endif()
endif()
add_library(${OPENJPEG_LIBRARY_NAME}_JPWL ${JPWL_SRCS} ${OPJ_SRCS})
set_target_properties(${OPENJPEG_LIBRARY_NAME}_JPWL PROPERTIES ${OPENJPEG_LIBRARY_PROPERTIES})

# Install library
install(TARGETS ${OPENJPEG_LIBRARY_NAME}_JPWL
  DESTINATION ${OPENJPEG_INSTALL_LIB_DIR} COMPONENT Libraries
)

# Build executables

include_directories(
  ${OPENJPEG_SOURCE_DIR}/libopenjpeg
  ${OPENJPEG_SOURCE_DIR}/common
  )

if(LCMS_INCLUDE_DIR)
  include_directories( ${LCMS_INCLUDE_DIR} )
endif()
if(PNG_FOUND)
  include_directories(${PNG_INCLUDE_DIR})
endif()
if(TIFF_FOUND)
  include_directories(${TIFF_INCLUDE_DIR})
endif()


add_executable(JPWL_j2k_to_image
../codec/j2k_to_image.c
../codec/convert.c
../codec/index.c
${OPENJPEG_SOURCE_DIR}/common/color.c
${common_SRCS}
)

target_link_libraries(JPWL_j2k_to_image ${OPENJPEG_LIBRARY_NAME}_JPWL ${LCMS_LIB})
if(PNG_FOUND)
  target_link_libraries(JPWL_j2k_to_image ${PNG_LIBRARIES})
endif()
if(TIFF_FOUND)
  target_link_libraries(JPWL_j2k_to_image ${TIFF_LIBRARIES})
endif()
if(UNIX)
  target_link_libraries(JPWL_j2k_to_image m)
endif()


add_executable(JPWL_image_to_j2k
../codec/convert.c
../codec/index.c
../codec/image_to_j2k.c
${common_SRCS}
)

target_link_libraries(JPWL_image_to_j2k ${OPENJPEG_LIBRARY_NAME}_JPWL ${LCMS_LIB})
if(PNG_FOUND)
  target_link_libraries(JPWL_image_to_j2k ${PNG_LIBRARIES})
endif()
if(TIFF_FOUND)
  target_link_libraries(JPWL_image_to_j2k ${TIFF_LIBRARIES})
endif()
if(UNIX)
  target_link_libraries(JPWL_image_to_j2k m)
endif()

install(TARGETS JPWL_image_to_j2k JPWL_j2k_to_image
  DESTINATION ${OPENJPEG_INSTALL_BIN_DIR} COMPONENT Binaries
)
