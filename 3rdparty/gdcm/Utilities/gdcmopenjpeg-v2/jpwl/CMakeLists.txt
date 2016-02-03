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

add_library(openjpeg_JPWL ${JPWL_SRCS} ${OPJ_SRCS})

# Do the proper thing when building static...if only there was configured
# headers or def files instead
#if(NOT BUILD_SHARED_LIBS)
#  add_definitions(-DOPJ_STATIC)
#endif()

include_directories(
  ${OPENJPEG_SOURCE_DIR}/libopenjpeg
  )

find_package(TIFF REQUIRED)

add_executable(JPWL_j2k_to_image
../codec/convert.c ../codec/j2k_to_image.c
)
target_link_libraries(JPWL_j2k_to_image openjpeg_JPWL ${TIFF_LIBRARIES})
if(UNIX)
  target_link_libraries(JPWL_j2k_to_image m)
endif()

add_executable(JPWL_image_to_j2k
../codec/convert.c ../codec/image_to_j2k.c)
target_link_libraries(JPWL_image_to_j2k openjpeg_JPWL ${TIFF_LIBRARIES})
if(UNIX)
  target_link_libraries(JPWL_image_to_j2k m)
endif()
