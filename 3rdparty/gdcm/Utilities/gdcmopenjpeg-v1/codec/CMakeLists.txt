  # Build the demo app, small examples

# First thing define the common source:
set(common_SRCS
  convert.c
  index.c
  ${${OPENJPEG_NAMESPACE}_SOURCE_DIR}/common/color.c
)

# If not getopt was found then add it to the lib:
if(DONT_HAVE_GETOPT)
  set(common_SRCS
    ${common_SRCS}
    ${OPENJPEG_SOURCE_DIR}/common/getopt.c
  )
endif()

# Headers file are located here:
include_directories(
  ${OPENJPEG_SOURCE_DIR}/libopenjpeg
  ${LCMS_INCLUDE_DIR}
  ${OPENJPEG_SOURCE_DIR}/common
  )
if(PNG_FOUND)
  include_directories(${PNG_INCLUDE_DIR})
endif()
if(TIFF_FOUND)
  include_directories(${TIFF_INCLUDE_DIR})
endif()

if(WIN32)
  if(BUILD_SHARED_LIBS)
    add_definitions(-DOPJ_EXPORTS)
  else()
    add_definitions(-DOPJ_STATIC)
  endif()
endif()

# Loop over all executables:
foreach(exe j2k_to_image image_to_j2k j2k_dump)
  add_executable(${exe} ${exe}.c ${common_SRCS})
  target_link_libraries(${exe} ${OPENJPEG_LIBRARY_NAME} ${LCMS_LIB})
  if(PNG_FOUND)
    target_link_libraries(${exe} ${PNG_LIBRARIES})
  endif()
  if(TIFF_FOUND)
    target_link_libraries(${exe} ${TIFF_LIBRARIES})
  endif()
  add_test(NAME ${exe} COMMAND ${exe})
  # calling those exe without option will make them fail always:
  set_tests_properties(${exe} PROPERTIES WILL_FAIL TRUE)
  # On unix you need to link to the math library:
  if(UNIX)
    target_link_libraries(${exe} m)
  endif()
  # Install exe
  install(TARGETS ${exe}
    EXPORT OpenJPEGTargets
    DESTINATION ${OPENJPEG_INSTALL_BIN_DIR} COMPONENT Applications
  )
endforeach()

# Install man pages
install(
  FILES       ../doc/man/man1/image_to_j2k.1
              ../doc/man/man1/j2k_dump.1
              ../doc/man/man1/j2k_to_image.1
  DESTINATION ${OPENJPEG_INSTALL_MAN_DIR}/man1)
#

if(BUILD_TESTING)
# Do testing here, once we know the examples are being built:
file(GLOB_RECURSE OPENJPEG_DATA_IMAGES_GLOB
  "${JPEG2000_CONFORMANCE_DATA_ROOT}/*.j2k"
  "${JPEG2000_CONFORMANCE_DATA_ROOT}/*.j2c"
  "${JPEG2000_CONFORMANCE_DATA_ROOT}/*.jp2"
  )

foreach(filename ${OPENJPEG_DATA_IMAGES_GLOB})
  get_filename_component(filename_temp ${filename} NAME)
  get_filename_component(filename_ext ${filename} EXT)
  execute_process(COMMAND j2k_dump -i ${filename}
    OUTPUT_VARIABLE dump_success
    OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/${filename_temp}.dump
    ERROR_QUIET
  )
  if(dump_success)
  file(READ ${CMAKE_CURRENT_BINARY_DIR}/${filename_temp}.dump numcomp_file)
  string(REGEX REPLACE ".*numcomps=([0-9]+).*" "\\1"
    numcomps "${numcomp_file}")
  #message( "found:${output_variable} for ${filename_temp}" )
  endif()
  add_test(NAME dump-${filename_temp} COMMAND j2k_dump -i ${filename})
  foreach(codec_type ppm pgx bmp tif raw tga png)
    add_test(NAME j2i-${filename_temp}-${codec_type} COMMAND j2k_to_image -i ${filename} -o ${filename_temp}.${codec_type})
    add_test(NAME i2j-${filename_temp}-${codec_type} COMMAND image_to_j2k -i ${filename_temp}.${codec_type} -o ${filename_temp}.${codec_type}${filename_ext})
    #if(UNIX)
    #  add_test(cmp-${filename_temp}-${codec_type} cmp ${filename} ${filename_temp}.${codec_type}${filename_ext})
    #endif()
  endforeach()
endforeach()
endif()
