# Build the demo app, small examples

# First thing define the common source:
set(common_SRCS
  convert.c
  index.c
)

# Then check if getopt is present:
include (${CMAKE_ROOT}/Modules/CheckIncludeFile.cmake)
set(DONT_HAVE_GETOPT 1)
if(UNIX) #I am pretty sure only *nix sys have this anyway
  CHECK_INCLUDE_FILE("getopt.h" CMAKE_HAVE_GETOPT_H)
  # Seems like we need the contrary:
  if(CMAKE_HAVE_GETOPT_H)
    set(DONT_HAVE_GETOPT 0)
  endif()
endif()

# If not getopt was found then add it to the lib:
if(DONT_HAVE_GETOPT)
  add_definitions(-DDONT_HAVE_GETOPT)
  set(common_SRCS
    ${common_SRCS}
    compat/getopt.c
  )
endif()

# Headers file are located here:
include_directories(
  ${OPENJPEG_SOURCE_DIR}/libopenjpeg
  )

# Do the proper thing when building static...if only there was configured
# headers or def files instead
if(NOT BUILD_SHARED_LIBS)
  add_definitions(-DOPJ_STATIC)
endif()

find_package(TIFF REQUIRED)
find_package(PNG REQUIRED)
include_directories( ${PNG_INCLUDE_DIR} )
include_directories( ${TIFF_INCLUDE_DIR} )

# Loop over all executables:
foreach(exe j2k_to_image image_to_j2k j2k_dump)
  add_executable(${exe} ${exe}.c ${common_SRCS})
  target_link_libraries(${exe} ${OPJ_PREFIX}openjpeg ${TIFF_LIBRARIES} ${PNG_LIBRARIES})
  add_test(NAME ${exe} COMMAND ${exe})
  # calling those exe without option will make them fail always:
  set_tests_properties(${exe} PROPERTIES WILL_FAIL TRUE)
  # On unix you need to link to the math library:
  if(UNIX)
    target_link_libraries(${exe} m)
  endif()
  # Install exe
  install(TARGETS ${exe}
    EXPORT ${GDCM_TARGETS_NAME}
    DESTINATION ${OPENJPEG_INSTALL_BIN_DIR} COMPONENT Applications
  )
endforeach()

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
