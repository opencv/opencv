# Build the demo app, small examples

# First thing define the common source:

# Then check if getopt is present:
include (${CMAKE_ROOT}/Modules/CheckIncludeFile.cmake)

# Headers file are located here:
include_directories(
  ${OPENJPEG_SOURCE_DIR}/libopenjpeg
  )

# Do the proper thing when building static...if only there was configured
# headers or def files instead
if(NOT BUILD_SHARED_LIBS)
  add_definitions(-DOPJ_STATIC)
endif()


# Loop over all executables:
foreach(exe test_encoder test_decoder)
  add_executable(${exe} ${exe}.c)
  target_link_libraries(${exe} ${OPJ_PREFIX}openjpeg)
endforeach()
