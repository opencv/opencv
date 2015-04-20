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

#include(${OPENJPEG_SOURCE_DIR}/CMake/Free_CMakeImport.cmake)
add_definitions ( -DFREEIMAGE_LIB )

# Loop over all executables:
foreach(exe test2_encoder test2_decoder)
  add_executable(${exe} ${exe}.c)
  target_link_libraries(${exe} ${OPJ_PREFIX}openjpeg)
  target_link_libraries(${exe} ${FREEIMAGE_LIBRARIES})
endforeach()
