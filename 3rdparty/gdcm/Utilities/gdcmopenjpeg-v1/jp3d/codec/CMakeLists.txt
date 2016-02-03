# Build the demo app, small examples

# First thing define the common source:
set(common_SRCS
  convert.c
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
    getopt.c
  )
endif()

if(WIN32)
  if(BUILD_SHARED_LIBS)
    add_definitions(-DOPJ_EXPORTS)
  else()
    add_definitions(-DOPJ_STATIC)
  endif()
endif()

# Loop over all executables:
foreach(exe jp3d_to_volume volume_to_jp3d)
  add_executable(${exe} ${exe}.c ${common_SRCS})
  target_link_libraries(${exe} ${OPENJPEG_LIBRARY_NAME}_JP3D) # ${TIFF_LIBRARIES})
  # On unix you need to link to the math library:
  if(UNIX)
    target_link_libraries(${exe} m)
  endif()
  # Install exe
  install_targets(/bin/ ${exe})
endforeach()


