# Check if getopt is present:
include (${CMAKE_ROOT}/Modules/CheckIncludeFile.cmake)
set(DONT_HAVE_GETOPT 1)
if(UNIX) #I am pretty sure only *nix sys have this anyway
  CHECK_INCLUDE_FILE("getopt.h" CMAKE_HAVE_GETOPT_H)
  # Seems like we need the contrary:
  if(CMAKE_HAVE_GETOPT_H)
    set(DONT_HAVE_GETOPT 0)
  endif()
endif()

if(DONT_HAVE_GETOPT)
  add_definitions(-DDONT_HAVE_GETOPT)
endif()

