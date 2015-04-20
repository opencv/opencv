#
# This module is provided as GDCM_USE_FILE by GDCMConfig.cmake.
# It can be INCLUDEd in a project to load the needed compiler and linker
# settings to use GDCM:
#   find_package(GDCM REQUIRED)
#   include(${GDCM_USE_FILE})

if(NOT GDCM_USE_FILE_INCLUDED)
  set(GDCM_USE_FILE_INCLUDED 1)

  # Add include directories needed to use GDCM.
  include_directories(${GDCM_INCLUDE_DIRS})

  # Add link directories needed to use GDCM.
  link_directories(${GDCM_LIBRARY_DIRS})

  # Add cmake module path.
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${GDCM_CMAKE_DIR}")

  # Use VTK.
  if(GDCM_USE_VTK)
    set(VTK_DIR ${GDCM_VTK_DIR})
    find_package(VTK)
    if(VTK_FOUND)
      include(${VTK_USE_FILE})
    else()
      message("VTK not found in GDCM_VTK_DIR=\"${GDCM_VTK_DIR}\".")
    endif()
  endif()

endif()
