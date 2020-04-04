# --- Intel librealsense ---

if(NOT HAVE_LIBREALSENSE)
  find_package(realsense2 QUIET)
  if(realsense2_FOUND)
    set(HAVE_LIBREALSENSE TRUE)
    set(LIBREALSENSE_VERSION "${realsense2_VERSION}" PARENT_SCOPE) # informational
    ocv_add_external_target(librealsense "" "${realsense2_LIBRARY}" "HAVE_LIBREALSENSE")
  endif()
endif()

if(NOT HAVE_LIBREALSENSE)
  find_path(LIBREALSENSE_INCLUDE_DIR "librealsense2/rs.hpp"
    PATHS "${LIBREALSENSE_INCLUDE}" ENV LIBREALSENSE_INCLUDE)
  find_library(LIBREALSENSE_LIBRARIES "realsense2"
    PATHS "${LIBREALSENSE_LIB}" ENV LIBREALSENSE_LIB)
  if(LIBREALSENSE_INCLUDE_DIR AND LIBREALSENSE_LIBRARIES)
    set(HAVE_LIBREALSENSE TRUE)
    file(STRINGS "${LIBREALSENSE_INCLUDE_DIR}/librealsense2/rs.h" ver_strings REGEX "#define +RS2_API_(MAJOR|MINOR|PATCH|BUILD)_VERSION.*")
    string(REGEX REPLACE ".*RS2_API_MAJOR_VERSION[^0-9]+([0-9]+).*" "\\1" ver_major "${ver_strings}")
    string(REGEX REPLACE ".*RS2_API_MINOR_VERSION[^0-9]+([0-9]+).*" "\\1" ver_minor "${ver_strings}")
    string(REGEX REPLACE ".*RS2_API_PATCH_VERSION[^0-9]+([0-9]+).*" "\\1" ver_patch "${ver_strings}")
    set(LIBREALSENSE_VERSION "${ver_major}.${ver_minor}.${ver_patch}" PARENT_SCOPE) # informational
    ocv_add_external_target(librealsense "${LIBREALSENSE_INCLUDE_DIR}" "${LIBREALSENSE_LIBRARIES}" "HAVE_LIBREALSENSE")
  endif()
endif()

set(HAVE_LIBREALSENSE ${HAVE_LIBREALSENSE} PARENT_SCOPE)
