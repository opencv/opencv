SET(OPENCV_VERSION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/modules/core/include/opencv2/core/version.hpp")
FILE(STRINGS "${OPENCV_VERSION_FILE}" OPENCV_VERSION_PARTS REGEX "#define CV_.+OR_VERSION[ ]+[0-9]+" )
string(REGEX REPLACE ".+CV_MAJOR_VERSION[ ]+([0-9]+).*" "\\1" OPENCV_VERSION_MAJOR "${OPENCV_VERSION_PARTS}")
string(REGEX REPLACE ".+CV_MINOR_VERSION[ ]+([0-9]+).*" "\\1" OPENCV_VERSION_MINOR "${OPENCV_VERSION_PARTS}")
string(REGEX REPLACE ".+CV_SUBMINOR_VERSION[ ]+([0-9]+).*" "\\1" OPENCV_VERSION_PATCH "${OPENCV_VERSION_PARTS}")
set(OPENCV_VERSION "${OPENCV_VERSION_MAJOR}.${OPENCV_VERSION_MINOR}.${OPENCV_VERSION_PATCH}")

set(OPENCV_SOVERSION "${OPENCV_VERSION_MAJOR}.${OPENCV_VERSION_MINOR}")

# create a dependency on version file
# we never use output of the following command but cmake will rerun automatically if the version file changes
configure_file("${OPENCV_VERSION_FILE}" "${CMAKE_BINARY_DIR}/junk/version.junk" COPYONLY)

if(WIN32)
  # Postfix of DLLs:
  set(OPENCV_DLLVERSION "${OPENCV_VERSION_MAJOR}${OPENCV_VERSION_MINOR}${OPENCV_VERSION_PATCH}")
  set(OPENCV_DEBUG_POSTFIX d)
else()
  # Postfix of so's:
  set(OPENCV_DLLVERSION "")
  set(OPENCV_DEBUG_POSTFIX)
endif()

#name mangling
set(OPENCV_INCLUDE_PREFIX include)
if(INSTALL_TO_MANGLED_PATHS)
  set(OPENCV_INCLUDE_PREFIX include/opencv-${OPENCV_VERSION})
endif()
