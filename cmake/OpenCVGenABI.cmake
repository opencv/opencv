if (NOT GENERATE_ABI_DESCRIPTOR)
  return()
endif()

set(filename "opencv_abi.xml")
set(path1 "${CMAKE_BINARY_DIR}/${filename}")

set(modules "${OPENCV_MODULES_PUBLIC}")
ocv_list_filterout(modules "opencv_ts")

message(STATUS "Generating ABI compliance checker configuration: ${filename}")

if (OPENCV_VCSVERSION AND NOT OPENCV_VCSVERSION STREQUAL "unknown")
  set(OPENCV_ABI_VERSION "${OPENCV_VCSVERSION}")
else()
  set(OPENCV_ABI_VERSION "${OPENCV_VERSION}")
endif()

# Headers
set(OPENCV_ABI_HEADERS "{RELPATH}/${OPENCV_INCLUDE_INSTALL_PATH}")

# Libraries
set(OPENCV_ABI_LIBRARIES "{RELPATH}/${OPENCV_LIB_INSTALL_PATH}")

# Options
set(OPENCV_ABI_GCC_OPTIONS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
string(REGEX REPLACE "([^ ]) +([^ ])" "\\1\\n    \\2" OPENCV_ABI_GCC_OPTIONS "${OPENCV_ABI_GCC_OPTIONS}")

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/templates/opencv_abi.xml.in" "${path1}")
