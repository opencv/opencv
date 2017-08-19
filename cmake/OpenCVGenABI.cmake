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

set(OPENCV_ABI_SKIP_HEADERS "")
set(OPENCV_ABI_SKIP_LIBRARIES "")
foreach(mod ${OPENCV_MODULES_BUILD})
  string(REGEX REPLACE "^opencv_" "" mod "${mod}")
  if(NOT OPENCV_MODULE_opencv_${mod}_CLASS STREQUAL "PUBLIC"
      OR NOT "${OPENCV_MODULE_opencv_${mod}_LOCATION}" STREQUAL "${OpenCV_SOURCE_DIR}/modules/${mod}" # opencv_contrib
  )
    # headers
    foreach(h ${OPENCV_MODULE_opencv_${mod}_HEADERS})
      file(RELATIVE_PATH h "${OPENCV_MODULE_opencv_${mod}_LOCATION}/include" "${h}")
      list(APPEND OPENCV_ABI_SKIP_HEADERS "${h}")
    endforeach()
    # libraries
    if(TARGET opencv_${mod}) # opencv_world
      list(APPEND OPENCV_ABI_SKIP_LIBRARIES "\$<TARGET_FILE_NAME:opencv_${mod}>")
    endif()
  endif()
endforeach()
string(REPLACE ";" "\n    " OPENCV_ABI_SKIP_HEADERS "${OPENCV_ABI_SKIP_HEADERS}")
string(REPLACE ";" "\n    " OPENCV_ABI_SKIP_LIBRARIES "${OPENCV_ABI_SKIP_LIBRARIES}")

# Options
set(OPENCV_ABI_GCC_OPTIONS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} -DOPENCV_ABI_CHECK=1")
string(REGEX REPLACE "([^ ]) +([^ ])" "\\1\\n    \\2" OPENCV_ABI_GCC_OPTIONS "${OPENCV_ABI_GCC_OPTIONS}")

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/templates/opencv_abi.xml.in" "${path1}.base")
file(GENERATE OUTPUT "${path1}" INPUT "${path1}.base")
