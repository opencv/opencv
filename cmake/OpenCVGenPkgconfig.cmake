if(MSVC OR IOS)
  return()
endif()

# --------------------------------------------------------------------------------------------
# according to man pkg-config
#  The package name specified on the pkg-config command line is defined to
#      be the name of the metadata file, minus the .pc extension. If a library
#      can install multiple versions simultaneously, it must give each version
#      its own name (for example, GTK 1.2 might have the package  name  "gtk+"
#      while GTK 2.0 has "gtk+-2.0").
#
# ${BIN_DIR}/unix-install/opencv.pc -> For use *with* "make install"
# -------------------------------------------------------------------------------------------

macro(fix_prefix lst isown)
  set(_lst)
  foreach(item ${${lst}})
    if(DEFINED TARGET_LOCATION_${item})
      set(item "${TARGET_LOCATION_${item}}")
      if(${isown})
        get_filename_component(item "${item}" NAME)
        ocv_get_libname(item "${item}")
      endif()
    endif()
    if(item MATCHES "^-l")
      list(APPEND _lst "${item}")
    elseif(item MATCHES "^-framework") # MacOS framework (assume single entry "-framework OpenCL")
      list(APPEND _lst "${item}")
    elseif(item MATCHES "[\\/]")
      get_filename_component(libdir "${item}" PATH)
      get_filename_component(_libname "${item}" NAME)
      ocv_get_libname(libname "${_libname}")
      list(APPEND _lst "-L${libdir}" "-l${libname}")
    else()
      list(APPEND _lst "-l${item}")
    endif()
  endforeach()
  set(${lst} ${_lst})
  unset(_lst)
endmacro()

if(NOT DEFINED CMAKE_HELPER_SCRIPT)

if(INSTALL_TO_MANGLED_PATHS)
  set(OPENCV_PC_FILE_NAME "opencv-${OPENCV_VERSION}.pc")
else()
  set(OPENCV_PC_FILE_NAME opencv.pc)
endif()

# build the list of opencv libs and dependencies for all modules
ocv_get_all_libs(_modules _extra _3rdparty)

#build the list of components

# Note:
#   when linking against static libraries, if libfoo depends on libbar, then
#   libfoo must come first in the linker flags.

# world and contrib_world are special targets whose library should come first,
# especially for static link.
if(_modules MATCHES "opencv_world")
  set(_modules "opencv_world")
endif()

if(_modules MATCHES "opencv_contrib_world")
  list(REMOVE_ITEM _modules "opencv_contrib_world")
  list(INSERT _modules 0 "opencv_contrib_world")
endif()

set(HELPER_SCRIPT "")
ocv_cmake_script_append_var(HELPER_SCRIPT
    BUILD_SHARED_LIBS
    CMAKE_BINARY_DIR
    CMAKE_INSTALL_PREFIX
    OpenCV_SOURCE_DIR
    OPENCV_PC_FILE_NAME
    OPENCV_VERSION_PLAIN
    OPENCV_LIB_INSTALL_PATH
    OPENCV_INCLUDE_INSTALL_PATH
    OPENCV_3P_LIB_INSTALL_PATH

    _modules
    _extra
    _3rdparty
)

foreach(item ${_modules} ${_extra} ${_3rdparty})
  if(TARGET ${item})
    set(HELPER_SCRIPT "${HELPER_SCRIPT}
set(TARGET_LOCATION_${item} \"$<TARGET_FILE:${item}>\")
")
  endif()
endforeach()

set(CMAKE_HELPER_SCRIPT "${CMAKE_BINARY_DIR}/OpenCVGenPkgConfig.info.cmake")
file(GENERATE OUTPUT "${CMAKE_HELPER_SCRIPT}" CONTENT "${HELPER_SCRIPT}")

add_custom_target(developer_scripts)
add_custom_command(
  OUTPUT "${CMAKE_BINARY_DIR}/unix-install/${OPENCV_PC_FILE_NAME}"
  COMMAND ${CMAKE_COMMAND} "-DCMAKE_HELPER_SCRIPT=${CMAKE_HELPER_SCRIPT}" -P "${OpenCV_SOURCE_DIR}/cmake/OpenCVGenPkgconfig.cmake"
  DEPENDS "${CMAKE_BINARY_DIR}/OpenCVGenPkgConfig.info.cmake"
          "${OpenCV_SOURCE_DIR}/cmake/OpenCVGenPkgconfig.cmake"
  COMMENT "Generate ${OPENCV_PC_FILE_NAME}"
)
add_custom_target(gen-pkgconfig ALL SOURCES "${CMAKE_BINARY_DIR}/unix-install/${OPENCV_PC_FILE_NAME}")
add_dependencies(developer_scripts gen-pkgconfig)


if(UNIX AND NOT ANDROID)
  install(FILES ${CMAKE_BINARY_DIR}/unix-install/${OPENCV_PC_FILE_NAME} DESTINATION ${OPENCV_LIB_INSTALL_PATH}/pkgconfig COMPONENT dev)
endif()

# =============================================================================
else() # DEFINED CMAKE_HELPER_SCRIPT

cmake_minimum_required(VERSION 2.8.12.2)
cmake_policy(SET CMP0012 NEW)
include("${CMAKE_HELPER_SCRIPT}")
include("${OpenCV_SOURCE_DIR}/cmake/OpenCVUtils.cmake")

fix_prefix(_modules 1)
fix_prefix(_extra 0)
fix_prefix(_3rdparty 1)

ocv_list_unique(_modules)
ocv_list_unique(_extra)
ocv_list_unique(_3rdparty)

set(OPENCV_PC_LIBS
  "-L\${exec_prefix}/${OPENCV_LIB_INSTALL_PATH}"
  "${_modules}"
)
if(BUILD_SHARED_LIBS)
  set(OPENCV_PC_LIBS_PRIVATE "${_extra}")
else()
  set(OPENCV_PC_LIBS_PRIVATE
    "-L\${exec_prefix}/${OPENCV_3P_LIB_INSTALL_PATH}"
    "${_3rdparty}"
    "${_extra}"
  )
endif()
string(REPLACE ";" " " OPENCV_PC_LIBS "${OPENCV_PC_LIBS}")
string(REPLACE ";" " " OPENCV_PC_LIBS_PRIVATE "${OPENCV_PC_LIBS_PRIVATE}")

#generate the .pc file
set(prefix      "${CMAKE_INSTALL_PREFIX}")
set(exec_prefix "\${prefix}")
set(libdir      "\${exec_prefix}/${OPENCV_LIB_INSTALL_PATH}")
set(includedir  "\${prefix}/${OPENCV_INCLUDE_INSTALL_PATH}")

configure_file("${OpenCV_SOURCE_DIR}/cmake/templates/opencv-XXX.pc.in"
               "${CMAKE_BINARY_DIR}/unix-install/${OPENCV_PC_FILE_NAME}"
               @ONLY)

endif() # DEFINED CMAKE_HELPER_SCRIPT
