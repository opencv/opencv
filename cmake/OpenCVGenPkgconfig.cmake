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

if(CMAKE_BUILD_TYPE MATCHES "Release")
  set(ocv_optkind OPT)
else()
  set(ocv_optkind DBG)
endif()

#build the list of opencv libs and dependencies for all modules
set(OpenCV_LIB_COMPONENTS "")
set(OpenCV_EXTRA_COMPONENTS "")
foreach(m ${OPENCV_MODULES_PUBLIC})
  list(INSERT OpenCV_LIB_COMPONENTS 0 ${${m}_MODULE_DEPS_${ocv_optkind}} ${m})
  if(${m}_EXTRA_DEPS_${ocv_optkind})
    list(INSERT OpenCV_EXTRA_COMPONENTS 0 ${${m}_EXTRA_DEPS_${ocv_optkind}})
  endif()
endforeach()

ocv_list_unique(OpenCV_LIB_COMPONENTS)
ocv_list_unique(OpenCV_EXTRA_COMPONENTS)
ocv_list_reverse(OpenCV_LIB_COMPONENTS)
ocv_list_reverse(OpenCV_EXTRA_COMPONENTS)

#build the list of components

# Note:
#   when linking against static libraries, if libfoo depends on libbar, then
#   libfoo must come first in the linker flags.

# world is a special target whose library should come first, especially for
# static link.
if(OpenCV_LIB_COMPONENTS MATCHES "opencv_world")
  list(REMOVE_ITEM OpenCV_LIB_COMPONENTS "opencv_world")
  list(INSERT OpenCV_LIB_COMPONENTS 0 "opencv_world")
endif()

set(OpenCV_LIB_COMPONENTS_)
foreach(CVLib ${OpenCV_LIB_COMPONENTS})

  get_target_property(libloc ${CVLib} LOCATION_${CMAKE_BUILD_TYPE})
  if(libloc MATCHES "3rdparty")
    set(libpath "\${exec_prefix}/share/OpenCV/3rdparty/${OPENCV_LIB_INSTALL_PATH}")
  else()
    set(libpath "\${exec_prefix}/${OPENCV_LIB_INSTALL_PATH}")
  endif()
  list(APPEND OpenCV_LIB_COMPONENTS_ "-L${libpath}")

  get_filename_component(libname ${CVLib} NAME_WE)
  string(REGEX REPLACE "^lib" "" libname "${libname}")
  list(APPEND OpenCV_LIB_COMPONENTS_ "-l${libname}")

endforeach()

# add extra dependencies required for OpenCV
if(OpenCV_EXTRA_COMPONENTS)
  foreach(extra_component ${OpenCV_EXTRA_COMPONENTS})
    if(TARGET "${extra_component}")
      get_target_property(extra_component_is_imported "${extra_component}" IMPORTED)
      if(extra_component_is_imported)
        get_target_property(extra_component "${extra_component}" LOCATION)
      endif()
    endif()

    if(extra_component MATCHES "^-l")
      list(APPEND OpenCV_LIB_COMPONENTS_ "${extra_component}")
    elseif(extra_component MATCHES "[\\/]")
      get_filename_component(libdir "${extra_component}" PATH)
      get_filename_component(libname "${extra_component}" NAME_WE)
      string(REGEX REPLACE "^lib" "" libname "${libname}")
      list(APPEND OpenCV_LIB_COMPONENTS_ "-L${libdir}" "-l${libname}")
    else()
      list(APPEND OpenCV_LIB_COMPONENTS_ "-l${extra_component}")
    endif()
  endforeach()
endif()

list(REMOVE_DUPLICATES OpenCV_LIB_COMPONENTS_)
string(REPLACE ";" " " OpenCV_LIB_COMPONENTS "${OpenCV_LIB_COMPONENTS_}")

#generate the .pc file
set(prefix      "${CMAKE_INSTALL_PREFIX}")
set(exec_prefix "\${prefix}")
set(libdir      "\${exec_prefix}/${OPENCV_LIB_INSTALL_PATH}")
set(includedir  "\${prefix}/${OPENCV_INCLUDE_INSTALL_PATH}")

if(INSTALL_TO_MANGLED_PATHS)
  set(OPENCV_PC_FILE_NAME "opencv-${OPENCV_VERSION}.pc")
else()
  set(OPENCV_PC_FILE_NAME opencv.pc)
endif()
configure_file("${OpenCV_SOURCE_DIR}/cmake/templates/opencv-XXX.pc.in"
               "${CMAKE_BINARY_DIR}/unix-install/${OPENCV_PC_FILE_NAME}"
               @ONLY)

if(UNIX AND NOT ANDROID)
  install(FILES ${CMAKE_BINARY_DIR}/unix-install/${OPENCV_PC_FILE_NAME} DESTINATION ${OPENCV_LIB_INSTALL_PATH}/pkgconfig COMPONENT dev)
endif()
