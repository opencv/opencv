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
    if(TARGET ${item})
      get_target_property(item "${item}" LOCATION_${CMAKE_BUILD_TYPE})
      if("${isown}")
        get_filename_component(item "${item}" NAME_WE)
        string(REGEX REPLACE "^lib(.*)" "\\1" item "${item}")
      endif()
    endif()
    if(item MATCHES "^-l")
      list(APPEND _lst "${item}")
    elseif(item MATCHES "^-framework") # MacOS framework (assume single entry "-framework OpenCL")
      list(APPEND _lst "${item}")
    elseif(item MATCHES "[\\/]")
      get_filename_component(libdir "${item}" PATH)
      get_filename_component(libname "${item}" NAME_WE)
      string(REGEX REPLACE "^lib(.*)" "\\1" libname "${libname}")
      list(APPEND _lst "-L${libdir}" "-l${libname}")
    else()
      list(APPEND _lst "-l${item}")
    endif()
  endforeach()
  set(${lst} ${_lst})
  unset(_lst)
endmacro()

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

fix_prefix(_modules TRUE)
fix_prefix(_extra FALSE)
fix_prefix(_3rdparty TRUE)

ocv_list_unique(_modules)
ocv_list_unique(_extra)
ocv_list_unique(_3rdparty)

set(OPENCV_PC_LIBS
  "-L\${exec_prefix}/${OPENCV_LIB_INSTALL_PATH}"
  "${_modules}"
)
if (BUILD_SHARED_LIBS)
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
