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
set(prefix      "${CMAKE_INSTALL_PREFIX}")
set(exec_prefix "\${prefix}")
set(libdir      "") #TODO: need link paths for OpenCV_EXTRA_COMPONENTS
set(includedir  "\${prefix}/${OPENCV_INCLUDE_INSTALL_PATH}")

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
set(OpenCV_LIB_COMPONENTS_ "")
foreach(CVLib ${OpenCV_LIB_COMPONENTS})
  get_target_property(libpath ${CVLib} LOCATION_${CMAKE_BUILD_TYPE})
  get_filename_component(libname "${libpath}" NAME)

  if(INSTALL_TO_MANGLED_PATHS)
    set(libname "${libname}.${OPENCV_VERSION}")
  endif()

  #need better solution....
  if(libpath MATCHES "3rdparty")
    set(installDir "share/OpenCV/3rdparty/${OPENCV_LIB_INSTALL_PATH}")
  else()
    set(installDir "${OPENCV_LIB_INSTALL_PATH}")
  endif()

  set(OpenCV_LIB_COMPONENTS_ "${OpenCV_LIB_COMPONENTS_} \${exec_prefix}/${installDir}/${libname}")
endforeach()

# add extra dependencies required for OpenCV
set(OpenCV_LIB_COMPONENTS ${OpenCV_LIB_COMPONENTS_})
if(OpenCV_EXTRA_COMPONENTS)
  foreach(extra_component ${OpenCV_EXTRA_COMPONENTS})

    if(extra_component MATCHES "^-[lL]" OR extra_component MATCHES "[\\/]")
      set(maybe_l_prefix "")
    else()
      set(maybe_l_prefix "-l")
    endif()

    set(OpenCV_LIB_COMPONENTS "${OpenCV_LIB_COMPONENTS} ${maybe_l_prefix}${extra_component}")

  endforeach()
endif()

#generate the .pc file
if(INSTALL_TO_MANGLED_PATHS)
  set(OPENCV_PC_FILE_NAME "opencv-${OPENCV_VERSION}.pc")
else()
  set(OPENCV_PC_FILE_NAME opencv.pc)
endif()
configure_file("${OpenCV_SOURCE_DIR}/cmake/templates/opencv-XXX.pc.cmake.in" "${CMAKE_BINARY_DIR}/unix-install/${OPENCV_PC_FILE_NAME}" @ONLY IMMEDIATE)

if(UNIX AND NOT ANDROID)
  install(FILES ${CMAKE_BINARY_DIR}/unix-install/${OPENCV_PC_FILE_NAME} DESTINATION ${OPENCV_LIB_INSTALL_PATH}/pkgconfig)
endif()
