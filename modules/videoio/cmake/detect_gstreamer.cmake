# --- GStreamer ---
if(NOT HAVE_GSTREAMER AND WIN32)
  set(env_paths "${GSTREAMER_DIR}" ENV GSTREAMER_ROOT)
  if(X86_64)
    list(APPEND env_paths ENV GSTREAMER_1_0_ROOT_X86_64 ENV GSTREAMER_ROOT_X86_64)
  else()
    list(APPEND env_paths ENV GSTREAMER_1_0_ROOT_X86 ENV GSTREAMER_ROOT_X86)
  endif()

  find_path(GSTREAMER_gst_INCLUDE_DIR
    gst/gst.h
    PATHS ${env_paths}
    PATH_SUFFIXES "include/gstreamer-1.0")
  find_path(GSTREAMER_glib_INCLUDE_DIR
    glib.h
    PATHS ${env_paths}
    PATH_SUFFIXES "include/glib-2.0")
  find_path(GSTREAMER_glibconfig_INCLUDE_DIR
    glibconfig.h
    PATHS ${env_paths}
    PATH_SUFFIXES "lib/glib-2.0/include")

  find_library(GSTREAMER_gstreamer_LIBRARY
    NAMES gstreamer gstreamer-1.0
    PATHS ${env_paths}
    PATH_SUFFIXES "lib")
  find_library(GSTREAMER_app_LIBRARY
    NAMES gstapp gstapp-1.0
    PATHS ${env_paths}
    PATH_SUFFIXES "lib")
  find_library(GSTREAMER_base_LIBRARY
    NAMES gstbase gstbase-1.0
    PATHS ${env_paths}
    PATH_SUFFIXES "lib")
  find_library(GSTREAMER_pbutils_LIBRARY
    NAMES gstpbutils gstpbutils-1.0
    PATHS ${env_paths}
    PATH_SUFFIXES "lib")
  find_library(GSTREAMER_riff_LIBRARY
    NAMES gstriff gstriff-1.0
    PATHS ${env_paths}
    PATH_SUFFIXES "lib")
  find_library(GSTREAMER_video_LIBRARY
    NAMES gstvideo gstvideo-1.0
    PATHS ${env_paths}
    PATH_SUFFIXES "lib")

  find_library(GSTREAMER_glib_LIBRARY
    NAMES glib-2.0
    PATHS ${env_paths}
    PATH_SUFFIXES "lib")
  find_library(GSTREAMER_gobject_LIBRARY
    NAMES gobject-2.0
    PATHS ${env_paths}
    PATH_SUFFIXES "lib")

  if(GSTREAMER_gst_INCLUDE_DIR
      AND GSTREAMER_glib_INCLUDE_DIR
      AND GSTREAMER_glibconfig_INCLUDE_DIR
      AND GSTREAMER_gstreamer_LIBRARY
      AND GSTREAMER_app_LIBRARY
      AND GSTREAMER_base_LIBRARY
      AND GSTREAMER_pbutils_LIBRARY
      AND GSTREAMER_riff_LIBRARY
      AND GSTREAMER_video_LIBRARY
      AND GSTREAMER_glib_LIBRARY
      AND GSTREAMER_gobject_LIBRARY)
    file(STRINGS "${GSTREAMER_gst_INCLUDE_DIR}/gst/gstversion.h" ver_strings REGEX "#define +GST_VERSION_(MAJOR|MINOR|MICRO|NANO).*")
    string(REGEX REPLACE ".*GST_VERSION_MAJOR[^0-9]+([0-9]+).*" "\\1" ver_major "${ver_strings}")
    string(REGEX REPLACE ".*GST_VERSION_MINOR[^0-9]+([0-9]+).*" "\\1" ver_minor "${ver_strings}")
    string(REGEX REPLACE ".*GST_VERSION_MICRO[^0-9]+([0-9]+).*" "\\1" ver_micro "${ver_strings}")
    set(GSTREAMER_VERSION "${ver_major}.${ver_minor}.${ver_micro}" PARENT_SCOPE) # informational
    set(HAVE_GSTREAMER TRUE)
    set(GSTREAMER_LIBRARIES
      ${GSTREAMER_gstreamer_LIBRARY}
      ${GSTREAMER_base_LIBRARY}
      ${GSTREAMER_app_LIBRARY}
      ${GSTREAMER_riff_LIBRARY}
      ${GSTREAMER_video_LIBRARY}
      ${GSTREAMER_pbutils_LIBRARY}
      ${GSTREAMER_glib_LIBRARY}
      ${GSTREAMER_gobject_LIBRARY})
    set(GSTREAMER_INCLUDE_DIRS
      ${GSTREAMER_gst_INCLUDE_DIR}
      ${GSTREAMER_glib_INCLUDE_DIR}
      ${GSTREAMER_glibconfig_INCLUDE_DIR})
  endif()
endif()

if(NOT HAVE_GSTREAMER AND PKG_CONFIG_FOUND)
  ocv_check_modules(GSTREAMER_base gstreamer-base-1.0)
  ocv_check_modules(GSTREAMER_app gstreamer-app-1.0)
  ocv_check_modules(GSTREAMER_riff gstreamer-riff-1.0)
  ocv_check_modules(GSTREAMER_pbutils gstreamer-pbutils-1.0)
  ocv_check_modules(GSTREAMER_video gstreamer-video-1.0)
  if(GSTREAMER_base_FOUND AND GSTREAMER_app_FOUND AND GSTREAMER_riff_FOUND AND GSTREAMER_pbutils_FOUND AND GSTREAMER_video_FOUND)
    set(HAVE_GSTREAMER TRUE)
    set(GSTREAMER_VERSION ${GSTREAMER_base_VERSION} PARENT_SCOPE) # informational
    set(GSTREAMER_LIBRARIES ${GSTREAMER_base_LIBRARIES} ${GSTREAMER_app_LIBRARIES} ${GSTREAMER_riff_LIBRARIES} ${GSTREAMER_pbutils_LIBRARIES} ${GSTREAMER_video_LIBRARIES})
    set(GSTREAMER_INCLUDE_DIRS ${GSTREAMER_base_INCLUDE_DIRS} ${GSTREAMER_app_INCLUDE_DIRS} ${GSTREAMER_riff_INCLUDE_DIRS} ${GSTREAMER_pbutils_INCLUDE_DIRS} ${GSTREAMER_video_INCLUDE_DIRS})
  endif()
endif()

if(HAVE_GSTREAMER)
  ocv_add_external_target(gstreamer "${GSTREAMER_INCLUDE_DIRS}" "${GSTREAMER_LIBRARIES}" "HAVE_GSTREAMER")
endif()

set(HAVE_GSTREAMER ${HAVE_GSTREAMER} PARENT_SCOPE)
