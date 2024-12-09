if(PROJECT_NAME STREQUAL "OpenCV")
  set(ENABLE_PLUGINS_DEFAULT ON)
  if(EMSCRIPTEN OR IOS OR WINRT)
    set(ENABLE_PLUGINS_DEFAULT OFF)
  endif()
  set(DNN_PLUGIN_LIST "" CACHE STRING "List of DNN backends to be compiled as plugins (openvino, etc or special value 'all')")
  set(DNN_ENABLE_PLUGINS "${ENABLE_PLUGINS_DEFAULT}" CACHE BOOL "Allow building and using of DNN plugins")
  mark_as_advanced(DNN_PLUGIN_LIST DNN_ENABLE_PLUGINS)

  string(REPLACE "," ";" DNN_PLUGIN_LIST "${DNN_PLUGIN_LIST}")  # support comma-separated list (,) too
  string(TOLOWER "${DNN_PLUGIN_LIST}" DNN_PLUGIN_LIST)
  if(NOT DNN_ENABLE_PLUGINS)
    if(DNN_PLUGIN_LIST)
      message(WARNING "DNN: plugins are disabled through DNN_ENABLE_PLUGINS, so DNN_PLUGIN_LIST='${DNN_PLUGIN_LIST}' is ignored")
      set(DNN_PLUGIN_LIST "")
    endif()
  else()
    # Make virtual plugins target
    if(NOT TARGET opencv_dnn_plugins)
      add_custom_target(opencv_dnn_plugins ALL)
    endif()
  endif()
endif()

#
# Detect available dependencies
#

# OpenVINO - detected by main CMake scripts (shared with G-API)
