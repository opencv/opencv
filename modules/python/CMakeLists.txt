# ----------------------------------------------------------------------------
#  CMake file for python support
# ----------------------------------------------------------------------------

if(ANDROID OR APPLE_FRAMEWORK OR WINRT)
  set(__disable_python2 ON)
  set(__disable_python3 ON)
elseif(BUILD_opencv_world OR (WIN32 AND CMAKE_BUILD_TYPE STREQUAL "Debug"))
  if(NOT DEFINED BUILD_opencv_python2)
    set(__disable_python2 ON)
  endif()
  if(NOT DEFINED BUILD_opencv_python3)
    set(__disable_python3 ON)
  endif()
endif()

if(__disable_python2)
  ocv_module_disable_(python2)
endif()
if(__disable_python3)
  ocv_module_disable_(python3)
endif()
if(__disable_python2 AND __disable_python3)
  return()
endif()

add_subdirectory(python2)
add_subdirectory(python3)
