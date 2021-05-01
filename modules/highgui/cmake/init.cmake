include(FindPkgConfig)

# FIXIT: stop using PARENT_SCOPE in dependencies
if(PROJECT_NAME STREQUAL "OpenCV")
  macro(add_backend backend_id cond_var)
    if(${cond_var})
      include("${CMAKE_CURRENT_LIST_DIR}/detect_${backend_id}.cmake")
    endif()
  endmacro()
else()
  function(add_backend backend_id cond_var)
    if(${cond_var})
      include("${CMAKE_CURRENT_LIST_DIR}/detect_${backend_id}.cmake")
    endif()
  endfunction()
endif()

add_backend("gtk" WITH_GTK)

# TODO win32
# TODO cocoa
# TODO qt
# TODO opengl

# FIXIT: move content of cmake/OpenCVFindLibsGUI.cmake here (need to resolve CMake scope issues)
