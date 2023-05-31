set(the_description "All OpenCV modules")
set(OPENCV_MODULE_IS_PART_OF_WORLD FALSE)
set(BUILD_opencv_world_INIT OFF)

if(NOT BUILD_SHARED_LIBS)
  set(OPENCV_MODULE_TYPE STATIC)
  set(OPENCV_WORLD_FLAGS_PROPERTY STATIC_LIBRARY_FLAGS)
else()
  set(OPENCV_WORLD_FLAGS_PROPERTY LINK_FLAGS)
endif()

function(include_one_module m)
  include("${OPENCV_MODULE_${m}_LOCATION}/CMakeLists.txt")
  foreach(var
      CMAKE_CXX_FLAGS CMAKE_C_FLAGS # Propagate warnings settings
  )
    set(${var} "${${var}}" PARENT_SCOPE)
  endforeach()
endfunction()

if(NOT OPENCV_INITIAL_PASS)
  set(ENABLE_PRECOMPILED_HEADERS OFF CACHE INTERNAL "" FORCE)
  project(opencv_world)

  message(STATUS "Processing WORLD modules...")
  foreach(m ${OPENCV_MODULES_BUILD})
    set(the_module ${m})
    if(OPENCV_MODULE_${m}_IS_PART_OF_WORLD)
      message(STATUS "    module ${m}...")
      set(CMAKE_CURRENT_SOURCE_DIR "${OPENCV_MODULE_${m}_LOCATION}")
      #add_subdirectory("${OPENCV_MODULE_${m}_LOCATION}" ${CMAKE_CURRENT_BINARY_DIR}/${m})
      include_one_module(${m})
    endif()
  endforeach()
  message(STATUS "Processing WORLD modules... DONE")
  set(CMAKE_CURRENT_SOURCE_DIR "${OPENCV_MODULE_opencv_world_LOCATION}")
endif()

ocv_add_module(world opencv_core)

set(headers_list)
set(sources_list)
set(link_deps "")
foreach(m ${OPENCV_MODULE_${the_module}_DEPS} opencv_world)
  if(OPENCV_MODULE_${m}_IS_PART_OF_WORLD)
    list(APPEND headers_list ${OPENCV_MODULE_${m}_HEADERS})
    list(APPEND sources_list ${OPENCV_MODULE_${m}_SOURCES})
  endif()
  if(NOT " ${OPENCV_MODULE_${m}_LINK_DEPS}" STREQUAL " ")
    list(APPEND link_deps ${OPENCV_MODULE_${m}_LINK_DEPS})
  endif()
endforeach()

ocv_glob_module_sources(HEADERS ${headers_list} SOURCES ${sources_list})

ocv_module_include_directories()

#message(STATUS "${OPENCV_MODULE_${the_module}_HEADERS}")
#message(STATUS "${OPENCV_MODULE_${the_module}_SOURCES}")
ocv_create_module(${link_deps})

if(";${OPENCV_MODULES_BUILD};" MATCHES ";opencv_viz;" AND OPENCV_MODULE_opencv_viz_IS_PART_OF_WORLD AND NOT (VTK_VERSION VERSION_LESS "8.90.0"))
  vtk_module_autoinit(TARGETS opencv_world MODULES ${VTK_LIBRARIES})
endif()

ocv_target_compile_definitions(${the_module} PRIVATE OPENCV_MODULE_IS_PART_OF_WORLD=1)

if(BUILD_opencv_imgcodecs AND OPENCV_MODULE_opencv_imgcodecs_IS_PART_OF_WORLD)
  ocv_imgcodecs_configure_target()
endif()
if(BUILD_opencv_highgui AND OPENCV_MODULE_opencv_highgui_IS_PART_OF_WORLD)
  ocv_highgui_configure_target()
endif()
