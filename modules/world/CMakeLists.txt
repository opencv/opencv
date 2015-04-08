set(the_description "All the selected OpenCV modules in a single binary")
set(OPENCV_MODULE_IS_PART_OF_WORLD FALSE)
set(BUILD_opencv_world_INIT OFF)

if(IOS OR NOT BUILD_SHARED_LIBS)
  set(OPENCV_MODULE_TYPE STATIC)
  set(OPENCV_WORLD_FLAGS_PROPERTY STATIC_LIBRARY_FLAGS)
else()
  set(OPENCV_WORLD_FLAGS_PROPERTY LINK_FLAGS)
endif()

function(include_one_module m)
  include("${OPENCV_MODULE_${m}_LOCATION}/CMakeLists.txt")
endfunction()

if(NOT OPENCV_INITIAL_PASS)
  project(opencv_world)

  message(STATUS "Processing WORLD modules...")
  foreach(m ${OPENCV_MODULES_BUILD})
    if(OPENCV_MODULE_${m}_IS_PART_OF_WORLD)
      message(STATUS "    module ${m}...")
      set(CMAKE_CURRENT_SOURCE_DIR ${OPENCV_MODULE_${m}_LOCATION})
      #add_subdirectory("${OPENCV_MODULE_${m}_LOCATION}" ${CMAKE_CURRENT_BINARY_DIR}/${m})
      include_one_module(${m})
    endif()
  endforeach()
  message(STATUS "Processing WORLD modules... DONE")
  set(CMAKE_CURRENT_SOURCE_DIR OPENCV_MODULE_${opencv_world}_LOCATION)
endif()

ocv_add_module(world opencv_core)

set(headers_list "HEADERS")
set(sources_list "SOURCES")
set(link_deps "")
foreach(m ${OPENCV_MODULE_${the_module}_DEPS})
  set(headers_list "${headers_list};${OPENCV_MODULE_${m}_HEADERS}")
  set(sources_list "${sources_list};${OPENCV_MODULE_${m}_SOURCES}")
  set(link_deps "${link_deps};${OPENCV_MODULE_${m}_LINK_DEPS}")
endforeach()

ocv_glob_module_sources(${headers_list} ${sources_list})

ocv_module_include_directories()

#message(STATUS "${OPENCV_MODULE_${the_module}_HEADERS}")
#message(STATUS "${OPENCV_MODULE_${the_module}_SOURCES}")
ocv_create_module(${link_deps})

if(BUILD_opencv_imgcodecs AND OPENCV_MODULE_opencv_imgcodecs_IS_PART_OF_WORLD)
  ocv_imgcodecs_configure_target()
endif()
if(BUILD_opencv_videoio AND OPENCV_MODULE_opencv_videoio_IS_PART_OF_WORLD)
  ocv_videoio_configure_target()
endif()
if(BUILD_opencv_highgui AND OPENCV_MODULE_opencv_highgui_IS_PART_OF_WORLD)
  ocv_highgui_configure_target()
endif()

if(IOS OR APPLE)
  set(merge_libs "")
  macro(ios_include_3party_libs)
    foreach(l ${ARGN})
      add_dependencies(${the_module} ${l})
      list(APPEND merge_libs "$<TARGET_LINKER_FILE:${l}>")
    endforeach()
  endmacro()

  if(WITH_PNG)
    ios_include_3party_libs(zlib libpng)
  endif()

  if(WITH_JPEG)
    ios_include_3party_libs(libjpeg)
  endif()

  add_custom_command(TARGET ${the_module} POST_BUILD
    COMMAND /usr/bin/libtool -static -o ${CMAKE_CURRENT_BINARY_DIR}/${the_module}_fat.a $<TARGET_LINKER_FILE:${the_module}> ${merge_libs}
    COMMAND mv ${CMAKE_CURRENT_BINARY_DIR}/${the_module}_fat.a $<TARGET_LINKER_FILE:${the_module}>
  )
endif()
