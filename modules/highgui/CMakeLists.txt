set(the_description "High-level GUI")

set(ENABLE_PLUGINS_DEFAULT ON)
if(EMSCRIPTEN OR IOS OR WINRT)
  set(ENABLE_PLUGINS_DEFAULT OFF)
endif()
set(HIGHGUI_PLUGIN_LIST "" CACHE STRING "List of GUI backends to be compiled as plugins (gtk, gtk2/gtk3, qt, win32 or special value 'all')")
set(HIGHGUI_ENABLE_PLUGINS "${ENABLE_PLUGINS_DEFAULT}" CACHE BOOL "Allow building and using of GUI plugins")
mark_as_advanced(HIGHGUI_PLUGIN_LIST HIGHGUI_ENABLE_PLUGINS)

string(REPLACE "," ";" HIGHGUI_PLUGIN_LIST "${HIGHGUI_PLUGIN_LIST}")  # support comma-separated list (,) too
if(NOT HIGHGUI_ENABLE_PLUGINS)
  if(HIGHGUI_PLUGIN_LIST)
    message(WARNING "HighGUI: plugins are disabled through HIGHGUI_ENABLE_PLUGINS, so HIGHGUI_PLUGIN_LIST='${HIGHGUI_PLUGIN_LIST}' is ignored")
    set(HIGHGUI_PLUGIN_LIST "")
  endif()
else()
  # Make virtual plugins target
  if(NOT TARGET opencv_highgui_plugins)
    add_custom_target(opencv_highgui_plugins ALL)
  endif()
endif()

if(ANDROID)
  ocv_add_module(highgui opencv_imgproc opencv_imgcodecs OPTIONAL opencv_videoio WRAP python)
else()
  ocv_add_module(highgui opencv_imgproc opencv_imgcodecs OPTIONAL opencv_videoio WRAP python java)
endif()

include(${CMAKE_CURRENT_LIST_DIR}/cmake/plugin.cmake)

set(tgts "PRIVATE")

set(highgui_hdrs
    ${CMAKE_CURRENT_LIST_DIR}/src/precomp.hpp
    )

set(highgui_srcs
    ${CMAKE_CURRENT_LIST_DIR}/src/backend.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/window.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/roiSelector.cpp
    )

# ----------------------------------------------------------------------------
#  CMake file for highgui. See root CMakeLists.txt
#   Some parts taken from version of Hartmut Seichter, HIT Lab NZ.
#   Jose Luis Blanco, 2008
# ----------------------------------------------------------------------------

if(DEFINED WINRT AND NOT DEFINED ENABLE_WINRT_MODE_NATIVE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /ZW")
endif()

if(APPLE)
  ocv_include_directories(${ZLIB_INCLUDE_DIRS})
  list(APPEND HIGHGUI_LIBRARIES ${ZLIB_LIBRARIES})
endif()

if(HAVE_WEBP)
  add_definitions(-DHAVE_WEBP)
endif()

file(GLOB highgui_ext_hdrs
     "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/*.hpp"
     "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/*.hpp"
     "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/*.h")

# Removing WinRT API headers by default
list(REMOVE_ITEM highgui_ext_hdrs "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/highgui_winrt.hpp")

if(HAVE_QT5)
  add_definitions(-DHAVE_QT)

  # "Automoc" doesn't work properly with opencv_world build, use QT5_WRAP_CPP() directly
  #set(CMAKE_AUTOMOC ON)

  set(CMAKE_INCLUDE_CURRENT_DIR ON)

  QT5_ADD_RESOURCES(_RCC_OUTFILES ${CMAKE_CURRENT_LIST_DIR}/src/window_QT.qrc)
  QT5_WRAP_CPP(_MOC_OUTFILES ${CMAKE_CURRENT_LIST_DIR}/src/window_QT.h)
  list(APPEND highgui_srcs
       ${CMAKE_CURRENT_LIST_DIR}/src/window_QT.cpp
       ${CMAKE_CURRENT_LIST_DIR}/src/window_QT.h
       ${_MOC_OUTFILES}
       ${_RCC_OUTFILES})

  foreach(dt5_dep Core Gui Widgets Test Concurrent)
    add_definitions(${Qt5${dt5_dep}_DEFINITIONS})
    include_directories(${Qt5${dt5_dep}_INCLUDE_DIRS})
    list(APPEND HIGHGUI_LIBRARIES ${Qt5${dt5_dep}_LIBRARIES})
  endforeach()

  if(HAVE_QT_OPENGL)
    add_definitions(-DHAVE_QT_OPENGL)
    add_definitions(${Qt5OpenGL_DEFINITIONS})
    include_directories(${Qt5OpenGL_INCLUDE_DIRS})
    list(APPEND HIGHGUI_LIBRARIES ${Qt5OpenGL_LIBRARIES})
  endif()

elseif(HAVE_QT)
  add_definitions(-DHAVE_QT)
  if(HAVE_QT_OPENGL)
    add_definitions(-DHAVE_QT_OPENGL)
    set(QT_USE_QTOPENGL TRUE)
  endif()
  include(${QT_USE_FILE})

  QT4_ADD_RESOURCES(_RCC_OUTFILES ${CMAKE_CURRENT_LIST_DIR}/src/window_QT.qrc)
  QT4_WRAP_CPP(_MOC_OUTFILES ${CMAKE_CURRENT_LIST_DIR}/src/window_QT.h)

  list(APPEND HIGHGUI_LIBRARIES ${QT_LIBRARIES})
  list(APPEND highgui_srcs ${CMAKE_CURRENT_LIST_DIR}/src/window_QT.cpp ${_MOC_OUTFILES} ${_RCC_OUTFILES})
  ocv_check_flag_support(CXX -Wno-missing-declarations _have_flag "")
  if(${_have_flag})
    set_source_files_properties(${_RCC_OUTFILES} PROPERTIES COMPILE_FLAGS -Wno-missing-declarations)
  endif()
elseif(WINRT)
  if(NOT WINRT_8_0)
    # Dependencies used by the implementation referenced
    # below are not available on WinRT 8.0.
    # Enabling it for WiRT 8.1+ only.

    # WinRT 8.1+ detected. Adding WinRT API header.
    message(STATUS "  ${name}: WinRT detected. Adding WinRT API header")
    list(APPEND highgui_ext_hdrs "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/highgui_winrt.hpp")


    list(APPEND highgui_srcs
      ${CMAKE_CURRENT_LIST_DIR}/src/window_winrt.cpp
      ${CMAKE_CURRENT_LIST_DIR}/src/window_winrt_bridge.cpp)
    list(APPEND highgui_hdrs
      ${CMAKE_CURRENT_LIST_DIR}/src/window_winrt_bridge.hpp)
  endif()

  # libraries below are neither available nor required
  # on ARM devices and/or Windows Phone
  if(WINRT_PHONE OR (OpenCV_ARCH STREQUAL "ARM"))
    list(REMOVE_ITEM HIGHGUI_LIBRARIES "comctl32" "gdi32" "ole32" "setupapi")
    if(WINRT_PHONE)
      message(STATUS "  ${name}: Windows Phone detected")
    elseif(OpenCV_ARCH STREQUAL "ARM")
      message(STATUS "  ${name}: ARM detected")
      if(WINRT_STORE)
        list(REMOVE_ITEM HIGHGUI_LIBRARIES "ws2_32")
        message(STATUS "  ${name}:   Removing 'ws2_32.lib'")
      endif()
    endif()
    message(STATUS "  ${name}:   Removing 'comctl32.lib, gdi32.lib, ole32.lib, setupapi.lib'")
    message(STATUS "  ${name}:   Leaving '${HIGHGUI_LIBRARIES}'")
  endif()
elseif(HAVE_WIN32UI)
  list(APPEND highgui_srcs ${CMAKE_CURRENT_LIST_DIR}/src/window_w32.cpp)
  if(OpenCV_ARCH STREQUAL "ARM64")
    list(APPEND HIGHGUI_LIBRARIES "comdlg32" "advapi32")
  endif()
elseif(HAVE_COCOA)
  add_definitions(-DHAVE_COCOA)
  list(APPEND highgui_srcs ${CMAKE_CURRENT_LIST_DIR}/src/window_cocoa.mm)
  list(APPEND HIGHGUI_LIBRARIES "-framework Cocoa")
endif()

if(TARGET ocv.3rdparty.gtk3 OR TARGET ocv.3rdparty.gtk2)
  if(TARGET ocv.3rdparty.gtk3 AND NOT WITH_GTK_2_X)
    set(__gtk_dependency "ocv.3rdparty.gtk3")
  else()
    set(__gtk_dependency "ocv.3rdparty.gtk2")
  endif()
  if(
    NOT HIGHGUI_PLUGIN_LIST STREQUAL "all"
    AND NOT "gtk" IN_LIST HIGHGUI_PLUGIN_LIST
    AND NOT "gtk2" IN_LIST HIGHGUI_PLUGIN_LIST
    AND NOT "gtk3" IN_LIST HIGHGUI_PLUGIN_LIST
  )
    list(APPEND highgui_srcs ${CMAKE_CURRENT_LIST_DIR}/src/window_gtk.cpp)
    list(APPEND tgts ${__gtk_dependency})
    if(TARGET ocv.3rdparty.gthread)
      list(APPEND tgts ocv.3rdparty.gthread)
    endif()
    if(TARGET ocv.3rdparty.gtkglext
        AND NOT OPENCV_GTK_DISABLE_GTKGLEXT
    )
      list(APPEND tgts ocv.3rdparty.gtkglext)
    endif()
  elseif("gtk" IN_LIST HIGHGUI_PLUGIN_LIST)
    ocv_create_builtin_highgui_plugin(opencv_highgui_gtk ${__gtk_dependency} "window_gtk.cpp")
    if(TARGET ocv.3rdparty.gthread)
      ocv_target_link_libraries(opencv_highgui_gtk ocv.3rdparty.gthread)
    endif()
    if(TARGET ocv.3rdparty.gtkglext)
      ocv_target_link_libraries(opencv_highgui_gtk ocv.3rdparty.gtkglext)
    endif()
  else()
    if(TARGET ocv.3rdparty.gtk3 AND ("gtk3" IN_LIST HIGHGUI_PLUGIN_LIST OR HIGHGUI_PLUGIN_LIST STREQUAL "all"))
      ocv_create_builtin_highgui_plugin(opencv_highgui_gtk3 ocv.3rdparty.gtk3 "window_gtk.cpp")
      if(TARGET ocv.3rdparty.gthread)
        ocv_target_link_libraries(opencv_highgui_gtk3 ocv.3rdparty.gthread)
      endif()
      if(TARGET ocv.3rdparty.gtkglext)
        ocv_target_link_libraries(opencv_highgui_gtk3 ocv.3rdparty.gtkglext)
      endif()
    endif()
    if(TARGET ocv.3rdparty.gtk2 AND ("gtk2" IN_LIST HIGHGUI_PLUGIN_LIST OR HIGHGUI_PLUGIN_LIST STREQUAL "all"))
      ocv_create_builtin_highgui_plugin(opencv_highgui_gtk2 ocv.3rdparty.gtk2 "window_gtk.cpp")
      if(TARGET ocv.3rdparty.gthread)
        ocv_target_link_libraries(opencv_highgui_gtk2 ocv.3rdparty.gthread)
      endif()
      if(TARGET ocv.3rdparty.gtkglext)
        ocv_target_link_libraries(opencv_highgui_gtk2 ocv.3rdparty.gtkglext)
      endif()
    endif()
  endif()
endif()

if(TRUE)
  # these variables are set by 'ocv_append_build_options(HIGHGUI ...)'
  foreach(P ${HIGHGUI_INCLUDE_DIRS})
    ocv_include_directories(${P})
  endforeach()

  foreach(P ${HIGHGUI_LIBRARY_DIRS})
    link_directories(${P})
  endforeach()
endif()

if(tgts STREQUAL "PRIVATE")
  set(tgts "")
endif()

# install used dependencies only
if(NOT BUILD_SHARED_LIBS
    AND NOT (CMAKE_VERSION VERSION_LESS "3.13.0")  # upgrade CMake: https://gitlab.kitware.com/cmake/cmake/-/merge_requests/2152
)
  foreach(tgt in ${tgts})
    if(tgt MATCHES "^ocv\.3rdparty\.")
      install(TARGETS ${tgt} EXPORT OpenCVModules)
    endif()
  endforeach()
endif()

source_group("Src" FILES ${highgui_srcs} ${highgui_hdrs})
source_group("Include" FILES ${highgui_ext_hdrs})
ocv_set_module_sources(HEADERS ${highgui_ext_hdrs} SOURCES ${highgui_srcs} ${highgui_hdrs})
ocv_module_include_directories()

ocv_create_module(${HIGHGUI_LIBRARIES})

macro(ocv_highgui_configure_target)
if(APPLE)
  add_apple_compiler_options(${the_module})
endif()

if(MSVC)
  set_target_properties(${the_module} PROPERTIES LINK_FLAGS "/NODEFAULTLIB:atlthunk.lib /NODEFAULTLIB:atlsd.lib /NODEFAULTLIB:libcmt.lib /DEBUG")
endif()

ocv_warnings_disable(CMAKE_CXX_FLAGS -Wno-deprecated-declarations)
endmacro()

if(NOT BUILD_opencv_world)
  ocv_highgui_configure_target()
endif()

ocv_add_accuracy_tests(${tgts})
#ocv_add_perf_tests(${tgts})

if(HIGHGUI_ENABLE_PLUGINS)
  ocv_target_compile_definitions(${the_module} PRIVATE ENABLE_PLUGINS)
  if(TARGET opencv_test_highgui)
    ocv_target_compile_definitions(opencv_test_highgui PRIVATE ENABLE_PLUGINS)
  endif()
endif()

ocv_target_link_libraries(${the_module} LINK_PRIVATE ${tgts})
