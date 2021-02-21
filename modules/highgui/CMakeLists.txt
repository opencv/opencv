set(the_description "High-level GUI")
if(ANDROID)
  ocv_add_module(highgui opencv_imgproc opencv_imgcodecs OPTIONAL opencv_videoio WRAP python)
else()
  ocv_add_module(highgui opencv_imgproc opencv_imgcodecs OPTIONAL opencv_videoio WRAP python java)
endif()

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

set(highgui_hdrs
    ${CMAKE_CURRENT_LIST_DIR}/src/precomp.hpp
    )

set(highgui_srcs
    ${CMAKE_CURRENT_LIST_DIR}/src/window.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/roiSelector.cpp
    )

file(GLOB highgui_ext_hdrs
     "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/*.hpp"
     "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/*.hpp"
     "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/*.h")

# Removing WinRT API headers by default
list(REMOVE_ITEM highgui_ext_hdrs "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/highgui_winrt.hpp")

if(HAVE_QT5)
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
    add_definitions(${Qt5OpenGL_DEFINITIONS})
    include_directories(${Qt5OpenGL_INCLUDE_DIRS})
    list(APPEND HIGHGUI_LIBRARIES ${Qt5OpenGL_LIBRARIES})
  endif()

elseif(HAVE_QT)
  if (HAVE_QT_OPENGL)
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
elseif(HAVE_GTK OR HAVE_GTK3)
  list(APPEND highgui_srcs ${CMAKE_CURRENT_LIST_DIR}/src/window_gtk.cpp)
elseif(HAVE_COCOA)
  list(APPEND highgui_srcs ${CMAKE_CURRENT_LIST_DIR}/src/window_cocoa.mm)
  list(APPEND HIGHGUI_LIBRARIES "-framework Cocoa")
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

ocv_add_accuracy_tests()
ocv_add_perf_tests()
