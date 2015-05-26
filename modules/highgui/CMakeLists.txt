if (WINRT)
  ocv_module_disable(highgui)
endif()

set(the_description "High-level GUI and Media I/O")
ocv_add_module(highgui opencv_imgproc opencv_imgcodecs opencv_videoio WRAP python)

# ----------------------------------------------------------------------------
#  CMake file for highgui. See root CMakeLists.txt
#   Some parts taken from version of Hartmut Seichter, HIT Lab NZ.
#   Jose Luis Blanco, 2008
# ----------------------------------------------------------------------------

# Compilation with /ZW is not allowed for *.c files
if(HAVE_WINRT_CX AND NOT WINRT)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /ZW")
endif()

if(APPLE)
  ocv_include_directories(${ZLIB_INCLUDE_DIRS})
  list(APPEND HIGHGUI_LIBRARIES ${ZLIB_LIBRARIES})
endif()

set(highgui_hdrs
    ${CMAKE_CURRENT_LIST_DIR}/src/precomp.hpp
    )

set(highgui_srcs
    ${CMAKE_CURRENT_LIST_DIR}/src/window.cpp
    )

file(GLOB highgui_ext_hdrs
     "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/*.hpp"
     "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/*.hpp"
     "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/*.h")

if(HAVE_QT5)
  set(CMAKE_AUTOMOC ON)
  set(CMAKE_INCLUDE_CURRENT_DIR ON)

  QT5_ADD_RESOURCES(_RCC_OUTFILES ${CMAKE_CURRENT_LIST_DIR}/src/window_QT.qrc)
  list(APPEND highgui_srcs
       ${CMAKE_CURRENT_LIST_DIR}/src/window_QT.cpp
       ${CMAKE_CURRENT_LIST_DIR}/src/window_QT.h
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
  ocv_check_flag_support(CXX -Wno-missing-declarations _have_flag)
  if(${_have_flag})
    set_source_files_properties(${_RCC_OUTFILES} PROPERTIES COMPILE_FLAGS -Wno-missing-declarations)
  endif()
elseif(HAVE_WIN32UI)
  list(APPEND highgui_srcs ${CMAKE_CURRENT_LIST_DIR}/src/window_w32.cpp)
elseif(HAVE_GTK OR HAVE_GTK3)
  list(APPEND highgui_srcs ${CMAKE_CURRENT_LIST_DIR}/src/window_gtk.cpp)
elseif(HAVE_CARBON)
  list(APPEND highgui_srcs ${CMAKE_CURRENT_LIST_DIR}/src/window_carbon.cpp)
  list(APPEND HIGHGUI_LIBRARIES "-framework Carbon" "-framework QuickTime")
elseif(HAVE_COCOA)
  list(APPEND highgui_srcs ${CMAKE_CURRENT_LIST_DIR}/src/window_cocoa.mm)
  list(APPEND HIGHGUI_LIBRARIES "-framework Cocoa")
endif()

if(UNIX)
  #these variables are set by CHECK_MODULE macro
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
  add_apple_compiler_options(the_module)
endif()

if(BUILD_SHARED_LIBS)
  add_definitions(-DHIGHGUI_EXPORTS)
endif()

if(MSVC)
  set_target_properties(${the_module} PROPERTIES LINK_FLAGS "/NODEFAULTLIB:atlthunk.lib /NODEFAULTLIB:atlsd.lib /NODEFAULTLIB:libcmt.lib /DEBUG")
endif()

if(NOT BUILD_opencv_world)
  #stop automatic dependencies propagation for this module
  set_target_properties(${the_module} PROPERTIES LINK_INTERFACE_LIBRARIES "")
endif()

ocv_warnings_disable(CMAKE_CXX_FLAGS -Wno-deprecated-declarations)
endmacro()

if(NOT BUILD_opencv_world)
  ocv_highgui_configure_target()
endif()

ocv_add_accuracy_tests()
ocv_add_perf_tests()
