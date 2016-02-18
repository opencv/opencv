set(the_description "High-level GUI and Media I/O")
ocv_add_module(highgui opencv_imgproc opencv_imgcodecs opencv_videoio OPTIONAL opencv_androidcamera)

# ----------------------------------------------------------------------------
#  CMake file for highgui. See root CMakeLists.txt
#   Some parts taken from version of Hartmut Seichter, HIT Lab NZ.
#   Jose Luis Blanco, 2008
# ----------------------------------------------------------------------------

<<<<<<< HEAD
if(HAVE_WINRT_CX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /ZW")
=======
ocv_clear_vars(GRFMT_LIBS)

if(HAVE_WINRT_CX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /ZW")
endif()

if(HAVE_PNG OR HAVE_TIFF OR HAVE_OPENEXR)
  ocv_include_directories(${ZLIB_INCLUDE_DIR})
  list(APPEND GRFMT_LIBS ${ZLIB_LIBRARIES})
endif()

if(HAVE_JPEG)
  ocv_include_directories(${JPEG_INCLUDE_DIR})
  list(APPEND GRFMT_LIBS ${JPEG_LIBRARIES})
endif()

if(HAVE_PNG)
  add_definitions(${PNG_DEFINITIONS})
  ocv_include_directories(${PNG_INCLUDE_DIR})
  list(APPEND GRFMT_LIBS ${PNG_LIBRARIES})
endif()

if(HAVE_TIFF)
  ocv_include_directories(${TIFF_INCLUDE_DIR})
  list(APPEND GRFMT_LIBS ${TIFF_LIBRARIES})
>>>>>>> a28cde9c3bf69e7839971c29900fbbd4963998bd
endif()

if(APPLE)
  ocv_include_directories(${ZLIB_INCLUDE_DIRS})
  list(APPEND HIGHGUI_LIBRARIES ${ZLIB_LIBRARIES})
endif()

set(highgui_hdrs
<<<<<<< HEAD
    ${CMAKE_CURRENT_LIST_DIR}/src/precomp.hpp
    )

set(highgui_srcs
    ${CMAKE_CURRENT_LIST_DIR}/src/window.cpp
=======
    src/precomp.hpp
    src/utils.hpp
    )

set(highgui_srcs
    src/cap.cpp
    src/cap_images.cpp
    src/loadsave.cpp
    src/utils.cpp
    src/window.cpp
>>>>>>> a28cde9c3bf69e7839971c29900fbbd4963998bd
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

<<<<<<< HEAD
=======
if(WIN32 AND NOT ARM)
  list(APPEND highgui_srcs src/cap_cmu.cpp)
endif()

if (WIN32 AND HAVE_DSHOW)
  list(APPEND highgui_srcs src/cap_dshow.cpp)
endif()

if (WIN32 AND HAVE_MSMF)
  list(APPEND highgui_srcs src/cap_msmf.hpp)
  list(APPEND highgui_srcs src/cap_msmf.cpp)
endif()

if (WIN32 AND HAVE_VFW)
  list(APPEND highgui_srcs src/cap_vfw.cpp)
endif()

if(HAVE_XINE)
  list(APPEND highgui_srcs src/cap_xine.cpp)
endif(HAVE_XINE)

if(HAVE_DC1394_2)
  list(APPEND highgui_srcs src/cap_dc1394_v2.cpp)
endif(HAVE_DC1394_2)

if(HAVE_DC1394)
  list(APPEND highgui_srcs src/cap_dc1394.cpp)
endif(HAVE_DC1394)

if(HAVE_GSTREAMER)
  list(APPEND highgui_srcs src/cap_gstreamer.cpp)
endif(HAVE_GSTREAMER)

if(HAVE_UNICAP)
  list(APPEND highgui_srcs src/cap_unicap.cpp)
endif(HAVE_UNICAP)

if(HAVE_LIBV4L)
  list(APPEND highgui_srcs src/cap_libv4l.cpp)
elseif(HAVE_CAMV4L OR HAVE_CAMV4L2 OR HAVE_VIDEOIO)
  list(APPEND highgui_srcs src/cap_v4l.cpp)
endif()

if(HAVE_OPENNI)
  list(APPEND highgui_srcs src/cap_openni.cpp)
  ocv_include_directories(${OPENNI_INCLUDE_DIR})
  list(APPEND HIGHGUI_LIBRARIES ${OPENNI_LIBRARY})
endif(HAVE_OPENNI)

if(HAVE_opencv_androidcamera)
  list(APPEND highgui_srcs src/cap_android.cpp)
  add_definitions(-DHAVE_ANDROID_NATIVE_CAMERA)#TODO: remove this line
endif(HAVE_opencv_androidcamera)

if(HAVE_XIMEA)
  list(APPEND highgui_srcs src/cap_ximea.cpp)
  if(XIMEA_PATH)
    ocv_include_directories(${XIMEA_PATH})
  endif()
  if(XIMEA_LIBRARY_DIR)
    link_directories("${XIMEA_LIBRARY_DIR}")
  endif()
  if(WIN32 AND X86_64)
    list(APPEND HIGHGUI_LIBRARIES m3apiX64)
  elseif(APPLE)
    list(APPEND HIGHGUI_LIBRARIES "-framework m3api")
  else()
    list(APPEND HIGHGUI_LIBRARIES m3api)
  endif()
endif(HAVE_XIMEA)

if(HAVE_FFMPEG)
  list(APPEND highgui_hdrs src/cap_ffmpeg_impl.hpp)
  list(APPEND highgui_srcs src/cap_ffmpeg.cpp)
  if(UNIX AND BZIP2_LIBRARIES)
    list(APPEND HIGHGUI_LIBRARIES ${BZIP2_LIBRARIES})
  endif()
  if(APPLE)
    list(APPEND HIGHGUI_LIBRARIES "-framework VideoDecodeAcceleration" bz2)
  endif()
endif(HAVE_FFMPEG)

if(HAVE_PVAPI)
  add_definitions(-DHAVE_PVAPI)
  add_definitions(${PVAPI_DEFINITIONS})
  ocv_include_directories(${PVAPI_INCLUDE_PATH})
  set(highgui_srcs src/cap_pvapi.cpp ${highgui_srcs})
  list(APPEND HIGHGUI_LIBRARIES ${PVAPI_LIBRARY})
endif()

if(HAVE_GIGE_API)
  add_definitions(-DHAVE_GIGE_API)
  ocv_include_directories(${GIGEAPI_INCLUDE_PATH})
  set(highgui_srcs src/cap_giganetix.cpp ${highgui_srcs})
  list(APPEND HIGHGUI_LIBRARIES ${GIGEAPI_LIBRARIES})
  list(APPEND highgui_srcs src/cap_giganetix.cpp)
endif(HAVE_GIGE_API)

if(HAVE_IMAGEIO AND IOS)
  list(APPEND HIGHGUI_LIBRARIES "-framework ImageIO")
endif()

if(HAVE_AVFOUNDATION)
  list(APPEND highgui_srcs src/cap_avfoundation.mm)
  list(APPEND HIGHGUI_LIBRARIES "-framework AVFoundation" "-framework QuartzCore")
endif()

if(HAVE_QUICKTIME)
  list(APPEND highgui_srcs src/cap_qt.cpp)
  list(APPEND HIGHGUI_LIBRARIES "-framework Carbon" "-framework QuickTime" "-framework CoreFoundation" "-framework QuartzCore")
elseif(HAVE_QTKIT)
  list(APPEND highgui_srcs src/cap_qtkit.mm)
  list(APPEND HIGHGUI_LIBRARIES "-framework QTKit" "-framework QuartzCore" "-framework AppKit")
endif()

if(HAVE_INTELPERC)
  list(APPEND highgui_srcs src/cap_intelperc.cpp)
  ocv_include_directories(${INTELPERC_INCLUDE_DIR})
  list(APPEND HIGHGUI_LIBRARIES ${INTELPERC_LIBRARIES})
endif(HAVE_INTELPERC)

if(IOS)
  add_definitions(-DHAVE_IOS=1)
  list(APPEND highgui_srcs src/ios_conversions.mm src/cap_ios_abstract_camera.mm src/cap_ios_photo_camera.mm src/cap_ios_video_camera.mm)
  list(APPEND HIGHGUI_LIBRARIES "-framework Accelerate" "-framework AVFoundation" "-framework CoreGraphics" "-framework CoreImage" "-framework CoreMedia" "-framework CoreVideo" "-framework QuartzCore" "-framework AssetsLibrary")
endif()

if(WIN32)
  link_directories("${OpenCV_SOURCE_DIR}/3rdparty/lib") # for ffmpeg wrapper only
  include_directories(AFTER SYSTEM "${OpenCV_SOURCE_DIR}/3rdparty/include") # for directshow in VS2005 and multi-monitor support on MinGW
endif()

>>>>>>> a28cde9c3bf69e7839971c29900fbbd4963998bd
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
