set(the_description "High-level GUI and Media I/O")
ocv_add_module(highgui opencv_imgproc OPTIONAL opencv_androidcamera)

# ----------------------------------------------------------------------------
#  CMake file for highgui. See root CMakeLists.txt
#   Some parts taken from version of Hartmut Seichter, HIT Lab NZ.
#   Jose Luis Blanco, 2008
# ----------------------------------------------------------------------------

ocv_clear_vars(GRFMT_LIBS)

if(HAVE_WINRT_CX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /ZW")
endif()

if(HAVE_PNG OR HAVE_TIFF OR HAVE_OPENEXR)
  ocv_include_directories(${ZLIB_INCLUDE_DIRS})
  list(APPEND GRFMT_LIBS ${ZLIB_LIBRARIES})
endif()

if(HAVE_JPEG)
  ocv_include_directories(${JPEG_INCLUDE_DIR})
  list(APPEND GRFMT_LIBS ${JPEG_LIBRARIES})
endif()

if(WITH_WEBP)
  add_definitions(-DHAVE_WEBP)
  ocv_include_directories(${WEBP_INCLUDE_DIR})
  list(APPEND GRFMT_LIBS ${WEBP_LIBRARIES})
endif()

if(HAVE_PNG)
  add_definitions(${PNG_DEFINITIONS})
  ocv_include_directories(${PNG_INCLUDE_DIR})
  list(APPEND GRFMT_LIBS ${PNG_LIBRARIES})
endif()

if(HAVE_TIFF)
  ocv_include_directories(${TIFF_INCLUDE_DIR})
  list(APPEND GRFMT_LIBS ${TIFF_LIBRARIES})
endif()

if(HAVE_JASPER)
  ocv_include_directories(${JASPER_INCLUDE_DIR})
  list(APPEND GRFMT_LIBS ${JASPER_LIBRARIES})
endif()

if(HAVE_OPENEXR)
  include_directories(SYSTEM ${OPENEXR_INCLUDE_PATHS})
  list(APPEND GRFMT_LIBS ${OPENEXR_LIBRARIES})
endif()

file(GLOB grfmt_hdrs src/grfmt*.hpp)
file(GLOB grfmt_srcs src/grfmt*.cpp)
list(APPEND grfmt_hdrs src/bitstrm.hpp)
list(APPEND grfmt_srcs src/bitstrm.cpp)
list(APPEND grfmt_hdrs src/rgbe.hpp)
list(APPEND grfmt_srcs src/rgbe.cpp)

source_group("Src\\grfmts" FILES ${grfmt_hdrs} ${grfmt_srcs})

set(highgui_hdrs
    src/precomp.hpp
    src/utils.hpp
    src/cap_ffmpeg_impl.hpp
    )

set(highgui_srcs
    src/cap.cpp
    src/cap_images.cpp
    src/cap_ffmpeg.cpp
    src/loadsave.cpp
    src/utils.cpp
    src/window.cpp
    )

file(GLOB highgui_ext_hdrs "include/opencv2/*.hpp" "include/opencv2/${name}/*.hpp" "include/opencv2/${name}/*.h")

if(HAVE_QT5)
  set(CMAKE_AUTOMOC ON)
  set(CMAKE_INCLUDE_CURRENT_DIR ON)

  QT5_ADD_RESOURCES(_RCC_OUTFILES src/window_QT.qrc)
  list(APPEND highgui_srcs src/window_QT.cpp src/window_QT.h ${_RCC_OUTFILES})

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

  QT4_ADD_RESOURCES(_RCC_OUTFILES src/window_QT.qrc)
  QT4_WRAP_CPP(_MOC_OUTFILES src/window_QT.h)

  list(APPEND HIGHGUI_LIBRARIES ${QT_LIBRARIES})
  list(APPEND highgui_srcs src/window_QT.cpp ${_MOC_OUTFILES} ${_RCC_OUTFILES})
  ocv_check_flag_support(CXX -Wno-missing-declarations _have_flag)
  if(${_have_flag})
    set_source_files_properties(${_RCC_OUTFILES} PROPERTIES COMPILE_FLAGS -Wno-missing-declarations)
  endif()
elseif(HAVE_WIN32UI)
  list(APPEND highgui_srcs src/window_w32.cpp)
elseif(HAVE_GTK)
  list(APPEND highgui_srcs src/window_gtk.cpp)
elseif(HAVE_CARBON)
  list(APPEND highgui_srcs src/window_carbon.cpp)
  list(APPEND HIGHGUI_LIBRARIES "-framework Carbon" "-framework QuickTime")
elseif(HAVE_COCOA)
  list(APPEND highgui_srcs src/window_cocoa.mm)
  list(APPEND HIGHGUI_LIBRARIES "-framework Cocoa")
endif()

if(WIN32 AND NOT ARM)
  list(APPEND highgui_srcs src/cap_cmu.cpp)
endif()

if (WIN32 AND HAVE_DSHOW)
  list(APPEND highgui_srcs src/cap_dshow.cpp)
endif()

if (WIN32 AND HAVE_MSMF)
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
  ocv_include_directories(${XIMEA_PATH})
  if(XIMEA_LIBRARY_DIR)
    link_directories("${XIMEA_LIBRARY_DIR}")
  endif()
  if(X86_64)
    list(APPEND HIGHGUI_LIBRARIES m3apiX64)
  else()
    list(APPEND HIGHGUI_LIBRARIES m3api)
  endif()
endif(HAVE_XIMEA)

if(HAVE_FFMPEG)
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
  include_directories(AFTER SYSTEM "${OpenCV_SOURCE_DIR}/3rdparty/include/ffmpeg_") # for tests
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
ocv_set_module_sources(HEADERS ${highgui_ext_hdrs} SOURCES ${highgui_srcs} ${highgui_hdrs} ${grfmt_srcs} ${grfmt_hdrs})
ocv_module_include_directories()

ocv_create_module(${GRFMT_LIBS} ${HIGHGUI_LIBRARIES})

if(APPLE)
  ocv_check_flag_support(OBJCXX "-fobjc-exceptions" HAVE_OBJC_EXCEPTIONS)
  if(HAVE_OBJC_EXCEPTIONS)
    foreach(source ${OPENCV_MODULE_${the_module}_SOURCES})
      if("${source}" MATCHES "\\.mm$")
        get_source_file_property(flags "${source}" COMPILE_FLAGS)
        if(flags)
          set(flags "${_flags} -fobjc-exceptions")
        else()
          set(flags "-fobjc-exceptions")
        endif()

        set_source_files_properties("${source}" PROPERTIES COMPILE_FLAGS "${flags}")
      endif()
    endforeach()
  endif()
endif()

if(BUILD_SHARED_LIBS)
  add_definitions(-DHIGHGUI_EXPORTS)
endif()

if(MSVC)
  set_target_properties(${the_module} PROPERTIES LINK_FLAGS "/NODEFAULTLIB:atlthunk.lib /NODEFAULTLIB:atlsd.lib /NODEFAULTLIB:libcmt.lib /DEBUG")
endif()

#stop automatic dependencies propagation for this module
set_target_properties(${the_module} PROPERTIES LINK_INTERFACE_LIBRARIES "")

ocv_add_precompiled_headers(${the_module})
ocv_warnings_disable(CMAKE_CXX_FLAGS -Wno-deprecated-declarations)

if(WIN32 AND WITH_FFMPEG)
  #copy ffmpeg dll to the output folder
  if(MSVC64 OR MINGW64)
    set(FFMPEG_SUFFIX _64)
  endif()

  set(ffmpeg_bare_name "opencv_ffmpeg${FFMPEG_SUFFIX}.dll")
  set(ffmpeg_bare_name_ver "opencv_ffmpeg${OPENCV_DLLVERSION}${FFMPEG_SUFFIX}.dll")
  set(ffmpeg_path "${OpenCV_SOURCE_DIR}/3rdparty/ffmpeg/${ffmpeg_bare_name}")

  if(MSVC_IDE)
    add_custom_command(TARGET ${the_module} POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy "${ffmpeg_path}" "${EXECUTABLE_OUTPUT_PATH}/Release/${ffmpeg_bare_name_ver}"
                       COMMAND ${CMAKE_COMMAND} -E copy "${ffmpeg_path}" "${EXECUTABLE_OUTPUT_PATH}/Debug/${ffmpeg_bare_name_ver}"
                       COMMENT "Copying ${ffmpeg_path} to the output directory")
  elseif(MSVC AND (CMAKE_GENERATOR MATCHES "Visual"))
    add_custom_command(TARGET ${the_module} POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy "${ffmpeg_path}" "${EXECUTABLE_OUTPUT_PATH}/${CMAKE_BUILD_TYPE}/${ffmpeg_bare_name_ver}"
                       COMMENT "Copying ${ffmpeg_path} to the output directory")
  else()
    add_custom_command(TARGET ${the_module} POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy "${ffmpeg_path}" "${EXECUTABLE_OUTPUT_PATH}/${ffmpeg_bare_name_ver}"
                       COMMENT "Copying ${ffmpeg_path} to the output directory")
  endif()

  install(FILES "${ffmpeg_path}" DESTINATION ${OPENCV_BIN_INSTALL_PATH} COMPONENT libs RENAME "${ffmpeg_bare_name_ver}")
endif()

ocv_add_accuracy_tests()
ocv_add_perf_tests()
