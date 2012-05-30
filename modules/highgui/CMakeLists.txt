set(the_description "High-level GUI and Media I/O")
ocv_add_module(highgui opencv_imgproc OPTIONAL opencv_androidcamera)

# ----------------------------------------------------------------------------
#  CMake file for highgui. See root CMakeLists.txt
#   Some parts taken from version of Hartmut Seichter, HIT Lab NZ.
#   Jose Luis Blanco, 2008
# ----------------------------------------------------------------------------

ocv_clear_vars(GRFMT_LIBS)

if(WITH_PNG OR WITH_TIFF OR WITH_OPENEXR)
  ocv_include_directories(${ZLIB_INCLUDE_DIR})
  list(APPEND GRFMT_LIBS ${ZLIB_LIBRARIES})
endif()

if(WITH_JPEG)
  add_definitions(-DHAVE_JPEG)
  ocv_include_directories(${JPEG_INCLUDE_DIR})
  list(APPEND GRFMT_LIBS ${JPEG_LIBRARIES})
endif()

if(WITH_PNG)
  add_definitions(-DHAVE_PNG)
  add_definitions(${PNG_DEFINITIONS})
  ocv_include_directories(${PNG_INCLUDE_DIR})
  list(APPEND GRFMT_LIBS ${PNG_LIBRARIES})
endif()

if(WITH_TIFF)
  add_definitions(-DHAVE_TIFF)
  ocv_include_directories(${TIFF_INCLUDE_DIR})
  list(APPEND GRFMT_LIBS ${TIFF_LIBRARIES})
endif()

if(WITH_JASPER)
  add_definitions(-DHAVE_JASPER)
  ocv_include_directories(${JASPER_INCLUDE_DIR})
  list(APPEND GRFMT_LIBS ${JASPER_LIBRARIES})
endif()

if(WITH_OPENEXR AND OPENEXR_FOUND)
  add_definitions(-DHAVE_OPENEXR)
  ocv_include_directories(${OPENEXR_INCLUDE_PATHS})
  list(APPEND GRFMT_LIBS ${OPENEXR_LIBRARIES})
endif()

file(GLOB grfmt_hdrs src/grfmt*.hpp)
file(GLOB grfmt_srcs src/grfmt*.cpp)
list(APPEND grfmt_hdrs src/bitstrm.hpp)
list(APPEND grfmt_srcs src/bitstrm.cpp)

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
    src/precomp.cpp
    src/utils.cpp
    src/window.cpp
    )

file(GLOB highgui_ext_hdrs "include/opencv2/${name}/*.hpp" "include/opencv2/${name}/*.h")

if(HAVE_QT)
  if (HAVE_QT_OPENGL)
    set(QT_USE_QTOPENGL TRUE)
  endif()
  include(${QT_USE_FILE})

  QT4_ADD_RESOURCES(_RCC_OUTFILES src/window_QT.qrc)
  QT4_WRAP_CPP(_MOC_OUTFILES src/window_QT.h)

  list(APPEND HIGHGUI_LIBRARIES ${QT_LIBRARIES} ${QT_QTTEST_LIBRARY})
  list(APPEND highgui_srcs src/window_QT.cpp ${_MOC_OUTFILES} ${_RCC_OUTFILES} )
elseif(WIN32)
  list(APPEND highgui_srcs src/window_w32.cpp)
elseif(HAVE_GTK)
  list(APPEND highgui_srcs src/window_gtk.cpp)
elseif(APPLE)
  if(WITH_CARBON)
    add_definitions(-DHAVE_CARBON=1)
    list(APPEND highgui_srcs src/window_carbon.cpp)
    list(APPEND HIGHGUI_LIBRARIES "-framework Carbon" "-framework QuickTime")
  elseif(NOT IOS)
    add_definitions(-DHAVE_COCOA=1)
    list(APPEND highgui_srcs src/window_cocoa.mm)
    list(APPEND HIGHGUI_LIBRARIES "-framework Cocoa")
  endif()
endif()
    
if(WIN32)
  list(APPEND highgui_srcs src/cap_vfw.cpp src/cap_cmu.cpp src/cap_dshow.cpp)
endif(WIN32)

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
elseif(HAVE_CAMV4L OR HAVE_CAMV4L2)
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
  link_directories(${XIMEA_LIBRARY_DIR})
  list(APPEND HIGHGUI_LIBRARIES m3api)
endif(HAVE_XIMEA)

if(HAVE_FFMPEG)
  if(UNIX AND BZIP2_LIBRARIES)
    list(APPEND HIGHGUI_LIBRARIES ${BZIP2_LIBRARIES})
  endif()
  if(APPLE)
    list(APPEND HIGHGUI_LIBRARIES "-framework VideoDecodeAcceleration")
  endif()
endif(HAVE_FFMPEG)

if(HAVE_PVAPI)
  add_definitions(-DHAVE_PVAPI)
  ocv_include_directories(${PVAPI_INCLUDE_PATH})
  if(X86)
    set(PVAPI_SDK_SUBDIR x86)
  elseif(X86_64)
    set(PVAPI_SDK_SUBDIR x64)
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES arm)
    set(PVAPI_SDK_SUBDIR arm)
  endif()
  if(PVAPI_SDK_SUBDIR AND CMAKE_COMPILER_IS_GNUCXX)
    get_filename_component(PVAPI_EXPECTED_LIB_PATH "${PVAPI_INCLUDE_PATH}/../lib-pc/${PVAPI_SDK_SUBDIR}/${CMAKE_OPENCV_GCC_VERSION_MAJOR}.${CMAKE_OPENCV_GCC_VERSION_MINOR}" ABSOLUTE)
    link_directories(${PVAPI_EXPECTED_LIB_PATH})
  endif()
  set(highgui_srcs src/cap_pvapi.cpp ${highgui_srcs})
  set(HIGHGUI_LIBRARIES ${HIGHGUI_LIBRARIES} PvAPI)
endif()

if(WITH_IMAGEIO)
  add_definitions(-DHAVE_IMAGEIO=1)
  if(IOS)
    list(APPEND HIGHGUI_LIBRARIES "-framework ImageIO")
  endif()
  #TODO: check if need to link with some framework on OS X: -framework ApplicationServices ??
endif(WITH_IMAGEIO)

if(WITH_AVFOUNDATION)
  add_definitions(-DHAVE_AVFOUNDATION=1)
  list(APPEND highgui_srcs src/cap_avfoundation.mm)
  list(APPEND HIGHGUI_LIBRARIES "-framework AVFoundation" "-framework QuartzCore")
elseif(APPLE)
  add_definitions(-DHAVE_QUICKTIME=1)
  if(WITH_QUICKTIME)
    list(APPEND highgui_srcs src/cap_qt.cpp)
    list(APPEND HIGHGUI_LIBRARIES "-framework Carbon" "-framework QuickTime" "-framework CoreFoundation" "-framework QuartzCore")
  else()
    list(APPEND highgui_srcs src/cap_qtkit.mm)
    list(APPEND HIGHGUI_LIBRARIES "-framework QTKit" "-framework QuartzCore")
  endif()
endif()

if(WIN32)
  link_directories("${OpenCV_SOURCE_DIR}/3rdparty/lib") # for ffmpeg wrapper only
  include_directories(AFTER "${OpenCV_SOURCE_DIR}/3rdparty/include") # for directshow in VS2005 and multi-monitor support on MinGW
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

ocv_list_unique(HIGHGUI_LIBRARIES)
ocv_create_module(${GRFMT_LIBS} ${HIGHGUI_LIBRARIES})

if(BUILD_SHARED_LIBS)
  add_definitions(-DHIGHGUI_EXPORTS)
endif()

if(MSVC)
  set_target_properties(${the_module} PROPERTIES LINK_FLAGS "/NODEFAULTLIB:atlthunk.lib /NODEFAULTLIB:atlsd.lib /NODEFAULTLIB:libcmt.lib /DEBUG")
endif()

#stop automatic dependencies propagation for this module
set_target_properties(${the_module} PROPERTIES LINK_INTERFACE_LIBRARIES "")

ocv_add_precompiled_headers(${the_module})

if(CMAKE_COMPILER_IS_GNUCXX AND NOT ENABLE_NOISY_WARNINGS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
endif()

if(WIN32 AND WITH_FFMPEG)
  #copy ffmpeg dll to the output folder
  if(MSVC64 OR MINGW64)
    set(FFMPEG_SUFFIX _64)
  endif()
  
  set(ffmpeg_bare_name "opencv_ffmpeg${FFMPEG_SUFFIX}.dll")
  set(ffmpeg_bare_name_ver "opencv_ffmpeg${OPENCV_DLLVERSION}${FFMPEG_SUFFIX}.dll")
  set(ffmpeg_path "${OpenCV_SOURCE_DIR}/3rdparty/ffmpeg/${ffmpeg_bare_name}")
  
  #if(MSVC AND CMAKE_VERSION VERSION_GREATER "2.8.2")
  #  add_custom_command(TARGET ${the_module} POST_BUILD
  #                     COMMAND ${CMAKE_COMMAND} -E copy "${ffmpeg_path}" "${EXECUTABLE_OUTPUT_PATH}/$<CONFIGURATION>/${ffmpeg_bare_name_ver}"
  #                     COMMENT "Copying ${ffmpeg_path} to the output directory")
  #else
  if(MSVC_IDE)
    add_custom_command(TARGET ${the_module} POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy "${ffmpeg_path}" "${EXECUTABLE_OUTPUT_PATH}/Release/${ffmpeg_bare_name_ver}"
                       COMMAND ${CMAKE_COMMAND} -E copy "${ffmpeg_path}" "${EXECUTABLE_OUTPUT_PATH}/Debug/${ffmpeg_bare_name_ver}"
                       COMMENT "Copying ${ffmpeg_path} to the output directory")
  elseif(MSVC)
    add_custom_command(TARGET ${the_module} POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy "${ffmpeg_path}" "${EXECUTABLE_OUTPUT_PATH}/${CMAKE_BUILD_TYPE}/${ffmpeg_bare_name_ver}"
                       COMMENT "Copying ${ffmpeg_path} to the output directory")
  else()
    add_custom_command(TARGET ${the_module} POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy "${ffmpeg_path}" "${EXECUTABLE_OUTPUT_PATH}/${ffmpeg_bare_name_ver}"
                       COMMENT "Copying ${ffmpeg_path} to the output directory")
  endif()
  
  install(FILES "${ffmpeg_path}" DESTINATION bin COMPONENT main RENAME "${ffmpeg_bare_name_ver}")
endif()

ocv_add_accuracy_tests()
ocv_add_perf_tests()
