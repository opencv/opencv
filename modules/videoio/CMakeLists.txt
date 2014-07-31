set(the_description "Media I/O")
ocv_add_module(videoio opencv_imgproc opencv_imgcodecs OPTIONAL opencv_androidcamera)

# ----------------------------------------------------------------------------
#  CMake file for videoio. See root CMakeLists.txt
#   Some parts taken from version of Hartmut Seichter, HIT Lab NZ.
#   Jose Luis Blanco, 2008
# ----------------------------------------------------------------------------

if(HAVE_WINRT_CX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /ZW")
endif()

if(APPLE)
  ocv_include_directories(${ZLIB_INCLUDE_DIRS})
  list(APPEND VIDEOIO_LIBRARIES ${ZLIB_LIBRARIES})
endif()

set(videoio_hdrs
    src/precomp.hpp
    src/cap_ffmpeg_impl.hpp
    )

set(videoio_srcs
    src/cap.cpp
    src/cap_images.cpp
    src/cap_ffmpeg.cpp
    )

file(GLOB videoio_ext_hdrs "include/opencv2/*.hpp" "include/opencv2/${name}/*.hpp" "include/opencv2/${name}/*.h")

if(WIN32 AND NOT ARM)
  list(APPEND videoio_srcs src/cap_cmu.cpp)
endif()

if (WIN32 AND HAVE_DSHOW)
  list(APPEND videoio_srcs src/cap_dshow.cpp)
  list(APPEND videoio_hdrs src/cap_dshow.hpp)
endif()

if (WIN32 AND HAVE_MSMF)
  list(APPEND videoio_srcs src/cap_msmf.cpp)
endif()

if (WIN32 AND HAVE_VFW)
  list(APPEND videoio_srcs src/cap_vfw.cpp)
endif()

if(HAVE_XINE)
  list(APPEND videoio_srcs src/cap_xine.cpp)
endif(HAVE_XINE)

if(HAVE_DC1394_2)
  list(APPEND videoio_srcs src/cap_dc1394_v2.cpp)
endif(HAVE_DC1394_2)

if(HAVE_DC1394)
  list(APPEND videoio_srcs src/cap_dc1394.cpp)
endif(HAVE_DC1394)

if(HAVE_GSTREAMER)
  list(APPEND videoio_srcs src/cap_gstreamer.cpp)
endif(HAVE_GSTREAMER)

if(HAVE_UNICAP)
  list(APPEND videoio_srcs src/cap_unicap.cpp)
endif(HAVE_UNICAP)

if(HAVE_LIBV4L)
  list(APPEND videoio_srcs src/cap_libv4l.cpp)
elseif(HAVE_CAMV4L OR HAVE_CAMV4L2 OR HAVE_VIDEOIO)
  list(APPEND videoio_srcs src/cap_v4l.cpp)
endif()

if(HAVE_OPENNI)
  list(APPEND videoio_srcs src/cap_openni.cpp)
  ocv_include_directories(${OPENNI_INCLUDE_DIR})
  list(APPEND VIDEOIO_LIBRARIES ${OPENNI_LIBRARY})
endif(HAVE_OPENNI)

if(HAVE_OPENNI2)
  list(APPEND videoio_srcs src/cap_openni2.cpp)
  ocv_include_directories(${OPENNI2_INCLUDE_DIR})
  list(APPEND VIDEOIO_LIBRARIES ${OPENNI2_LIBRARY})
endif(HAVE_OPENNI2)

if(HAVE_opencv_androidcamera)
  list(APPEND videoio_srcs src/cap_android.cpp)
  add_definitions(-DHAVE_ANDROID_NATIVE_CAMERA)#TODO: remove this line
endif(HAVE_opencv_androidcamera)

if(HAVE_XIMEA)
  list(APPEND videoio_srcs src/cap_ximea.cpp)
  ocv_include_directories(${XIMEA_PATH})
  if(XIMEA_LIBRARY_DIR)
    link_directories("${XIMEA_LIBRARY_DIR}")
  endif()
  if(X86_64)
    list(APPEND VIDEOIO_LIBRARIES m3apiX64)
  else()
    list(APPEND VIDEOIO_LIBRARIES m3api)
  endif()
endif(HAVE_XIMEA)

if(HAVE_FFMPEG)
  if(UNIX AND BZIP2_LIBRARIES)
    list(APPEND VIDEOIO_LIBRARIES ${BZIP2_LIBRARIES})
  endif()
  if(APPLE)
    list(APPEND VIDEOIO_LIBRARIES "-framework VideoDecodeAcceleration" bz2)
  endif()
endif(HAVE_FFMPEG)

if(HAVE_PVAPI)
  add_definitions(-DHAVE_PVAPI)
  add_definitions(${PVAPI_DEFINITIONS})
  ocv_include_directories(${PVAPI_INCLUDE_PATH})
  set(videoio_srcs src/cap_pvapi.cpp ${videoio_srcs})
  list(APPEND VIDEOIO_LIBRARIES ${PVAPI_LIBRARY})
endif()

if(HAVE_GIGE_API)
  add_definitions(-DHAVE_GIGE_API)
  ocv_include_directories(${GIGEAPI_INCLUDE_PATH})
  set(videoio_srcs src/cap_giganetix.cpp ${videoio_srcs})
  list(APPEND VIDEOIO_LIBRARIES ${GIGEAPI_LIBRARIES})
  list(APPEND videoio_srcs src/cap_giganetix.cpp)
endif(HAVE_GIGE_API)

if(HAVE_AVFOUNDATION)
  list(APPEND videoio_srcs src/cap_avfoundation.mm)
  list(APPEND VIDEOIO_LIBRARIES "-framework AVFoundation" "-framework QuartzCore")
endif()

if(HAVE_QUICKTIME)
  list(APPEND videoio_srcs src/cap_qt.cpp)
  list(APPEND VIDEOIO_LIBRARIES "-framework Carbon" "-framework QuickTime" "-framework CoreFoundation" "-framework QuartzCore")
elseif(HAVE_QTKIT)
  list(APPEND videoio_srcs src/cap_qtkit.mm)
  list(APPEND VIDEOIO_LIBRARIES "-framework QTKit" "-framework QuartzCore" "-framework AppKit")
endif()

if(HAVE_INTELPERC)
  list(APPEND videoio_srcs src/cap_intelperc.cpp)
  ocv_include_directories(${INTELPERC_INCLUDE_DIR})
  list(APPEND VIDEOIO_LIBRARIES ${INTELPERC_LIBRARIES})
endif(HAVE_INTELPERC)

if(IOS)
  add_definitions(-DHAVE_IOS=1)
  list(APPEND videoio_srcs src/ios_conversions.mm src/cap_ios_abstract_camera.mm src/cap_ios_photo_camera.mm src/cap_ios_video_camera.mm)
  list(APPEND VIDEOIO_LIBRARIES "-framework Accelerate" "-framework AVFoundation" "-framework CoreGraphics" "-framework CoreImage" "-framework CoreMedia" "-framework CoreVideo" "-framework QuartzCore" "-framework AssetsLibrary")
endif()

if(WIN32)
  link_directories("${OpenCV_SOURCE_DIR}/3rdparty/lib") # for ffmpeg wrapper only
  include_directories(AFTER SYSTEM "${OpenCV_SOURCE_DIR}/3rdparty/include") # for directshow in VS2005 and multi-monitor support on MinGW
  include_directories(AFTER SYSTEM "${OpenCV_SOURCE_DIR}/3rdparty/include/ffmpeg_") # for tests
endif()

if(UNIX)
  #these variables are set by CHECK_MODULE macro
  foreach(P ${VIDEOIO_INCLUDE_DIRS})
    ocv_include_directories(${P})
  endforeach()

  foreach(P ${VIDEOIO_LIBRARY_DIRS})
    link_directories(${P})
  endforeach()
endif()

source_group("Src" FILES ${videoio_srcs} ${videoio_hdrs})
source_group("Include" FILES ${videoio_ext_hdrs})
ocv_set_module_sources(HEADERS ${videoio_ext_hdrs} SOURCES ${videoio_srcs} ${videoio_hdrs})
ocv_module_include_directories()

ocv_create_module(${VIDEOIO_LIBRARIES})

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
  add_definitions(-DVIDEOIO_EXPORTS)
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
