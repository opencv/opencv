set(VIDEOIO_ENABLE_PLUGINS_DEFAULT ON)
if(EMSCRIPTEN OR IOS OR XROS OR WINRT)
  set(VIDEOIO_ENABLE_PLUGINS_DEFAULT OFF)
endif()

set(VIDEOIO_PLUGIN_LIST "" CACHE STRING "List of videoio backends to be compiled as plugins (ffmpeg, gstreamer, mfx, msmf or special value 'all')")
set(VIDEOIO_ENABLE_PLUGINS "${VIDEOIO_ENABLE_PLUGINS_DEFAULT}" CACHE BOOL "Allow building and using of videoio plugins")
mark_as_advanced(VIDEOIO_PLUGIN_LIST VIDEOIO_ENABLE_PLUGINS)

string(REPLACE "," ";" VIDEOIO_PLUGIN_LIST "${VIDEOIO_PLUGIN_LIST}")  # support comma-separated list (,) too
string(TOLOWER "${VIDEOIO_PLUGIN_LIST}" VIDEOIO_PLUGIN_LIST)
if(NOT VIDEOIO_ENABLE_PLUGINS)
  if(VIDEOIO_PLUGIN_LIST)
    message(WARNING "VideoIO: plugins are disabled through VIDEOIO_ENABLE_PLUGINS, so VIDEOIO_PLUGIN_LIST='${VIDEOIO_PLUGIN_LIST}' is ignored")
    set(VIDEOIO_PLUGIN_LIST "")
  endif()
else()
  # Make virtual opencv_videoio_plugins target
  if(NOT TARGET opencv_videoio_plugins)
    add_custom_target(opencv_videoio_plugins ALL)
  endif()
endif()

ocv_add_module(videoio opencv_imgproc opencv_imgcodecs WRAP java objc python)

set(videoio_hdrs ${CMAKE_CURRENT_LIST_DIR}/src/precomp.hpp)

set(videoio_srcs
  "${CMAKE_CURRENT_LIST_DIR}/src/videoio_registry.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/src/videoio_c.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/src/cap.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/src/cap_images.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/src/cap_mjpeg_encoder.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/src/cap_mjpeg_decoder.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/src/backend_plugin.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/src/backend_static.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/src/container_avi.cpp")

file(GLOB videoio_ext_hdrs
  "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/*.hpp"
  "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/*.hpp"
  "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/*.h"
  "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/legacy/*.h")

if(OPENCV_DEBUG_POSTFIX)
  ocv_append_source_file_compile_definitions("${CMAKE_CURRENT_LIST_DIR}/src/backend_plugin.cpp" "DEBUG_POSTFIX=${OPENCV_DEBUG_POSTFIX}")
endif()

# Removing WinRT API headers by default
list(REMOVE_ITEM videoio_ext_hdrs "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/cap_winrt.hpp")

# Remove iOS API header by default
list(REMOVE_ITEM videoio_ext_hdrs "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/cap_ios.h")

if(DEFINED WINRT AND NOT DEFINED ENABLE_WINRT_MODE_NATIVE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /ZW")
endif()

# Dependencies used by the implementation referenced
# below are not available on WinRT 8.0.
# Enabling it for WiRT 8.1+ only.
if(DEFINED WINRT AND NOT DEFINED WINRT_8_0 AND NOT DEFINED ENABLE_WINRT_MODE_NATIVE)

    # WinRT detected. Adding WinRT API header
    message(STATUS "  ${name}: WinRT detected. Adding WinRT API header")
    list(APPEND videoio_ext_hdrs "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/cap_winrt.hpp")

    # Adding WinRT internal sources and headers
    list(APPEND videoio_srcs
        ${CMAKE_CURRENT_LIST_DIR}/src/cap_winrt_capture.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/cap_winrt_bridge.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/cap_winrt_video.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/cap_winrt/CaptureFrameGrabber.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/cap_winrt/MediaStreamSink.cpp)
    list(APPEND videoio_hdrs
        ${CMAKE_CURRENT_LIST_DIR}/src/cap_winrt_capture.hpp
        ${CMAKE_CURRENT_LIST_DIR}/src/cap_winrt_bridge.hpp
        ${CMAKE_CURRENT_LIST_DIR}/src/cap_winrt_video.hpp
        ${CMAKE_CURRENT_LIST_DIR}/src/cap_winrt/MFIncludes.hpp
        ${CMAKE_CURRENT_LIST_DIR}/src/cap_winrt/CaptureFrameGrabber.hpp
        ${CMAKE_CURRENT_LIST_DIR}/src/cap_winrt/MediaSink.hpp
        ${CMAKE_CURRENT_LIST_DIR}/src/cap_winrt/MediaStreamSink.hpp)
endif()

include(${CMAKE_CURRENT_LIST_DIR}/cmake/plugin.cmake)

set(tgts "PRIVATE")

if(TARGET ocv.3rdparty.mediasdk)
  if("mfx" IN_LIST VIDEOIO_PLUGIN_LIST OR VIDEOIO_PLUGIN_LIST STREQUAL "all")
    ocv_create_builtin_videoio_plugin("opencv_videoio_intel_mfx" ocv.3rdparty.mediasdk "cap_mfx_common.cpp" "cap_mfx_reader.cpp" "cap_mfx_writer.cpp" "cap_mfx_plugin.cpp")
  else()
    list(APPEND videoio_srcs
      ${CMAKE_CURRENT_LIST_DIR}/src/cap_mfx_common.cpp
      ${CMAKE_CURRENT_LIST_DIR}/src/cap_mfx_reader.cpp
      ${CMAKE_CURRENT_LIST_DIR}/src/cap_mfx_writer.cpp)
    list(APPEND videoio_hdrs
      ${CMAKE_CURRENT_LIST_DIR}/src/cap_mfx_common.hpp
      ${CMAKE_CURRENT_LIST_DIR}/src/cap_mfx_reader.hpp
      ${CMAKE_CURRENT_LIST_DIR}/src/cap_mfx_writer.hpp)
    list(APPEND tgts ocv.3rdparty.mediasdk)
  endif()
endif()

if(TARGET ocv.3rdparty.dshow)
  list(APPEND videoio_srcs ${CMAKE_CURRENT_LIST_DIR}/src/cap_dshow.cpp)
  list(APPEND videoio_hdrs ${CMAKE_CURRENT_LIST_DIR}/src/cap_dshow.hpp)
  list(APPEND tgts ocv.3rdparty.dshow)
endif()

if(TARGET ocv.3rdparty.msmf)
  if("msmf" IN_LIST VIDEOIO_PLUGIN_LIST OR VIDEOIO_PLUGIN_LIST STREQUAL "all")
    ocv_create_builtin_videoio_plugin("opencv_videoio_msmf" ocv.3rdparty.msmf "cap_msmf.cpp")
  else()
    list(APPEND videoio_srcs ${CMAKE_CURRENT_LIST_DIR}/src/cap_msmf.hpp)
    list(APPEND videoio_srcs ${CMAKE_CURRENT_LIST_DIR}/src/cap_msmf.cpp)
    list(APPEND tgts ocv.3rdparty.msmf)
  endif()
endif()

if(TARGET ocv.3rdparty.xine)
  list(APPEND videoio_srcs ${CMAKE_CURRENT_LIST_DIR}/src/cap_xine.cpp)
  list(APPEND tgts ocv.3rdparty.xine)
endif()

if(TARGET ocv.3rdparty.dc1394_2)
  list(APPEND videoio_srcs ${CMAKE_CURRENT_LIST_DIR}/src/cap_dc1394_v2.cpp)
  list(APPEND tgts ocv.3rdparty.dc1394_2)
endif()

if(TARGET ocv.3rdparty.gstreamer)
  if("gstreamer" IN_LIST VIDEOIO_PLUGIN_LIST OR VIDEOIO_PLUGIN_LIST STREQUAL "all")
    ocv_create_builtin_videoio_plugin("opencv_videoio_gstreamer" ocv.3rdparty.gstreamer "cap_gstreamer.cpp")
  else()
    list(APPEND videoio_srcs ${CMAKE_CURRENT_LIST_DIR}/src/cap_gstreamer.cpp)
    list(APPEND tgts ocv.3rdparty.gstreamer)
  endif()
endif()

if(TARGET ocv.3rdparty.v4l)
  list(APPEND videoio_srcs ${CMAKE_CURRENT_LIST_DIR}/src/cap_v4l.cpp)
  list(APPEND tgts ocv.3rdparty.v4l)
endif()

if(TARGET ocv.3rdparty.openni2)
  list(APPEND videoio_srcs ${CMAKE_CURRENT_LIST_DIR}/src/cap_openni2.cpp)
  list(APPEND tgts ocv.3rdparty.openni2)
endif()

if(TARGET ocv.3rdparty.ximea)
  list(APPEND videoio_srcs ${CMAKE_CURRENT_LIST_DIR}/src/cap_ximea.cpp)
  list(APPEND tgts ocv.3rdparty.ximea)
endif()

if(TARGET ocv.3rdparty.ueye)
  if("ueye" IN_LIST VIDEOIO_PLUGIN_LIST OR VIDEOIO_PLUGIN_LIST STREQUAL "all")
    ocv_create_builtin_videoio_plugin("opencv_videoio_ueye" ocv.3rdparty.ueye "cap_ueye.cpp")
  else()
    list(APPEND videoio_srcs ${CMAKE_CURRENT_LIST_DIR}/src/cap_ueye.cpp)
    list(APPEND tgts ocv.3rdparty.ueye)
  endif()
endif()

if(TARGET ocv.3rdparty.ffmpeg)
  if(HAVE_FFMPEG_WRAPPER)
    list(APPEND tgts ocv.3rdparty.ffmpeg)
  elseif("ffmpeg" IN_LIST VIDEOIO_PLUGIN_LIST OR VIDEOIO_PLUGIN_LIST STREQUAL "all")
    ocv_create_builtin_videoio_plugin("opencv_videoio_ffmpeg" ocv.3rdparty.ffmpeg "cap_ffmpeg.cpp")
    if(TARGET ocv.3rdparty.ffmpeg.plugin_deps)
      ocv_target_link_libraries(opencv_videoio_ffmpeg ocv.3rdparty.ffmpeg.plugin_deps)
    endif()
    if(TARGET ocv.3rdparty.mediasdk
        AND NOT OPENCV_FFMPEG_DISABLE_MEDIASDK
    )
      ocv_target_link_libraries(opencv_videoio_ffmpeg ocv.3rdparty.mediasdk)
    endif()
  else()
    list(APPEND videoio_hdrs ${CMAKE_CURRENT_LIST_DIR}/src/cap_ffmpeg_impl.hpp)
    list(APPEND videoio_srcs ${CMAKE_CURRENT_LIST_DIR}/src/cap_ffmpeg.cpp)
    list(APPEND tgts ocv.3rdparty.ffmpeg)
    if(TARGET ocv.3rdparty.ffmpeg.builtin_deps)
      list(APPEND tgts ocv.3rdparty.ffmpeg.builtin_deps)
    endif()
    if(TARGET ocv.3rdparty.mediasdk
        AND NOT OPENCV_FFMPEG_DISABLE_MEDIASDK
    )
      list(APPEND tgts ocv.3rdparty.mediasdk)
    endif()
  endif()
endif()

if(TARGET ocv.3rdparty.pvapi)
  set(videoio_srcs ${CMAKE_CURRENT_LIST_DIR}/src/cap_pvapi.cpp ${videoio_srcs})
  list(APPEND tgts ocv.3rdparty.pvapi)
endif()

if(TARGET ocv.3rdparty.aravis)
  list(APPEND videoio_srcs ${CMAKE_CURRENT_LIST_DIR}/src/cap_aravis.cpp)
  list(APPEND tgts ocv.3rdparty.aravis)
endif()

if(TARGET ocv.3rdparty.avfoundation)
  if(IOS OR XROS)
    list(APPEND videoio_srcs ${CMAKE_CURRENT_LIST_DIR}/src/cap_avfoundation.mm)
  else()
    list(APPEND videoio_srcs ${CMAKE_CURRENT_LIST_DIR}/src/cap_avfoundation_mac.mm)
  endif()
  list(APPEND tgts ocv.3rdparty.avfoundation)
endif()

if(TARGET ocv.3rdparty.librealsense)
  list(APPEND videoio_srcs ${CMAKE_CURRENT_LIST_DIR}/src/cap_librealsense.cpp)
  list(APPEND tgts ocv.3rdparty.librealsense)
endif()

if(TARGET ocv.3rdparty.gphoto2)
  list(APPEND videoio_srcs ${CMAKE_CURRENT_LIST_DIR}/src/cap_gphoto2.cpp)
  list(APPEND tgts ocv.3rdparty.gphoto2)
endif()

if(TARGET ocv.3rdparty.cap_ios)
  list(APPEND videoio_srcs
    ${CMAKE_CURRENT_LIST_DIR}/src/cap_ios_abstract_camera.mm
    ${CMAKE_CURRENT_LIST_DIR}/src/cap_ios_photo_camera.mm
    ${CMAKE_CURRENT_LIST_DIR}/src/cap_ios_video_camera.mm)
  list(APPEND tgts ocv.3rdparty.cap_ios)
  list(APPEND videoio_ext_hdrs "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/cap_ios.h")
endif()

if(TARGET ocv.3rdparty.android_mediandk)
  list(APPEND videoio_srcs
    ${CMAKE_CURRENT_LIST_DIR}/src/cap_android_mediandk.cpp)
  list(APPEND tgts ocv.3rdparty.android_mediandk)
endif()

if(TARGET ocv.3rdparty.android_native_camera)
  list(APPEND videoio_srcs
    ${CMAKE_CURRENT_LIST_DIR}/src/cap_android_camera.cpp)
  list(APPEND tgts ocv.3rdparty.android_native_camera)
endif()

if(TARGET ocv.3rdparty.obsensor)
if(OBSENSOR_USE_ORBBEC_SDK)
    list(APPEND videoio_srcs
      ${CMAKE_CURRENT_LIST_DIR}/src/cap_obsensor_liborbbec.hpp
      ${CMAKE_CURRENT_LIST_DIR}/src/cap_obsensor_liborbbec.cpp)
  else()
    list(APPEND videoio_srcs
      ${CMAKE_CURRENT_LIST_DIR}/src/cap_obsensor_capture.cpp)
    list(APPEND videoio_hdrs
      ${CMAKE_CURRENT_LIST_DIR}/src/cap_obsensor_capture.hpp
      ${CMAKE_CURRENT_LIST_DIR}/src/cap_obsensor/obsensor_stream_channel_interface.hpp)
    if(HAVE_OBSENSOR_MSMF)
      list(APPEND videoio_srcs
        ${CMAKE_CURRENT_LIST_DIR}/src/cap_obsensor/obsensor_uvc_stream_channel.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/cap_obsensor/obsensor_stream_channel_msmf.cpp)
      list(APPEND videoio_hdrs
        ${CMAKE_CURRENT_LIST_DIR}/src/cap_obsensor/obsensor_uvc_stream_channel.hpp
        ${CMAKE_CURRENT_LIST_DIR}/src/cap_obsensor/obsensor_stream_channel_msmf.hpp)
    elseif(HAVE_OBSENSOR_V4L2)
      list(APPEND videoio_srcs
        ${CMAKE_CURRENT_LIST_DIR}/src/cap_obsensor/obsensor_uvc_stream_channel.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/cap_obsensor/obsensor_stream_channel_v4l2.cpp)
      list(APPEND videoio_hdrs
        ${CMAKE_CURRENT_LIST_DIR}/src/cap_obsensor/obsensor_uvc_stream_channel.hpp
        ${CMAKE_CURRENT_LIST_DIR}/src/cap_obsensor/obsensor_stream_channel_v4l2.hpp)
    endif()
  endif()
  list(APPEND tgts ocv.3rdparty.obsensor)
endif()

if(tgts STREQUAL "PRIVATE")
  set(tgts "")
endif()

ocv_install_used_external_targets(${tgts})

ocv_set_module_sources(HEADERS ${videoio_ext_hdrs} ${videoio_hdrs} SOURCES ${videoio_srcs})
ocv_module_include_directories()
ocv_create_module()
ocv_add_accuracy_tests(${tgts})
ocv_add_perf_tests(${tgts})

if(VIDEOIO_ENABLE_PLUGINS)
  ocv_target_compile_definitions(${the_module} PRIVATE ENABLE_PLUGINS)
endif()

ocv_target_link_libraries(${the_module} LINK_PRIVATE ${tgts})

# copy FFmpeg dll to the output folder
if(WIN32 AND HAVE_FFMPEG_WRAPPER)
  if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(FFMPEG_SUFFIX _64)
  endif()
  set(ffmpeg_dir "${OpenCV_BINARY_DIR}/3rdparty/ffmpeg")
  set(ffmpeg_bare_name "opencv_videoio_ffmpeg${FFMPEG_SUFFIX}.dll")
  set(ffmpeg_bare_name_ver "opencv_videoio_ffmpeg${OPENCV_DLLVERSION}${FFMPEG_SUFFIX}.dll")
  set(ffmpeg_path "${ffmpeg_dir}/${ffmpeg_bare_name}")
  if(MSVC_IDE)
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ffmpeg_path}" "${EXECUTABLE_OUTPUT_PATH}/Release/${ffmpeg_bare_name_ver}"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ffmpeg_path}" "${EXECUTABLE_OUTPUT_PATH}/Debug/${ffmpeg_bare_name_ver}")
  elseif(MSVC AND (CMAKE_GENERATOR MATCHES "Visual"))
    execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ffmpeg_path}" "${EXECUTABLE_OUTPUT_PATH}/${CMAKE_BUILD_TYPE}/${ffmpeg_bare_name_ver}")
  else()
    execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ffmpeg_path}" "${EXECUTABLE_OUTPUT_PATH}/${ffmpeg_bare_name_ver}")
  endif()
  install(FILES "${ffmpeg_path}" DESTINATION ${OPENCV_BIN_INSTALL_PATH} COMPONENT libs RENAME "${ffmpeg_bare_name_ver}")
  if(INSTALL_CREATE_DISTRIB)
    install(FILES "${ffmpeg_dir}/opencv_videoio_ffmpeg${FFMPEG_SUFFIX}.dll" DESTINATION "bin/" COMPONENT libs RENAME "opencv_videoio_ffmpeg${OPENCV_DLLVERSION}${FFMPEG_SUFFIX}.dll")
  endif()
endif()
