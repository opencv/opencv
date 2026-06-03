# Binaries branch name: ffmpeg/5.x_20260602
# Binaries were created for OpenCV: a0a660fcb1e58a295e6caa6aee64ed4d369b0181
ocv_update(FFMPEG_BINARIES_COMMIT "06dc20cad65dc7fcf784f70c95d46750520889a7")
ocv_update(FFMPEG_FILE_HASH_BIN32 "9cef7a78b6f7ec8cf1a3935c058cfac5")
ocv_update(FFMPEG_FILE_HASH_BIN64 "a821a1135251859655090c795af05789")
ocv_update(FFMPEG_FILE_HASH_CMAKE "e09efc33312d1173be8a9446f3b088fe")

function(download_win_ffmpeg script_var)
  set(${script_var} "" PARENT_SCOPE)

  set(ids BIN32 BIN64 CMAKE)
  set(name_BIN32 "opencv_videoio_ffmpeg.dll")
  set(name_BIN64 "opencv_videoio_ffmpeg_64.dll")
  set(name_CMAKE "ffmpeg_version.cmake")

  set(FFMPEG_DOWNLOAD_DIR "${OpenCV_BINARY_DIR}/3rdparty/ffmpeg")

  set(status TRUE)
  foreach(id ${ids})
    ocv_download(FILENAME ${name_${id}}
               HASH ${FFMPEG_FILE_HASH_${id}}
               URL
                 "$ENV{OPENCV_FFMPEG_URL}"
                 "${OPENCV_FFMPEG_URL}"
                 "https://raw.githubusercontent.com/opencv/opencv_3rdparty/${FFMPEG_BINARIES_COMMIT}/ffmpeg/"
               DESTINATION_DIR ${FFMPEG_DOWNLOAD_DIR}
               ID FFMPEG
               RELATIVE_URL
               STATUS res)
    if(NOT res)
      set(status FALSE)
    endif()
  endforeach()
  if(status)
    set(${script_var} "${FFMPEG_DOWNLOAD_DIR}/ffmpeg_version.cmake" PARENT_SCOPE)
  endif()
endfunction()

if(OPENCV_INSTALL_FFMPEG_DOWNLOAD_SCRIPT)
  configure_file("${CMAKE_CURRENT_LIST_DIR}/ffmpeg-download.ps1.in" "${CMAKE_BINARY_DIR}/win-install/ffmpeg-download.ps1" @ONLY)
  install(FILES "${CMAKE_BINARY_DIR}/win-install/ffmpeg-download.ps1" DESTINATION "." COMPONENT libs)
endif()

ocv_install_3rdparty_licenses(ffmpeg license.txt readme.txt)
