# Binaries branch name: ffmpeg/master_20200609
# Binaries were created for OpenCV: 5f3012fc9afbffbf53a38f1468523d3454b3e2f6
ocv_update(FFMPEG_BINARIES_COMMIT "1df9bf0c0c6c2cf225bd3d8e4cf5985198352454")
ocv_update(FFMPEG_FILE_HASH_BIN32 "854b3460c435d04277e1f1ecc06cb809")
ocv_update(FFMPEG_FILE_HASH_BIN64 "3a46d6356220796e044817ae3a21cc31")
ocv_update(FFMPEG_FILE_HASH_CMAKE "ad57c038ba34b868277ccbe6dd0f9602")

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
