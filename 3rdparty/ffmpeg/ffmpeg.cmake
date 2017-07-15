# Binary branch name: ffmpeg/master_20170704
# Binaries were created for OpenCV: f670a9927026629a4083e05a1612f0adcad7727e
set(FFMPEG_BINARIES_COMMIT "a86e53eb35737a50e5100e26af3aa1d29e810890")
set(FFMPEG_FILE_HASH_BIN32 "79c35cc654778e66237444bc562afbca")
set(FFMPEG_FILE_HASH_BIN64 "0dc72775ec3c14d1e049f51dc1280dbb")
set(FFMPEG_FILE_HASH_CMAKE "ec59008da403fb18ab3c1ed66aed583b")

function(download_win_ffmpeg script_var)
  set(${script_var} "" PARENT_SCOPE)

  set(ids BIN32 BIN64 CMAKE)
  set(name_BIN32 "opencv_ffmpeg.dll")
  set(name_BIN64 "opencv_ffmpeg_64.dll")
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
