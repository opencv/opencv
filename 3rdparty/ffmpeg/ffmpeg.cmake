# Binary branch name: ffmpeg/master_20171009
# Binaries were created for OpenCV: 8ac2c5d620b467d3f22802e96c88ddde6da707af
set(FFMPEG_BINARIES_COMMIT "66b1fed06cf3510235f367f96aa26da5cb234a15")
set(FFMPEG_FILE_HASH_BIN32 "3ae76b105113d944984b2351c61e21c6")
set(FFMPEG_FILE_HASH_BIN64 "cf3bb5bc9d393b022ea7a42eb63e794d")
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
