# Binary branch name: ffmpeg/master_20170418
# Binaries were created for OpenCV: b993b9b7c7f6f5f37d10acacb2962812228410ba
set(FFMPEG_BINARIES_COMMIT "86c4a841055f2612774e85b4292bb20e5fe8a783")
set(FFMPEG_FILE_HASH_BIN32 "3dea5f7f009b44601fe95728328e0f9e")
set(FFMPEG_FILE_HASH_BIN64 "9debe701975ef074bd6661981f3f0716")
set(FFMPEG_FILE_HASH_CMAKE "208c00f03d2f6f39fa6262649e0bfc8d")

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
