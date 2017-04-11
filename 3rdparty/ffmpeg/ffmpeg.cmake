function(download_win_ffmpeg script_var)
  set(${script_var} "" PARENT_SCOPE)

  set(ids BIN32 BIN64 CMAKE)
  set(name_BIN32 "opencv_ffmpeg.dll")
  set(name_BIN64 "opencv_ffmpeg_64.dll")
  set(name_CMAKE "ffmpeg_version.cmake")

  # Binary branch name: ffmpeg/master_20161202
  # Binaries were created for OpenCV: 594c136d1fcbb5816c57516e50f9cbeffbd90835
  set(FFMPEG_BINARIES_COMMIT "2a19d0006415955c79431116e4634f04d5eb5a74")
  set(FFMPEG_FILE_HASH_BIN32 "f081abd9d6ca7e425d340ce586f9c090")
  set(FFMPEG_FILE_HASH_BIN64 "a423363a6eb76d362ca6c406c96c8db6")
  set(FFMPEG_FILE_HASH_CMAKE "5346ae1854fc7aa569a722e85af480ec")

  set(FFMPEG_BINARIES_COMMIT "2a19d0006415955c79431116e4634f04d5eb5a74")
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
