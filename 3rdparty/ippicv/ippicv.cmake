function(download_ippicv root_var)
  set(${root_var} "" PARENT_SCOPE)

  # Commit SHA in the opencv_3rdparty repo
  set(IPPICV_COMMIT "81a676001ca8075ada498583e4166079e5744668")
  # Define actual ICV versions
  if(APPLE)
    set(OPENCV_ICV_NAME "ippicv_macosx_20151201.tgz")
    set(OPENCV_ICV_HASH "4ff1fde9a7cfdfe7250bfcd8334e0f2f")
    set(OPENCV_ICV_PLATFORM "macosx")
    set(OPENCV_ICV_PACKAGE_SUBDIR "ippicv_osx")
  elseif((UNIX AND NOT ANDROID) OR (UNIX AND ANDROID_ABI MATCHES "x86"))
    set(OPENCV_ICV_NAME "ippicv_linux_20151201.tgz")
    set(OPENCV_ICV_HASH "808b791a6eac9ed78d32a7666804320e")
    set(OPENCV_ICV_PLATFORM "linux")
    set(OPENCV_ICV_PACKAGE_SUBDIR "ippicv_lnx")
  elseif(WIN32 AND NOT ARM)
    set(OPENCV_ICV_NAME "ippicv_windows_20151201.zip")
    set(OPENCV_ICV_HASH "04e81ce5d0e329c3fbc606ae32cad44d")
    set(OPENCV_ICV_PLATFORM "windows")
    set(OPENCV_ICV_PACKAGE_SUBDIR "ippicv_win")
  else()
    return()
  endif()

  set(THE_ROOT "${OpenCV_BINARY_DIR}/3rdparty/ippicv")
  ocv_download(FILENAME ${OPENCV_ICV_NAME}
               HASH ${OPENCV_ICV_HASH}
               URL
                 "${OPENCV_IPPICV_URL}"
                 "$ENV{OPENCV_IPPICV_URL}"
                 "https://raw.githubusercontent.com/opencv/opencv_3rdparty/${IPPICV_COMMIT}/ippicv/"
               DESTINATION_DIR "${THE_ROOT}"
               STATUS res
               UNPACK RELATIVE_URL)

  if(res)
    set(${root_var} "${THE_ROOT}/${OPENCV_ICV_PACKAGE_SUBDIR}" PARENT_SCOPE)
  endif()
endfunction()
