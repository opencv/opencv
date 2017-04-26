function(download_ippicv root_var)
  set(${root_var} "" PARENT_SCOPE)

  # Commit SHA in the opencv_3rdparty repo
  set(IPPICV_COMMIT "a62e20676a60ee0ad6581e217fe7e4bada3b95db")
  # Define actual ICV versions
  if(APPLE)
    set(OPENCV_ICV_PLATFORM "macosx")
    set(OPENCV_ICV_PACKAGE_SUBDIR "ippicv_mac")
    if(X86_64)
      set(OPENCV_ICV_NAME "ippicv_2017u2_mac_intel64_20170418.tgz")
      set(OPENCV_ICV_HASH "0c25953c99dbb499ff502485a9356d8d")
    else()
      set(OPENCV_ICV_NAME "ippicv_2017u2_mac_ia32_20170418.tgz")
      set(OPENCV_ICV_HASH "5f225948f3f64067c681293c098d50d8")
    endif()
  elseif((UNIX AND NOT ANDROID) OR (UNIX AND ANDROID_ABI MATCHES "x86"))
    set(OPENCV_ICV_PLATFORM "linux")
    set(OPENCV_ICV_PACKAGE_SUBDIR "ippicv_lnx")
    if(X86_64)
      set(OPENCV_ICV_NAME "ippicv_2017u2_lnx_intel64_20170418.tgz")
      set(OPENCV_ICV_HASH "87cbdeb627415d8e4bc811156289fa3a")
    else()
      set(OPENCV_ICV_NAME "ippicv_2017u2_lnx_ia32_20170418.tgz")
      set(OPENCV_ICV_HASH "f2cece00d802d4dea86df52ed095257e")
    endif()
  elseif(WIN32 AND NOT ARM)
    set(OPENCV_ICV_PLATFORM "windows")
    set(OPENCV_ICV_PACKAGE_SUBDIR "ippicv_win")
    if(X86_64)
      set(OPENCV_ICV_NAME "ippicv_2017u2_win_intel64_20170418.zip")
      set(OPENCV_ICV_HASH "75060a0c662c0800f48995b7e9b085f6")
    else()
      set(OPENCV_ICV_NAME "ippicv_2017u2_win_ia32_20170418.zip")
      set(OPENCV_ICV_HASH "60fcf3ccd9a2ebc9e432ffb5cb91638b")
    endif()
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
               ID IPPICV
               STATUS res
               UNPACK RELATIVE_URL)

  if(res)
    set(${root_var} "${THE_ROOT}/${OPENCV_ICV_PACKAGE_SUBDIR}" PARENT_SCOPE)
  endif()
endfunction()
