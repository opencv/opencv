function(download_ippicv root_var)
  set(${root_var} "" PARENT_SCOPE)

  # Commit SHA in the opencv_3rdparty repo
  set(IPPICV_COMMIT "c7c6d527dde5fee7cb914ee9e4e20f7436aab3a1")
  # Define actual ICV versions
  if(APPLE)
    set(OPENCV_ICV_PLATFORM "macosx")
    set(OPENCV_ICV_PACKAGE_SUBDIR "ippicv_mac")
    set(OPENCV_ICV_NAME "ippicv_2021.9.1_mac_intel64_20230919_general.tgz")
    set(OPENCV_ICV_HASH "14f01c5a4780bfae9dde9b0aaf5e56fc")
  elseif((UNIX AND NOT ANDROID) OR (UNIX AND ANDROID_ABI MATCHES "x86"))
    set(OPENCV_ICV_PLATFORM "linux")
    set(OPENCV_ICV_PACKAGE_SUBDIR "ippicv_lnx")
    if(X86_64)
      set(OPENCV_ICV_NAME "ippicv_2021.10.1_lnx_intel64_20231206_general.tgz")
      set(OPENCV_ICV_HASH "90884d3b9508f31f6a154165591b8b0b")
    else()
      set(OPENCV_ICV_NAME "ippicv_2021.10.1_lnx_ia32_20231206_general.tgz")
      set(OPENCV_ICV_HASH "d9510f3ce08f6074aac472a5c19a3b53")
    endif()
  elseif(WIN32 AND NOT ARM)
    set(OPENCV_ICV_PLATFORM "windows")
    set(OPENCV_ICV_PACKAGE_SUBDIR "ippicv_win")
    if(X86_64)
      set(OPENCV_ICV_NAME "ippicv_2021.10.1_win_intel64_20231206_general.zip")
      set(OPENCV_ICV_HASH "2d5f137d4dd8a5205cc1edb5616fb3da")
    else()
      set(OPENCV_ICV_NAME "ippicv_2021.10.1_win_ia32_20231206_general.zip")
      set(OPENCV_ICV_HASH "63c41a943e93ca87541b71ab67f207b5")
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
