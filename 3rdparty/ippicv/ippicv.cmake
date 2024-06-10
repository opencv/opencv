function(download_ippicv root_var)
  set(${root_var} "" PARENT_SCOPE)

  # Commit SHA in the opencv_3rdparty repo
  set(IPPICV_COMMIT "fd27188235d85e552de31425e7ea0f53ba73ba53")
  # Define actual ICV versions
  if(APPLE)
    set(IPPICV_COMMIT "0cc4aa06bf2bef4b05d237c69a5a96b9cd0cb85a")
    set(OPENCV_ICV_PLATFORM "macosx")
    set(OPENCV_ICV_PACKAGE_SUBDIR "ippicv_mac")
    set(OPENCV_ICV_NAME "ippicv_2021.9.1_mac_intel64_20230919_general.tgz")
    set(OPENCV_ICV_HASH "14f01c5a4780bfae9dde9b0aaf5e56fc")
  elseif((UNIX AND NOT ANDROID) OR (UNIX AND ANDROID_ABI MATCHES "x86"))
    set(OPENCV_ICV_PLATFORM "linux")
    set(OPENCV_ICV_PACKAGE_SUBDIR "ippicv_lnx")
    if(X86_64)
      set(OPENCV_ICV_NAME "ippicv_2021.11.0_lnx_intel64_20240201_general.tgz")
      set(OPENCV_ICV_HASH "0f2745ff705ecae31176dad437608f6f")
    else()
      set(OPENCV_ICV_NAME "ippicv_2021.11.0_lnx_ia32_20240201_general.tgz")
      set(OPENCV_ICV_HASH "63e381bf08076ca34fd5264203043a45")
    endif()
  elseif(WIN32 AND NOT ARM)
    set(OPENCV_ICV_PLATFORM "windows")
    set(OPENCV_ICV_PACKAGE_SUBDIR "ippicv_win")
    if(X86_64)
      set(OPENCV_ICV_NAME "ippicv_2021.11.0_win_intel64_20240201_general.zip")
      set(OPENCV_ICV_HASH "59d154bf54a1e3eea20d7248f81a2a8e")
    else()
      set(OPENCV_ICV_NAME "ippicv_2021.11.0_win_ia32_20240201_general.zip")
      set(OPENCV_ICV_HASH "7a6d8ac5825c02fea6cbfc1201b521b5")
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
