function(download_ippicv root_var)
  set(${root_var} "" PARENT_SCOPE)

  # Commit SHA in the opencv_3rdparty repo
  set(IPPICV_COMMIT "1224f78da6684df04397ac0f40c961ed37f79ccb")
  # Define actual ICV versions
  if(APPLE)
    set(OPENCV_ICV_PLATFORM "macosx")
    set(OPENCV_ICV_PACKAGE_SUBDIR "ippicv_mac")
    set(OPENCV_ICV_NAME "ippicv_2021.8_mac_intel64_20230330_general.tgz")
    set(OPENCV_ICV_HASH "d2b234a86af1b616958619a4560356d9")
  elseif((UNIX AND NOT ANDROID) OR (UNIX AND ANDROID_ABI MATCHES "x86"))
    set(OPENCV_ICV_PLATFORM "linux")
    set(OPENCV_ICV_PACKAGE_SUBDIR "ippicv_lnx")
    if(X86_64)
      set(OPENCV_ICV_NAME "ippicv_2021.8_lnx_intel64_20230330_general.tgz")
      set(OPENCV_ICV_HASH "43219bdc7e3805adcbe3a1e2f1f3ef3b")
    else()
      set(OPENCV_ICV_NAME "ippicv_2021.8_lnx_ia32_20230330_general.tgz")
      set(OPENCV_ICV_HASH "165875443d72faa3fd2146869da90d07")
    endif()
  elseif(WIN32 AND NOT ARM)
    set(OPENCV_ICV_PLATFORM "windows")
    set(OPENCV_ICV_PACKAGE_SUBDIR "ippicv_win")
    if(X86_64)
      set(OPENCV_ICV_NAME "ippicv_2021.8_win_intel64_20230330_general.zip")
      set(OPENCV_ICV_HASH "71e4f58de939f0348ec7fb58ffb17dbf")
    else()
      set(OPENCV_ICV_NAME "ippicv_2021.8_win_ia32_20230330_general.zip")
      set(OPENCV_ICV_HASH "57fd4648cfe64eae9e2ad9d50173a553")
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
