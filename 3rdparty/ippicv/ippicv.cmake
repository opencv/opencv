function(download_ippicv root_var)
  set(${root_var} "" PARENT_SCOPE)

  # Commit SHA in the opencv_3rdparty repo
  set(IPPICV_COMMIT "32e315a5b106a7b89dbed51c28f8120a48b368b4")
  # Define actual ICV versions
  if(APPLE)
    set(OPENCV_ICV_PLATFORM "macosx")
    set(OPENCV_ICV_PACKAGE_SUBDIR "ippicv_mac")
    if(X86_64)
      set(OPENCV_ICV_NAME "ippicv_2019_mac_intel64_general_20180723.tgz")
      set(OPENCV_ICV_HASH "fe6b2bb75ae0e3f19ad3ae1a31dfa4a2")
    else()
      set(OPENCV_ICV_NAME "ippicv_2019_mac_ia32_general_20180723.tgz")
      set(OPENCV_ICV_HASH "b5dfa78c87eb75c64470cbe5ec876f4f")
    endif()
  elseif((UNIX AND NOT ANDROID) OR (UNIX AND ANDROID_ABI MATCHES "x86"))
    set(OPENCV_ICV_PLATFORM "linux")
    set(OPENCV_ICV_PACKAGE_SUBDIR "ippicv_lnx")
    if(X86_64)
      set(OPENCV_ICV_NAME "ippicv_2019_lnx_intel64_general_20180723.tgz")
      set(OPENCV_ICV_HASH "c0bd78adb4156bbf552c1dfe90599607")
    else()
      set(OPENCV_ICV_NAME "ippicv_2019_lnx_ia32_general_20180723.tgz")
      set(OPENCV_ICV_HASH "4f38432c30bfd6423164b7a24bbc98a0")
    endif()
  elseif(WIN32 AND NOT ARM)
    set(OPENCV_ICV_PLATFORM "windows")
    set(OPENCV_ICV_PACKAGE_SUBDIR "ippicv_win")
    if(X86_64)
      set(OPENCV_ICV_NAME "ippicv_2019_win_intel64_20180723_general.zip")
      set(OPENCV_ICV_HASH "1d222685246896fe089f88b8858e4b2f")
    else()
      set(OPENCV_ICV_NAME "ippicv_2019_win_ia32_20180723_general.zip")
      set(OPENCV_ICV_HASH "0157251a2eb9cd63a3ebc7eed0f3e59e")
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
