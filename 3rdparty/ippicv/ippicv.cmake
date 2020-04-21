function(download_ippicv root_var)
  set(${root_var} "" PARENT_SCOPE)

  # Commit SHA in the opencv_3rdparty repo
  set(IPPICV_COMMIT "a56b6ac6f030c312b2dce17430eef13aed9af274")
  # Define actual ICV versions
  if(APPLE)
    set(OPENCV_ICV_PLATFORM "macosx")
    set(OPENCV_ICV_PACKAGE_SUBDIR "ippicv_mac")
    set(OPENCV_ICV_NAME "ippicv_2020_mac_intel64_20191018_general.tgz")
    set(OPENCV_ICV_HASH "1c3d675c2a2395d094d523024896e01b")
  elseif((UNIX AND NOT ANDROID) OR (UNIX AND ANDROID_ABI MATCHES "x86"))
    set(OPENCV_ICV_PLATFORM "linux")
    set(OPENCV_ICV_PACKAGE_SUBDIR "ippicv_lnx")
    if(X86_64)
      set(OPENCV_ICV_NAME "ippicv_2020_lnx_intel64_20191018_general.tgz")
      set(OPENCV_ICV_HASH "7421de0095c7a39162ae13a6098782f9")
    else()
      set(OPENCV_ICV_NAME "ippicv_2020_lnx_ia32_20191018_general.tgz")
      set(OPENCV_ICV_HASH "ad189a940fb60eb71f291321322fe3e8")
    endif()
  elseif(WIN32 AND NOT ARM)
    set(OPENCV_ICV_PLATFORM "windows")
    set(OPENCV_ICV_PACKAGE_SUBDIR "ippicv_win")
    if(X86_64)
      set(OPENCV_ICV_NAME "ippicv_2020_win_intel64_20191018_general.zip")
      set(OPENCV_ICV_HASH "879741a7946b814455eee6c6ffde2984")
    else()
      set(OPENCV_ICV_NAME "ippicv_2020_win_ia32_20191018_general.zip")
      set(OPENCV_ICV_HASH "cd39bdf0c2e1cac9a61101dad7a2413e")
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
