function(download_ippicv root_var)
  set(${root_var} "" PARENT_SCOPE)

  # Commit SHA in the opencv_3rdparty repo
  set(IPPICV_COMMIT "dfe3162c237af211e98b8960018b564bc209261d")
  # Define actual ICV versions
  if(APPLE)
    set(OPENCV_ICV_PLATFORM "macosx")
    set(OPENCV_ICV_PACKAGE_SUBDIR "ippicv_mac")
    if(X86_64)
      set(OPENCV_ICV_NAME "ippicv_2017u3_mac_intel64_general_20170822.tgz")
      set(OPENCV_ICV_HASH "c1ebb5dfa5b7f54b0c44e1917805a463")
    else()
      set(OPENCV_ICV_NAME "ippicv_2017u3_mac_ia32_general_20170822.tgz")
      set(OPENCV_ICV_HASH "49b05a669042753ae75895a445ebd612")
    endif()
  elseif((UNIX AND NOT ANDROID) OR (UNIX AND ANDROID_ABI MATCHES "x86"))
    set(OPENCV_ICV_PLATFORM "linux")
    set(OPENCV_ICV_PACKAGE_SUBDIR "ippicv_lnx")
    if(X86_64)
      set(OPENCV_ICV_NAME "ippicv_2017u3_lnx_intel64_general_20170822.tgz")
      set(OPENCV_ICV_HASH "4e0352ce96473837b1d671ce87f17359")
    else()
      set(OPENCV_ICV_NAME "ippicv_2017u3_lnx_ia32_general_20170822.tgz")
      set(OPENCV_ICV_HASH "dcdb0ba4b123f240596db1840cd59a76")
    endif()
  elseif(WIN32 AND NOT ARM)
    set(OPENCV_ICV_PLATFORM "windows")
    set(OPENCV_ICV_PACKAGE_SUBDIR "ippicv_win")
    if(X86_64)
      set(OPENCV_ICV_NAME "ippicv_2017u3_win_intel64_general_20170822.zip")
      set(OPENCV_ICV_HASH "0421e642bc7ad741a2236d3ec4190bdd")
    else()
      set(OPENCV_ICV_NAME "ippicv_2017u3_win_ia32_general_20170822.zip")
      set(OPENCV_ICV_HASH "8a7680ae352c192de2e2e34936164bd0")
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
