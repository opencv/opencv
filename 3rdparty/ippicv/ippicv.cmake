function(download_ippicv root_var)
  set(${root_var} "" PARENT_SCOPE)

  # Commit SHA in the opencv_3rdparty repo
  set(IPPICV_COMMIT "bdb7bb85f34a8cb0d35e40a81f58da431aa1557a")
  # Define actual ICV versions
  if(APPLE)
    set(OPENCV_ICV_PLATFORM "macosx")
    set(OPENCV_ICV_PACKAGE_SUBDIR "ippicv_mac")
    if(X86_64)
      set(OPENCV_ICV_NAME "ippicv_2017u3_mac_intel64_general_20180518.tgz")
      set(OPENCV_ICV_HASH "3ae52b9be0fe73dd45bc5e9429cd3732")
    else()
      set(OPENCV_ICV_NAME "ippicv_2017u3_mac_ia32_general_20180518.tgz")
      set(OPENCV_ICV_HASH "698660b975b62bee3ef6c5af51e97544")
    endif()
  elseif((UNIX AND NOT ANDROID) OR (UNIX AND ANDROID_ABI MATCHES "x86"))
    set(OPENCV_ICV_PLATFORM "linux")
    set(OPENCV_ICV_PACKAGE_SUBDIR "ippicv_lnx")
    if(X86_64)
      set(OPENCV_ICV_NAME "ippicv_2017u3_lnx_intel64_general_20180518.tgz")
      set(OPENCV_ICV_HASH "b7cc351267db2d34b9efa1cd22ff0572")
    else()
      set(OPENCV_ICV_NAME "ippicv_2017u3_lnx_ia32_general_20180518.tgz")
      set(OPENCV_ICV_HASH "ea72de74dae3c604eb6348395366e78e")
    endif()
  elseif(WIN32 AND NOT ARM)
    set(OPENCV_ICV_PLATFORM "windows")
    set(OPENCV_ICV_PACKAGE_SUBDIR "ippicv_win")
    if(X86_64)
      set(OPENCV_ICV_NAME "ippicv_2017u3_win_intel64_general_20180518.zip")
      set(OPENCV_ICV_HASH "915ff92958089ede8ea532d3c4fe7187")
    else()
      set(OPENCV_ICV_NAME "ippicv_2017u3_win_ia32_general_20180518.zip")
      set(OPENCV_ICV_HASH "928168c2d99ab284047dfcfb7a821d91")
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
