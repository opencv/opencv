function(download_ippicv root_var)
  set(${root_var} "" PARENT_SCOPE)

  # Commit SHA in the opencv_3rdparty repo
  set(IPPICV_COMMIT "7f55c0c26be418d494615afca15218566775c725")
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
      set(OPENCV_ICV_NAME "ippicv_2021.12.0_lnx_intel64_20240425_general.tgz")
      set(OPENCV_ICV_HASH "d06e6d44ece88f7f17a6cd9216761186")
    else()
      set(OPENCV_ICV_NAME "ippicv_2021.12.0_lnx_ia32_20240425_general.tgz")
      set(OPENCV_ICV_HASH "85ffa2b9ed7802b93c23fa27b0097d36")
    endif()
  elseif(WIN32 AND NOT ARM)
    set(OPENCV_ICV_PLATFORM "windows")
    set(OPENCV_ICV_PACKAGE_SUBDIR "ippicv_win")
    if(X86_64)
      set(OPENCV_ICV_NAME "ippicv_2021.12.0_win_intel64_20240425_general.zip")
      set(OPENCV_ICV_HASH "402ff8c6b4986738fed71c44e1ce665d")
    else()
      set(OPENCV_ICV_NAME "ippicv_2021.12.0_win_ia32_20240425_general.zip")
      set(OPENCV_ICV_HASH "8b1d2a23957d57624d0de8f2a5cae5f1")
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
