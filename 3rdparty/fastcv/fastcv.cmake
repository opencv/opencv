function(download_fastcv root_dir)

  # Commit SHA in the opencv_3rdparty repo
  set(FASTCV_COMMIT "8d86e68dad8b80b8575a8d3cf401d3ee96c24148")

  # Define actual FastCV versions
  if(ANDROID)
    if(AARCH64)
      message(STATUS "Download FastCV for Android aarch64")
      set(FCV_PACKAGE_NAME  "fastcv_android_aarch64_2025_04_08.tgz")
      set(FCV_PACKAGE_HASH  "e028966a1d1b2f3f0bc5967d316e8b64")
    else()
      message(STATUS "Download FastCV for Android armv7")
      set(FCV_PACKAGE_NAME  "fastcv_android_arm32_2025_04_08.tgz")
      set(FCV_PACKAGE_HASH  "6fc1e812a4b3ef392469d2283e037ffe")
    endif()
  elseif(UNIX AND NOT APPLE AND NOT IOS AND NOT XROS)
    if(AARCH64)
      set(FCV_PACKAGE_NAME  "fastcv_linux_aarch64_2025_04_08.tgz")
      set(FCV_PACKAGE_HASH  "062a26639cd2788beee2e0dd8743d680")
    else()
      message("FastCV: fastcv lib for 32-bit Linux is not supported for now!")
    endif()
  endif(ANDROID)

  # Download Package

  set(OPENCV_FASTCV_URL "https://raw.githubusercontent.com/opencv/opencv_3rdparty/${FASTCV_COMMIT}/fastcv/")

  ocv_download( FILENAME        ${FCV_PACKAGE_NAME}
                HASH            ${FCV_PACKAGE_HASH}
                URL             ${OPENCV_FASTCV_URL}
                DESTINATION_DIR ${root_dir}
                ID              FASTCV
                STATUS          res
                UNPACK
                RELATIVE_URL)
  if(res)
    set(HAVE_FASTCV TRUE CACHE BOOL "FastCV status")
  else()
    message(WARNING "FastCV: package download failed!")
  endif()

endfunction()
