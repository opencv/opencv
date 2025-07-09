function(download_fastcv root_dir)

  # Commit SHA in the opencv_3rdparty repo
  set(FASTCV_COMMIT "7f1830d57a77f4705eb76aefb339428f52bed95a")

  # Define actual FastCV versions
  if(ANDROID)
    if(AARCH64)
      message(STATUS "Download FastCV for Android aarch64")
      set(FCV_PACKAGE_NAME  "fastcv_android_aarch64_2025_07_09.tgz")
      set(FCV_PACKAGE_HASH  "8b9497858cf3c3502a0be4369d06ebf8")
    else()
      message(STATUS "Download FastCV for Android armv7")
      set(FCV_PACKAGE_NAME  "fastcv_android_arm32_2025_07_09.tgz")
      set(FCV_PACKAGE_HASH  "e0e6009c9f2f2b96140cd6a639c7383f")
    endif()
  elseif(UNIX AND NOT APPLE AND NOT IOS AND NOT XROS)
    if(AARCH64)
      set(FCV_PACKAGE_NAME  "fastcv_linux_aarch64_2025_07_09.tgz")
      set(FCV_PACKAGE_HASH  "05e254e0eb3c13fa23eb7213f0fe6d82")
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
