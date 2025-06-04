function(download_fastcv root_dir)

  # Commit SHA in the opencv_3rdparty repo
  set(FASTCV_COMMIT "8b2f8b77a3557b413e3b25cff4fae6f12b48094b")

  # Define actual FastCV versions
  if(ANDROID)
    if(AARCH64)
      message(STATUS "Download FastCV for Android aarch64")
      set(FCV_PACKAGE_NAME  "fastcv_android_aarch64_2025_04_29.tgz")
      set(FCV_PACKAGE_HASH  "d9172a9a3e5d92d080a4192cc5691001")
    else()
      message(STATUS "Download FastCV for Android armv7")
      set(FCV_PACKAGE_NAME  "fastcv_android_arm32_2025_04_29.tgz")
      set(FCV_PACKAGE_HASH  "246b5253233391cd2c74d01d49aee9c3")
    endif()
  elseif(UNIX AND NOT APPLE AND NOT IOS AND NOT XROS)
    if(AARCH64)
      set(FCV_PACKAGE_NAME  "fastcv_linux_aarch64_2025_05_29.tgz")
      set(FCV_PACKAGE_HASH  "decd490524f786e103125b8b948151f3")
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
