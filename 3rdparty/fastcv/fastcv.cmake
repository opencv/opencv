function(download_fastcv root_dir)

  # Commit SHA in the opencv_3rdparty repo
  set(FASTCV_COMMIT "f4413cc2ab7233fdfc383a4cded402c072677fb0")

  # Define actual FastCV versions
  if(ANDROID)
    if(AARCH64)
      message(STATUS "Download FastCV for Android aarch64")
      set(FCV_PACKAGE_NAME  "fastcv_android_aarch64_2024_12_11.tgz")
      set(FCV_PACKAGE_HASH  "9dac41e86597305f846212dae31a4a88")
    else()
      message(STATUS "Download FastCV for Android armv7")
      set(FCV_PACKAGE_NAME  "fastcv_android_arm32_2024_12_11.tgz")
      set(FCV_PACKAGE_HASH  "fe2d30334180b17e3031eee92aac43b6")
    endif()
  elseif(UNIX AND NOT APPLE AND NOT IOS AND NOT XROS)
    if(AARCH64)
      set(FCV_PACKAGE_NAME  "fastcv_linux_aarch64_2025_02_12.tgz")
      set(FCV_PACKAGE_HASH  "33ac2a59cf3e7d6402eee2e010de1202")
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
