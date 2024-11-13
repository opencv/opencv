function(download_kleidicv root_var)
  set(${root_var} "" PARENT_SCOPE)

  ocv_update(KLEIDICV_SRC_COMMIT "0.2.0")
  ocv_update(KLEIDICV_SRC_HASH "dabe522e8f55ac342d07a787391dab80")

  set(THE_ROOT "${OpenCV_BINARY_DIR}/3rdparty/kleidicv")
  ocv_download(FILENAME "kleidicv-${KLEIDICV_SRC_COMMIT}.tar.gz"
                HASH ${KLEIDICV_SRC_HASH}
                URL
                  "${OPENCV_KLEIDICV_URL}"
                  "$ENV{OPENCV_KLEIDICV_URL}"
                  "https://gitlab.arm.com/kleidi/kleidicv/-/archive/${KLEIDICV_SRC_COMMIT}/"
                DESTINATION_DIR ${THE_ROOT}
                ID KLEIDICV
                STATUS res
                UNPACK RELATIVE_URL)
  if(res)
    set(${root_var} "${OpenCV_BINARY_DIR}/3rdparty/kleidicv/kleidicv-${KLEIDICV_SRC_COMMIT}" PARENT_SCOPE)
  endif()
endfunction()
