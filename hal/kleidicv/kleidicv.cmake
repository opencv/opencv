function(download_kleidicv root_var)
  set(${root_var} "" PARENT_SCOPE)

  ocv_update(KLEIDICV_SRC_COMMIT "26.09")
  ocv_update(KLEIDICV_SRC_HASH "77bb081492b2aa68033d7a848987252b")

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
