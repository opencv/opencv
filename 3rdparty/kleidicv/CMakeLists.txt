project(kleidicv_hal)

set(KLEIDICV_SOURCE_PATH "" CACHE PATH "Directory containing KleidiCV sources")
ocv_update(KLEIDICV_SRC_COMMIT "a9971ba8bf1d8b008a32c6ed8fea05cd8eb16748")
ocv_update(KLEIDICV_SRC_HASH "53d1d80620395a12c7fee91460499368")

if(KLEIDICV_SOURCE_PATH)
  set(THE_ROOT "${KLEIDICV_SOURCE_PATH}")
else()
  ocv_download(FILENAME "kleidicv-${KLEIDICV_SRC_COMMIT}.tar.gz"
                HASH ${KLEIDICV_SRC_HASH}
                URL
                  "${OPENCV_KLEIDICV_URL}"
                  "$ENV{OPENCV_KLEIDICV_URL}"
                  "https://gitlab.arm.com/kleidi/kleidicv/-/archive/${KLEIDICV_SRC_COMMIT}/"
                DESTINATION_DIR "${OpenCV_BINARY_DIR}/3rdparty/kleidicv/"
                ID KLEIDICV
                STATUS res
                UNPACK RELATIVE_URL)
  set(THE_ROOT "${OpenCV_BINARY_DIR}/3rdparty/kleidicv/kleidicv-${KLEIDICV_SRC_COMMIT}")
endif()

include("${THE_ROOT}/adapters/opencv/CMakeLists.txt")
