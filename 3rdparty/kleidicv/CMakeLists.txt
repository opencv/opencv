project(kleidicv_hal)

set(KLEIDICV_SOURCE_PATH "" CACHE PATH "Directory containing KleidiCV sources")
ocv_update(KLEIDICV_SRC_HASH "53d1d80620395a12c7fee91460499368")

if(KLEIDICV_SOURCE_PATH)
  set(THE_ROOT "${KLEIDICV_SOURCE_PATH}")
else()
  ocv_download(FILENAME "kleidicv-a9971ba8bf1d8b008a32c6ed8fea05cd8eb16748.tar.gz"
                HASH ${KLEIDICV_SRC_HASH}
                URL
                  "${OPENCV_KLEIDICV_URL}"
                  "$ENV{OPENCV_KLEIDICV_URL}"
                  "https://gitlab.arm.com/kleidi/kleidicv/-/archive/a9971ba8bf1d8b008a32c6ed8fea05cd8eb16748/"
                DESTINATION_DIR "${OpenCV_BINARY_DIR}/3rdparty/kleidicv/"
                ID KLEIDICV
                STATUS res
                UNPACK RELATIVE_URL)
  set(THE_ROOT "${OpenCV_BINARY_DIR}/3rdparty/kleidicv/kleidicv-a9971ba8bf1d8b008a32c6ed8fea05cd8eb16748")
endif()

include("${THE_ROOT}/adapters/opencv/CMakeLists.txt")
