project(kleidicv_hal)

set(KLEIDICV_SOURCE_PATH "" CACHE PATH "Directory containing KleidiCV sources")
ocv_update(KLEIDICV_SRC_COMMIT "0.2.0")
ocv_update(KLEIDICV_SRC_HASH "dabe522e8f55ac342d07a787391dab80")

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

option(KLEIDICV_ENABLE_SME2 "" OFF) # not compatible with some CLang versions in NDK
include("${THE_ROOT}/adapters/opencv/CMakeLists.txt")
