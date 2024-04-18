project(kleidicv_hal)

set(KLEIDICV_SOURCE_PATH "" CACHE PATH "Directory containing KleidiCV sources")

if (KLEIDICV_SOURCE_PATH)
  set(THE_ROOT "${KLEIDICV_SOURCE_PATH}")
else()
  ocv_download(FILENAME "kleidicv-83a66640a7a2cb2ab112c676fb80f655f44123c5.tar.gz"
                HASH "13a5afafe2138c42eb50cff2ba3300d1"
                URL
                  "${OPENCV_KLEIDICV_URL}"
                  "$ENV{OPENCV_KLEIDICV_URL}"
                  "https://gitlab.arm.com/kleidi/kleidicv/-/archive/83a66640a7a2cb2ab112c676fb80f655f44123c5/"
                DESTINATION_DIR "${OpenCV_BINARY_DIR}/3rdparty/kleidicv/"
                ID KLEIDICV
                STATUS res
                UNPACK RELATIVE_URL)
  set(THE_ROOT "${OpenCV_BINARY_DIR}/3rdparty/kleidicv/kleidicv-83a66640a7a2cb2ab112c676fb80f655f44123c5")
endif()

include("${THE_ROOT}/adapters/opencv/CMakeLists.txt")
