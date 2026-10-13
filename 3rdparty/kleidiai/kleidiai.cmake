# Downloads Arm KleidiAI (Apache-2.0), same version as ONNX Runtime 1.26-1.30.
# The kernels OpenCV needs are built in 3rdparty/mlas/CMakeLists.txt.
function(download_kleidiai root_var)
  set(${root_var} "" PARENT_SCOPE)

  ocv_update(KLEIDIAI_SRC_VERSION "1.20.0")
  ocv_update(KLEIDIAI_SRC_HASH "c9cb67e02449a3d22b2a061825930a62")

  set(THE_ROOT "${OpenCV_BINARY_DIR}/3rdparty/kleidiai")
  ocv_download(FILENAME "kleidiai-v${KLEIDIAI_SRC_VERSION}.tar.gz"
                HASH ${KLEIDIAI_SRC_HASH}
                URL
                  "${OPENCV_KLEIDIAI_URL}"
                  "$ENV{OPENCV_KLEIDIAI_URL}"
                  "https://github.com/ARM-software/kleidiai/archive/refs/tags/v${KLEIDIAI_SRC_VERSION}.tar.gz"
                DESTINATION_DIR ${THE_ROOT}
                ID KLEIDIAI
                STATUS res
                UNPACK)
  if(res)
    set(${root_var} "${THE_ROOT}/kleidiai-${KLEIDIAI_SRC_VERSION}" PARENT_SCOPE)
  endif()
endfunction()
