#
# The script to detect Qualcomm FastCV(FCV) installation/package
#
# On return this will define:
#
# FCV_ROOT_DIR      - The root of FastCV installation
# FCV_HEADER_DIR    - The root of FastCV header files
# FCV_LIB_DIR       - The root of FastCV library files
#

if(FCV_ENABLE)
  set(FCV_ROOT_DIR    "${OpenCV_BINARY_DIR}/3rdparty/fastcv")
  set(FCV_HEADER_DIR  "${FCV_ROOT_DIR}/inc")
  set(FCV_LIB_DIR     "${FCV_ROOT_DIR}/libs")

  if((NOT EXISTS ${FCV_HEADER_DIR}) OR (NOT EXISTS ${FCV_LIB_DIR}))
    include("${OpenCV_SOURCE_DIR}/3rdparty/fastcv/fastcv.cmake")
    download_fastcv(${FCV_ROOT_DIR})
  endif()

endif(FCV_ENABLE)