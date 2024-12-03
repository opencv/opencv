project(kleidicv_hal)

if(HAVE_KLEIDICV)
  option(KLEIDICV_ENABLE_SME2 "" OFF) # not compatible with some CLang versions in NDK
  include("${KLEIDICV_SOURCE_PATH}/adapters/opencv/CMakeLists.txt")
endif()

