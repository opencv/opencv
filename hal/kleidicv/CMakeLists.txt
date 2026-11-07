project(kleidicv_hal)

if(HAVE_KLEIDICV)
  option(KLEIDICV_ENABLE_SME2 "" OFF) # not compatible with some CLang versions in NDK
  option(KLEIDICV_USE_CV_NAMESPACE_IN_OPENCV_HAL "" OFF)
  include("${KLEIDICV_SOURCE_PATH}/adapters/opencv/CMakeLists.txt")
  # HACK to suppress adapters/opencv/kleidicv_hal.cpp:343:12: warning: unused function 'from_opencv' [-Wunused-function]
  target_compile_options( kleidicv_hal PRIVATE
      $<TARGET_PROPERTY:kleidicv,COMPILE_OPTIONS>
      "-Wno-old-style-cast" "-Wno-unused-function"
  )
endif()
