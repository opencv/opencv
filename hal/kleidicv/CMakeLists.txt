project(kleidicv_hal)

if(HAVE_KLEIDICV)
  option(KLEIDICV_ENABLE_SME2 "" OFF) # not compatible with some CLang versions in NDK
  option(KLEIDICV_USE_CV_NAMESPACE_IN_OPENCV_HAL "" OFF)

  include("${KLEIDICV_SOURCE_PATH}/adapters/opencv/CMakeLists.txt")

  # Inherit KleidiCV compile options, but filter invalid flags on Apple Clang
  if(CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND
     CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")

    get_target_property(_kleidicv_opts kleidicv COMPILE_OPTIONS)
    if(_kleidicv_opts)
      list(FILTER _kleidicv_opts EXCLUDE REGEX "^-march=armv8-a$")
    endif()

    target_compile_options(kleidicv_hal PRIVATE
      ${_kleidicv_opts}
      "-Wno-old-style-cast" "-Wno-unused-function"
    )

  else()

    target_compile_options(kleidicv_hal PRIVATE
      $<TARGET_PROPERTY:kleidicv,COMPILE_OPTIONS>
      "-Wno-old-style-cast" "-Wno-unused-function"
    )

  endif()
endif()
