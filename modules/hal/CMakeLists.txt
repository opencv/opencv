set(the_description "The Hardware Acceleration Layer (HAL) module")

set(OPENCV_MODULE_TYPE STATIC)

if(OPENCV_HAL_HEADERS AND OPENCV_HAL_LIBS)
    set(OPENCV_HAL_HEADERS_INCLUDES "#include \"${OPENCV_HAL_HEADERS}\"")
    set(DEPS "${OPENCV_HAL_LIBS}")
else()
    set(OPENCV_HAL_HEADERS_INCLUDES "// using default HAL")
    set(DEPS "")
endif()

configure_file("${OpenCV_SOURCE_DIR}/cmake/templates/custom_hal.hpp.in" "${CMAKE_BINARY_DIR}/custom_hal.hpp" @ONLY)

if(UNIX)
  if(CMAKE_COMPILER_IS_GNUCXX OR CV_ICC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
  endif()
endif()

ocv_define_module(hal ${DEPS})
