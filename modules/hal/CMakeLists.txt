set(the_description "The Hardware Acceleration Layer (HAL) module")

set(OPENCV_MODULE_TYPE STATIC)

if(UNIX)
  if(CMAKE_COMPILER_IS_GNUCXX OR CV_ICC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
  endif()
endif()

ocv_define_module(hal)
