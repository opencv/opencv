if(BUILD_SHARED_LIBS)
add_definitions(-DGTEST_CREATE_SHARED_LIBRARY=1)
endif()
define_opencv_module(ts opencv_core)
