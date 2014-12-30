if(NOT HAVE_CUDA)
  ocv_module_disable(cudev)
endif()

set(the_description "CUDA device layer")

ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4189 /wd4505 -Wundef -Wmissing-declarations -Wunused-function -Wunused-variable -Wenum-compare -Wshadow)

ocv_add_module(cudev)

ocv_module_include_directories(opencv_core)

file(GLOB_RECURSE lib_hdrs "include/opencv2/*.hpp")
file(GLOB         lib_srcs "src/*.cpp")

ocv_set_module_sources(HEADERS ${lib_hdrs} SOURCES ${lib_srcs})

ocv_create_module()

if(BUILD_TESTS)
  add_subdirectory(test)
endif()
