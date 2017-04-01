if(NOT HAVE_CUDA)
  ocv_module_disable(cudev)
endif()

set(the_description "CUDA device layer")

ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4189 /wd4505 -Wundef -Wmissing-declarations -Wunused-function -Wunused-variable -Wenum-compare -Wshadow)

ocv_add_module(cudev)

ocv_module_include_directories(opencv_core opencv_hal)

file(GLOB_RECURSE lib_hdrs "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/*.hpp")
file(GLOB         lib_srcs "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp")

source_group("Include" FILES ${lib_hdrs})
source_group("Src" FILES ${lib_srcs})

ocv_glob_module_sources(HEADERS ${lib_hdrs} SOURCES ${lib_srcs})

ocv_create_module()

if(BUILD_TESTS AND NOT BUILD_opencv_world)
  add_subdirectory(test)
endif()
