if(NOT HAVE_CUDA)
  ocv_module_disable(cudev)
endif()

set(the_description "CUDA device layer")

ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4189 /wd4505 -Wundef -Wmissing-declarations -Wunused-function -Wunused-variable)

ocv_add_module(cudev)

file(GLOB_RECURSE lib_hdrs "include/opencv2/*.hpp")

add_custom_target(${the_module} SOURCES ${lib_hdrs})

if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(${the_module} PROPERTIES FOLDER "modules")
endif()

foreach(var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_DEBUG)
  # we remove /EHa as it generates warnings under windows
  string(REPLACE "/EHa" "" ${var} "${${var}}")

  # we remove -ggdb3 flag as it leads to preprocessor errors when compiling CUDA files (CUDA 4.1)
  string(REPLACE "-ggdb3" "" ${var} "${${var}}")

  # we remove -Wsign-promo as it generates warnings under linux
  string(REPLACE "-Wsign-promo" "" ${var} "${${var}}")

  # we remove -fvisibility-inlines-hidden because it's used for C++ compiler
  # but NVCC uses C compiler by default
  string(REPLACE "-fvisibility-inlines-hidden" "" ${var} "${${var}}")
endforeach()

if(BUILD_TESTS)
  add_subdirectory(test)
endif()
