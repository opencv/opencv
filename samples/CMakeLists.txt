# ----------------------------------------------------------------------------
#  CMake file for samples. See root CMakeLists.txt
#
# ----------------------------------------------------------------------------

add_subdirectory(c)
add_subdirectory(cpp)
add_subdirectory(gpu)
add_subdirectory(ocl)

if(ANDROID AND BUILD_ANDROID_EXAMPLES)
  add_subdirectory(android)
endif()
