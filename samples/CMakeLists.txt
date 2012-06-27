# ----------------------------------------------------------------------------
#  CMake file for samples. See root CMakeLists.txt 
#
# ----------------------------------------------------------------------------

if(NOT ANDROID)
  add_subdirectory(c)
  add_subdirectory(cpp)
  add_subdirectory(gpu)
endif()

if(ANDROID AND BUILD_ANDROID_EXAMPLES)
  add_subdirectory(android)
endif()
