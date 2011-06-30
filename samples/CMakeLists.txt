# ----------------------------------------------------------------------------
#  CMake file for samples. See root CMakeLists.txt 
#
# ----------------------------------------------------------------------------

add_subdirectory(c)
add_subdirectory(cpp)

if(NOT ANDROID)
    add_subdirectory(gpu)
endif()

if(ANDROID AND CAN_BUILD_ANDROID_PROJECTS)
   add_subdirectory(android)
endif()

if(0)
add_subdirectory(swig_python)
add_subdirectory(octave)
endif()
