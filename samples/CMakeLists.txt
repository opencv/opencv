# Detect if we want to build samples with library binaries or not
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_LIST_DIR)
#
# BUILD CASE 1: Build samples with library sources
#


# ----------------------------------------------------------------------------
#  CMake file for samples. See root CMakeLists.txt
#
# ----------------------------------------------------------------------------

add_subdirectory(cpp)
add_subdirectory(gpu)
add_subdirectory(tapi)

if(WIN32 AND HAVE_DIRECTX)
  add_subdirectory(directx)
endif()

if(ANDROID AND BUILD_ANDROID_EXAMPLES)
  add_subdirectory(android)
endif()

#
# END OF BUILD CASE 1: Build samples with library sources
#
else()
#
# BUILD CASE 2: Build samples with library binaries
#
cmake_minimum_required(VERSION 2.8)

project(samples C CXX)
option(BUILD_EXAMPLES "Build samples" ON)

find_package(OpenCV REQUIRED)

if(MSVC)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)

  if(NOT OpenCV_SHARED)
    foreach(flag_var
            CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE
            CMAKE_C_FLAGS_MINSIZEREL CMAKE_C_FLAGS_RELWITHDEBINFO
            CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
            CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif()
      if(${flag_var} MATCHES "/MDd")
        string(REGEX REPLACE "/MDd" "/MTd" ${flag_var} "${${flag_var}}")
      endif()
    endforeach(flag_var)

    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:atlthunk.lib /NODEFAULTLIB:msvcrt.lib /NODEFAULTLIB:msvcrtd.lib")
    set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} /NODEFAULTLIB:libcmt.lib")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /NODEFAULTLIB:libcmtd.lib")
  endif()
endif()

add_subdirectory(cpp)
# FIXIT: can't use cvconfig.h in samples: add_subdirectory(gpu)

if(WIN32)
  add_subdirectory(directx)
endif()

#
# END OF BUILD CASE 2: Build samples with library binaries
#
endif()
