if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_LIST_DIR)
#===================================================================================================
#
# Build as part of OpenCV
#
#===================================================================================================

include("${CMAKE_CURRENT_LIST_DIR}/samples_utils.cmake")

function(ocv_install_example_src relpath)
  if(INSTALL_C_EXAMPLES)
    file(GLOB files ${ARGN})
    install(FILES ${files}
            DESTINATION "${OPENCV_SAMPLES_SRC_INSTALL_PATH}/${relpath}"
            COMPONENT samples)
  endif()
endfunction()

if((TARGET Threads::Threads OR HAVE_PTHREAD OR MSVC OR APPLE) AND NOT OPENCV_EXAMPLES_DISABLE_THREADS)
  add_definitions(-DHAVE_THREADS=1)
endif()

add_subdirectory(cpp)
add_subdirectory(java/tutorial_code)
add_subdirectory(dnn)
add_subdirectory(gpu)
add_subdirectory(tapi)
add_subdirectory(opencl)
if(WIN32 AND HAVE_DIRECTX)
  add_subdirectory(directx)
endif()
if((NOT ANDROID) AND HAVE_OPENGL)
  add_subdirectory(opengl)
endif()
if(HAVE_OPENVX)
  add_subdirectory(openvx)
endif()
if(UNIX AND NOT ANDROID AND (HAVE_VA OR HAVE_VA_INTEL))
  add_subdirectory(va_intel)
endif()
if(ANDROID AND (BUILD_ANDROID_EXAMPLES OR INSTALL_ANDROID_EXAMPLES))
  add_subdirectory(android)
endif()
if(INSTALL_PYTHON_EXAMPLES)
  add_subdirectory(python)
endif()

ocv_install_example_src("." CMakeLists.txt)
if(INSTALL_C_EXAMPLES)
  install(DIRECTORY data DESTINATION "${OPENCV_SAMPLES_SRC_INSTALL_PATH}" COMPONENT samples_data)
endif()

else()
#===================================================================================================
#
#  Standalone mode
#
#===================================================================================================
cmake_minimum_required(VERSION 3.1)

# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

project(samples C CXX)
option(BUILD_EXAMPLES "Build samples" ON)

# Assuming following installation folder structure (default for UNIX):
# <install_root>/share/
# └── OpenCV/  <-- OPENCV_CONFIG_INSTALL_PATH
#     ├── OpenCVConfig.cmake  <-- file to be found by find_package
#     ├── ...
#     ├── samples/  <-- OPENCV_SAMPLES_SRC_INSTALL_PATH
#     │   ├── CMakeLists.txt  <-- this file
#     │   ├── cpp/
find_package(OpenCV REQUIRED PATHS "..")

include("${CMAKE_CURRENT_LIST_DIR}/samples_utils.cmake")

function(ocv_install_example_src)
  # not used in this branch
endfunction()

if(MSVC)
  if(NOT ENABLE_BUILD_HARDENING)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  endif()

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

add_definitions(-DDISABLE_OPENCV_24_COMPATIBILITY=1)  # Avoid C-like legacy API

if(OPENCV_EXAMPLES_DISABLE_THREADS)
  # nothing
elseif(MSVC OR APPLE)
  set(HAVE_THREADS 1)
else()
  find_package(Threads)
endif()
if((TARGET Threads::Threads OR HAVE_THREADS) AND NOT OPENCV_EXAMPLES_DISABLE_THREADS)
  add_definitions(-DHAVE_THREADS=1)
endif()

add_subdirectory(cpp)
if(WIN32)
  add_subdirectory(directx)
endif()
add_subdirectory(dnn)
# add_subdirectory(gpu)
add_subdirectory(opencl)
# add_subdirectory(opengl)
# add_subdirectory(openvx)
add_subdirectory(tapi)
# add_subdirectory(va_intel)

endif()
