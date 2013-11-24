# ===================================================================================
#  The OpenCV CMake configuration file
#
#             ** File generated automatically, do not modify **
#
#  Usage from an external project:
#    In your CMakeLists.txt, add these lines:
#
#    FIND_PACKAGE(OpenCV REQUIRED)
#    TARGET_LINK_LIBRARIES(MY_TARGET_NAME ${OpenCV_LIBS})
#
#    Or you can search for specific OpenCV modules:
#
#    FIND_PACKAGE(OpenCV REQUIRED core highgui)
#
#    If the module is found then OPENCV_<MODULE>_FOUND is set to TRUE.
#
#    This file will define the following variables:
#      - OpenCV_LIBS                     : The list of libraries to links against.
#      - OpenCV_LIB_DIR                  : The directory(es) where lib files are. Calling LINK_DIRECTORIES
#                                          with this path is NOT needed.
#      - OpenCV_INCLUDE_DIRS             : The OpenCV include directories.
#      - OpenCV_COMPUTE_CAPABILITIES     : The version of compute capability
#      - OpenCV_ANDROID_NATIVE_API_LEVEL : Minimum required level of Android API
#      - OpenCV_VERSION                  : The version of this OpenCV build. Example: "2.4.0"
#      - OpenCV_VERSION_MAJOR            : Major version part of OpenCV_VERSION. Example: "2"
#      - OpenCV_VERSION_MINOR            : Minor version part of OpenCV_VERSION. Example: "4"
#      - OpenCV_VERSION_PATCH            : Patch version part of OpenCV_VERSION. Example: "0"
#
#    Advanced variables:
#      - OpenCV_SHARED
#      - OpenCV_CONFIG_PATH
#      - OpenCV_LIB_COMPONENTS
#
# ===================================================================================
#
#    Windows pack specific options:
#      - OpenCV_STATIC
#      - OpenCV_CUDA

if(CMAKE_VERSION VERSION_GREATER 2.6)
  get_property(OpenCV_LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)
  if(NOT ";${OpenCV_LANGUAGES};" MATCHES ";CXX;")
    enable_language(CXX)
  endif()
endif()

if(NOT DEFINED OpenCV_STATIC)
  # look for global setting
  if(NOT DEFINED BUILD_SHARED_LIBS OR BUILD_SHARED_LIBS)
    set(OpenCV_STATIC OFF)
  else()
    set(OpenCV_STATIC ON)
  endif()
endif()

if(NOT DEFINED OpenCV_CUDA)
  # if user' app uses CUDA, then it probably wants CUDA-enabled OpenCV binaries
  if(CUDA_FOUND)
    set(OpenCV_CUDA ON)
  endif()
endif()

if(MSVC)
  if(CMAKE_CL_64)
    set(OpenCV_ARCH x64)
    set(OpenCV_TBB_ARCH intel64)
  else()
    set(OpenCV_ARCH x86)
    set(OpenCV_TBB_ARCH ia32)
  endif()
  if(MSVC_VERSION EQUAL 1400)
    set(OpenCV_RUNTIME vc8)
  elseif(MSVC_VERSION EQUAL 1500)
    set(OpenCV_RUNTIME vc9)
  elseif(MSVC_VERSION EQUAL 1600)
    set(OpenCV_RUNTIME vc10)
  elseif(MSVC_VERSION EQUAL 1700)
    set(OpenCV_RUNTIME vc11)
  endif()
elseif(MINGW)
  set(OpenCV_RUNTIME mingw)

  execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpmachine
                  OUTPUT_VARIABLE OPENCV_GCC_TARGET_MACHINE
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(CMAKE_OPENCV_GCC_TARGET_MACHINE MATCHES "64")
    set(MINGW64 1)
    set(OpenCV_ARCH x64)
  else()
    set(OpenCV_ARCH x86)
  endif()
endif()

if(CMAKE_VERSION VERSION_GREATER 2.6.2)
  unset(OpenCV_CONFIG_PATH CACHE)
endif()

if(NOT OpenCV_FIND_QUIETLY)
  message(STATUS "OpenCV ARCH: ${OpenCV_ARCH}")
  message(STATUS "OpenCV RUNTIME: ${OpenCV_RUNTIME}")
  message(STATUS "OpenCV STATIC: ${OpenCV_STATIC}")
endif()

get_filename_component(OpenCV_CONFIG_PATH "${CMAKE_CURRENT_LIST_FILE}" PATH CACHE)
if(OpenCV_RUNTIME AND OpenCV_ARCH)
  if(OpenCV_STATIC AND EXISTS "${OpenCV_CONFIG_PATH}/${OpenCV_ARCH}/${OpenCV_RUNTIME}/staticlib/OpenCVConfig.cmake")
    if(OpenCV_CUDA AND EXISTS "${OpenCV_CONFIG_PATH}/gpu/${OpenCV_ARCH}/${OpenCV_RUNTIME}/staticlib/OpenCVConfig.cmake")
      set(OpenCV_LIB_PATH "${OpenCV_CONFIG_PATH}/gpu/${OpenCV_ARCH}/${OpenCV_RUNTIME}/staticlib")
    else()
      set(OpenCV_LIB_PATH "${OpenCV_CONFIG_PATH}/${OpenCV_ARCH}/${OpenCV_RUNTIME}/staticlib")
    endif()
  elseif(EXISTS "${OpenCV_CONFIG_PATH}/${OpenCV_ARCH}/${OpenCV_RUNTIME}/lib/OpenCVConfig.cmake")
    if(OpenCV_CUDA AND EXISTS "${OpenCV_CONFIG_PATH}/gpu/${OpenCV_ARCH}/${OpenCV_RUNTIME}/lib/OpenCVConfig.cmake")
      set(OpenCV_LIB_PATH "${OpenCV_CONFIG_PATH}/gpu/${OpenCV_ARCH}/${OpenCV_RUNTIME}/lib")
    else()
      set(OpenCV_LIB_PATH "${OpenCV_CONFIG_PATH}/${OpenCV_ARCH}/${OpenCV_RUNTIME}/lib")
    endif()
  endif()
endif()

if(OpenCV_LIB_PATH AND EXISTS "${OpenCV_LIB_PATH}/OpenCVConfig.cmake")
  set(OpenCV_LIB_DIR_OPT "${OpenCV_LIB_PATH}" CACHE PATH "Path where release OpenCV libraries are located" FORCE)
  set(OpenCV_LIB_DIR_DBG "${OpenCV_LIB_PATH}" CACHE PATH "Path where debug OpenCV libraries are located" FORCE)
  set(OpenCV_3RDPARTY_LIB_DIR_OPT "${OpenCV_LIB_PATH}" CACHE PATH "Path where release 3rdpaty OpenCV dependencies are located" FORCE)
  set(OpenCV_3RDPARTY_LIB_DIR_DBG "${OpenCV_LIB_PATH}" CACHE PATH "Path where debug 3rdpaty OpenCV dependencies are located" FORCE)

  include("${OpenCV_LIB_PATH}/OpenCVConfig.cmake")

  if(OpenCV_CUDA)
    set(_OpenCV_LIBS "")
    foreach(_lib ${OpenCV_LIBS})
      string(REPLACE "${OpenCV_CONFIG_PATH}/gpu/${OpenCV_ARCH}/${OpenCV_RUNTIME}" "${OpenCV_CONFIG_PATH}/${OpenCV_ARCH}/${OpenCV_RUNTIME}" _lib2 "${_lib}")
      if(NOT EXISTS "${_lib}" AND EXISTS "${_lib2}")
        list(APPEND _OpenCV_LIBS "${_lib2}")
      else()
        list(APPEND _OpenCV_LIBS "${_lib}")
      endif()
    endforeach()
    set(OpenCV_LIBS ${_OpenCV_LIBS})
  endif()
  set(OpenCV_FOUND TRUE CACHE BOOL "" FORCE)
  set(OPENCV_FOUND TRUE CACHE BOOL "" FORCE)

  if(NOT OpenCV_FIND_QUIETLY)
    message(STATUS "Found OpenCV ${OpenCV_VERSION} in ${OpenCV_LIB_PATH}")
    if(NOT OpenCV_LIB_PATH MATCHES "/staticlib")
      get_filename_component(_OpenCV_LIB_PATH "${OpenCV_LIB_PATH}/../bin" ABSOLUTE)
      file(TO_NATIVE_PATH "${_OpenCV_LIB_PATH}" _OpenCV_LIB_PATH)
      message(STATUS "You might need to add ${_OpenCV_LIB_PATH} to your PATH to be able to run your applications.")
      if(OpenCV_LIB_PATH MATCHES "/gpu/")
        string(REPLACE "\\gpu" "" _OpenCV_LIB_PATH2 "${_OpenCV_LIB_PATH}")
        message(STATUS "GPU support is enabled so you might also need ${_OpenCV_LIB_PATH2} in your PATH (it must go after the ${_OpenCV_LIB_PATH}).")
      endif()
    endif()
  endif()
else()
  if(NOT OpenCV_FIND_QUIETLY)
    message(WARNING
"Found OpenCV Windows Pack but it has not binaries compatible with your configuration.
You should manually point CMake variable OpenCV_DIR to your build of OpenCV library."
    )
  endif()
  set(OpenCV_FOUND FALSE CACHE BOOL "" FORCE)
  set(OPENCV_FOUND FALSE CACHE BOOL "" FORCE)
endif()
