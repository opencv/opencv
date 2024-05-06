# load settings in case of "try compile"
set(TOOLCHAIN_CONFIG_FILE "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/toolchain.config.cmake")
get_property(__IN_TRY_COMPILE GLOBAL PROPERTY IN_TRY_COMPILE)
if(__IN_TRY_COMPILE)
  set(TOOLCHAIN_CONFIG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../toolchain.config.cmake")
  if(NOT EXISTS "${TOOLCHAIN_CONFIG_FILE}")
    # Hack for "try_compile" commands with other binary directory
    set(TOOLCHAIN_CONFIG_FILE "${CMAKE_PLATFORM_INFO_DIR}/../toolchain.config.cmake")
    if(NOT EXISTS "${TOOLCHAIN_CONFIG_FILE}")
      message(FATAL_ERROR "Current CMake version (${CMAKE_VERSION}) is not supported")
    endif()
  endif()
  include("${TOOLCHAIN_CONFIG_FILE}")
  macro(toolchain_save_config)
    # nothing
  endmacro()
else()
  macro(toolchain_save_config)
    set(__config "#message(\"Load TOOLCHAIN config...\")\n")
    get_cmake_property(__variableNames VARIABLES)
    set(__vars_list ${ARGN})
    list(APPEND __vars_list
        ${TOOLCHAIN_CONFIG_VARS}
        CMAKE_SYSTEM_NAME
        CMAKE_SYSTEM_VERSION
        CMAKE_SYSTEM_PROCESSOR
        CMAKE_C_COMPILER
        CMAKE_CXX_COMPILER
        CMAKE_C_FLAGS
        CMAKE_CXX_FLAGS
        CMAKE_SHARED_LINKER_FLAGS
        CMAKE_MODULE_LINKER_FLAGS
        CMAKE_EXE_LINKER_FLAGS
        CMAKE_SKIP_RPATH
        CMAKE_FIND_ROOT_PATH
    )
    foreach(__var ${__variableNames})
      foreach(_v ${__vars_list})
        if("x${__var}" STREQUAL "x${_v}")
          if(${__var} MATCHES " ")
            set(__config "${__config}set(${__var} \"${${__var}}\")\n")
          else()
            set(__config "${__config}set(${__var} ${${__var}})\n")
          endif()
        endif()
      endforeach()
    endforeach()
    if(EXISTS "${TOOLCHAIN_CONFIG_FILE}")
      file(READ "${TOOLCHAIN_CONFIG_FILE}" __config_old)
    endif()
    if("${__config_old}" STREQUAL "${__config}")
      # nothing
    else()
      #message("Update TOOLCHAIN config: ${__config}")
      file(WRITE "${TOOLCHAIN_CONFIG_FILE}" "${__config}")
    endif()
    unset(__config)
    unset(__config_old)
    unset(__vars_list)
    unset(__variableNames)
  endmacro()
endif() # IN_TRY_COMPILE

if((IPHONEOS OR IPHONESIMULATOR) AND NOT DEFINED IOS_ARCH)
  message(FATAL_ERROR "iOS toolchain requires ARCH option for proper configuration of compiler flags")
endif()
if((VISIONOS OR VISIONSIMULATOR) AND NOT DEFINED VISIONOS_ARCH)
  message(FATAL_ERROR "visionOS toolchain requires ARCH option for proper configuration of compiler flags")
endif()
if((IOS_ARCH MATCHES "^arm64") OR (VISIONOS_ARCH MATCHES "^arm64"))
  set(AARCH64 1)
elseif(IOS_ARCH MATCHES "^armv")
  set(ARM 1)
elseif((IOS_ARCH MATCHES "^x86_64") OR (VISIONOS_ARCH MATCHES "^x86_64"))
  set(X86_64 1)
elseif(IOS_ARCH MATCHES "^i386")
  set(X86 1)
else()
  if(IPHONEOS OR IPHONESIMULATOR)
    message(FATAL_ERROR "invalid value of IOS_ARCH='${IOS_ARCH}'")
  elseif(VISIONOS OR VISIONSIMULATOR)
    message(FATAL_ERROR "invalid value of VISIONOS_ARCH='${VISIONOS_ARCH}'")
  endif()
endif()

if(NOT DEFINED CMAKE_OSX_SYSROOT)
  if(IPHONEOS)
    set(CMAKE_OSX_SYSROOT "iphoneos")
    set(SYSTEM "iOS")
    set(ARCH "${IOS_ARCH}")
  elseif(IPHONESIMULATOR)
    set(CMAKE_OSX_SYSROOT "iphonesimulator")
    set(SYSTEM "iOS")
    set(ARCH "${IOS_ARCH}")
  elseif(VISIONOS)
    set(CMAKE_OSX_SYSROOT "xros")
    set(SYSTEM "visionOS")
    set(ARCH "${VISIONOS_ARCH}")
  elseif(VISIONSIMULATOR)
    set(CMAKE_OSX_SYSROOT "xrsimulator")
    set(SYSTEM "visionOS")
    set(ARCH "${VISIONOS_ARCH}")
  elseif(MAC_CATALYST)
    # Use MacOS SDK for Catalyst builds
    set(CMAKE_OSX_SYSROOT "macosx")
  endif()
endif()
set(CMAKE_MACOSX_BUNDLE YES)
set(CMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED "NO")

if(APPLE_FRAMEWORK AND NOT BUILD_SHARED_LIBS)
  set(CMAKE_OSX_ARCHITECTURES "${ARCH}" CACHE INTERNAL "Build architecture for iOS/visionOS" FORCE)
endif()

if((IPHONEOS OR IPHONESIMULATOR) AND NOT DEFINED IPHONEOS_DEPLOYMENT_TARGET)
  if(NOT DEFINED ENV{IPHONEOS_DEPLOYMENT_TARGET})
    message(FATAL_ERROR "IPHONEOS_DEPLOYMENT_TARGET is not specified")
  endif()
  set(IPHONEOS_DEPLOYMENT_TARGET "$ENV{IPHONEOS_DEPLOYMENT_TARGET}")
  set(DEPLOYMENT_TARGET "${IPHONEOS_DEPLOYMENT_TARGET}")
  set(DEPLOYMENT_TARGET_CMDLINE "IPHONEOS_DEPLOYMENT_TARGET=${IPHONEOS_DEPLOYMENT_TARGET}")
endif()

if((VISIONOS OR VISIONSIMULATOR) AND NOT DEFINED XROS_DEPLOYMENT_TARGET)
  if(NOT DEFINED ENV{XROS_DEPLOYMENT_TARGET})
    message(FATAL_ERROR "XROS_DEPLOYMENT_TARGET is not specified")
  endif()
  set(XROS_DEPLOYMENT_TARGET "$ENV{XROS_DEPLOYMENT_TARGET}")
  set(DEPLOYMENT_TARGET "${XROS_DEPLOYMENT_TARGET}")
  set(DEPLOYMENT_TARGET_CMDLINE "XROS_DEPLOYMENT_TARGET=${XROS_DEPLOYMENT_TARGET}")
endif()

if(NOT __IN_TRY_COMPILE)
  set(_xcodebuild_wrapper "")
  if(NOT (CMAKE_VERSION VERSION_LESS "3.25.1"))  # >= 3.25.1
    # no workaround is required (#23156)
  elseif(NOT (CMAKE_VERSION VERSION_LESS "3.25.0"))  # >= 3.25.0 < 3.25.1
    if(NOT OPENCV_SKIP_MESSAGE_ISSUE_23156)
      message(FATAL_ERROR "OpenCV: Please upgrade CMake to 3.25.1+. Current CMake version is ${CMAKE_VERSION}. Details: https://github.com/opencv/opencv/issues/23156")
    endif()
  else()  # < 3.25.0, apply workaround from #13912
    set(_xcodebuild_wrapper "${CMAKE_BINARY_DIR}/xcodebuild_wrapper")
  endif()
  if(_xcodebuild_wrapper AND NOT EXISTS "${_xcodebuild_wrapper}")
    set(_xcodebuild_wrapper_tmp "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/xcodebuild_wrapper")
    if(NOT DEFINED CMAKE_MAKE_PROGRAM)  # empty since CMake 3.10
      find_program(XCODEBUILD_PATH "xcodebuild")
      if(NOT XCODEBUILD_PATH)
        message(FATAL_ERROR "Specify CMAKE_MAKE_PROGRAM variable ('xcodebuild' absolute path)")
      endif()
      set(CMAKE_MAKE_PROGRAM "${XCODEBUILD_PATH}")
    endif()
    if(CMAKE_MAKE_PROGRAM STREQUAL _xcodebuild_wrapper)
      message(FATAL_ERROR "Can't prepare xcodebuild_wrapper")
    endif()
    if(APPLE_FRAMEWORK AND BUILD_SHARED_LIBS)
      set(XCODEBUILD_EXTRA_ARGS "${XCODEBUILD_EXTRA_ARGS} ${DEPLOYMENT_TARGET_CMDLINE} CODE_SIGN_IDENTITY='' CODE_SIGNING_REQUIRED=NO -sdk ${CMAKE_OSX_SYSROOT}")
    else()
      set(XCODEBUILD_EXTRA_ARGS "${XCODEBUILD_EXTRA_ARGS} ${DEPLOYMENT_TARGET_CMDLINE} CODE_SIGN_IDENTITY='' CODE_SIGNING_REQUIRED=NO ARCHS=${ARCH} -sdk ${CMAKE_OSX_SYSROOT}")
    endif()
    configure_file("${CMAKE_CURRENT_LIST_DIR}/xcodebuild_wrapper.in" "${_xcodebuild_wrapper_tmp}" @ONLY)
    file(COPY "${_xcodebuild_wrapper_tmp}" DESTINATION ${CMAKE_BINARY_DIR} FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
  endif()
  if(_xcodebuild_wrapper)
    set(CMAKE_MAKE_PROGRAM "${_xcodebuild_wrapper}" CACHE INTERNAL "" FORCE)
  endif()
endif()

# Standard settings
set(CMAKE_SYSTEM_NAME "${SYSTEM}")

# Apple Framework settings
if(APPLE_FRAMEWORK AND BUILD_SHARED_LIBS)
  set(CMAKE_SYSTEM_VERSION "${DEPLOYMENT_TARGET}")
  set(CMAKE_C_SIZEOF_DATA_PTR 4)
  set(CMAKE_CXX_SIZEOF_DATA_PTR 4)
else()
  set(CMAKE_SYSTEM_VERSION "${DEPLOYMENT_TARGET}")
  set(CMAKE_SYSTEM_PROCESSOR "${ARCH}")

  if(AARCH64 OR X86_64)
    set(CMAKE_C_SIZEOF_DATA_PTR 8)
    set(CMAKE_CXX_SIZEOF_DATA_PTR 8)
  else()
    set(CMAKE_C_SIZEOF_DATA_PTR 4)
    set(CMAKE_CXX_SIZEOF_DATA_PTR 4)
  endif()
endif()

# Include extra modules for the iOS platform files
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/platforms/ios/cmake/Modules")

# Force the compilers to clang for iOS
include(CMakeForceCompiler)
#CMAKE_FORCE_C_COMPILER (clang GNU)
#CMAKE_FORCE_CXX_COMPILER (clang++ GNU)

set(CMAKE_C_HAS_ISYSROOT 1)
set(CMAKE_CXX_HAS_ISYSROOT 1)
set(CMAKE_C_COMPILER_ABI ELF)
set(CMAKE_CXX_COMPILER_ABI ELF)

# Skip the platform compiler checks for cross compiling
set(CMAKE_CXX_COMPILER_WORKS TRUE)
set(CMAKE_C_COMPILER_WORKS TRUE)

if(NOT CMAKE_FIND_ROOT_PATH_MODE_LIBRARY)
  set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
endif()

if(NOT CMAKE_FIND_ROOT_PATH_MODE_INCLUDE)
  set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
endif()

if(NOT CMAKE_FIND_ROOT_PATH_MODE_PACKAGE)
  set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
endif()

if(NOT CMAKE_FIND_ROOT_PATH_MODE_PROGRAM)
  set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
endif()

toolchain_save_config(IOS_ARCH VISIONOS_ARCH ARCH DEPLOYMENT_TARGET)
