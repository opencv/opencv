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

if(NOT DEFINED IOS_ARCH)
  message(FATAL_ERROR "iOS toolchain requires ARCH option for proper configuration of compiler flags")
endif()
if(IOS_ARCH MATCHES "^arm64")
  set(AARCH64 1)
elseif(IOS_ARCH MATCHES "^armv")
  set(ARM 1)
elseif(IOS_ARCH MATCHES "^x86_64")
  set(X86_64 1)
elseif(IOS_ARCH MATCHES "^i386")
  set(X86 1)
else()
  message(FATAL_ERROR "iOS toolchain doesn't recognize ARCH='${IOS_ARCH}' value")
endif()

if(NOT DEFINED CMAKE_OSX_SYSROOT)
  if(IPHONEOS)
    set(CMAKE_OSX_SYSROOT "iphoneos")
  else()
    set(CMAKE_OSX_SYSROOT "iphonesimulator")
  endif()
endif()
set(CMAKE_MACOSX_BUNDLE YES)
set(CMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED "NO")

if(APPLE_FRAMEWORK AND NOT BUILD_SHARED_LIBS)
  set(CMAKE_OSX_ARCHITECTURES "${IOS_ARCH}" CACHE INTERNAL "Build architecture for iOS" FORCE)
endif()

if(NOT DEFINED IPHONEOS_DEPLOYMENT_TARGET)
  if(NOT DEFINED ENV{IPHONEOS_DEPLOYMENT_TARGET})
    message(FATAL_ERROR "IPHONEOS_DEPLOYMENT_TARGET is not specified")
  endif()
  set(IPHONEOS_DEPLOYMENT_TARGET "$ENV{IPHONEOS_DEPLOYMENT_TARGET}")
endif()

if(NOT __IN_TRY_COMPILE)
  set(_xcodebuild_wrapper "${CMAKE_BINARY_DIR}/xcodebuild_wrapper")
  if(NOT EXISTS "${_xcodebuild_wrapper}")
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
      set(XCODEBUILD_EXTRA_ARGS "${XCODEBUILD_EXTRA_ARGS} IPHONEOS_DEPLOYMENT_TARGET=${IPHONEOS_DEPLOYMENT_TARGET} -sdk ${CMAKE_OSX_SYSROOT}")
    else()
      set(XCODEBUILD_EXTRA_ARGS "${XCODEBUILD_EXTRA_ARGS} IPHONEOS_DEPLOYMENT_TARGET=${IPHONEOS_DEPLOYMENT_TARGET} ARCHS=${IOS_ARCH} -sdk ${CMAKE_OSX_SYSROOT}")
    endif()
    configure_file("${CMAKE_CURRENT_LIST_DIR}/xcodebuild_wrapper.in" "${_xcodebuild_wrapper_tmp}" @ONLY)
    file(COPY "${_xcodebuild_wrapper_tmp}" DESTINATION ${CMAKE_BINARY_DIR} FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
  endif()
  set(CMAKE_MAKE_PROGRAM "${_xcodebuild_wrapper}" CACHE INTERNAL "" FORCE)
endif()

# Standard settings
set(CMAKE_SYSTEM_NAME iOS)

# Apple Framework settings
if(APPLE_FRAMEWORK AND BUILD_SHARED_LIBS)
  set(CMAKE_SYSTEM_VERSION "${IPHONEOS_DEPLOYMENT_TARGET}")
  set(CMAKE_C_SIZEOF_DATA_PTR 4)
  set(CMAKE_CXX_SIZEOF_DATA_PTR 4)
else()
  set(CMAKE_SYSTEM_VERSION "${IPHONEOS_DEPLOYMENT_TARGET}")
  set(CMAKE_SYSTEM_PROCESSOR "${IOS_ARCH}")

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

# Search for programs in the build host directories
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM ONLY)
#   for libraries and headers in the target directories
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

toolchain_save_config(IOS_ARCH IPHONEOS_DEPLOYMENT_TARGET)
