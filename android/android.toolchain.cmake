# ------------------------------------------------------------------------------
#  Android CMake toolchain file, for use with the Android NDK r5-r7
#  Requires cmake 2.6.3 or newer (2.8.3 or newer is recommended).
#  See home page: http://code.google.com/p/android-cmake/
#
#  Usage Linux:
#   $ export ANDROID_NDK=/absolute/path/to/the/android-ndk
#   $ mkdir build && cd build
#   $ cmake -DCMAKE_TOOLCHAIN_FILE=path/to/the/android.toolchain.cmake ..
#   $ make -j8
#
#  Usage Linux (using standalone toolchain):
#   $ export ANDROID_STANDALONE_TOOLCHAIN=/absolute/path/to/android-toolchain
#   $ mkdir build && cd build
#   $ cmake -DCMAKE_TOOLCHAIN_FILE=path/to/the/android.toolchain.cmake ..
#   $ make -j8
#
#  Usage Windows:
#     You need native port of make to build your project.
#     Android NDK r7 already has make.exe on board.
#     For older NDK you have to install it separately.
#     For example, this one: http://gnuwin32.sourceforge.net/packages/make.htm
#
#   $ SET ANDROID_NDK=C:\absolute\path\to\the\android-ndk
#   $ mkdir build && cd build
#   $ cmake.exe -G"MinGW Makefiles"
#       -DCMAKE_TOOLCHAIN_FILE=path\to\the\android.toolchain.cmake
#       -DCMAKE_MAKE_PROGRAM="%ANDROID_NDK%\prebuilt\windows\bin\make.exe" ..
#   $ "%ANDROID_NDK%\prebuilt\windows\bin\make.exe"
#
#
#  Options (can be set as cmake parameters: -D<option_name>=<value>):
#    ANDROID_NDK=/opt/android-ndk - path to the NDK root.
#      Can be set as environment variable. Can be set only at first cmake run.
#
#    ANDROID_STANDALONE_TOOLCHAIN=/opt/android-toolchain - path to the 
#      standalone toolchain. This option is not used if full NDK is found
#      (ignored if ANDROID_NDK is set).
#      Can be set as environment variable. Can be set only at first cmake run.
#
#    ANDROID_ABI=armeabi-v7a -  specifies the target Application Binary
#      Interface (ABI). This option nearly matches to the APP_ABI variable 
#      used by ndk-build tool from Android NDK.
#      Possible values are:
#        "armeabi" - matches to the NDK ABI with the same name.
#           See ${ANDROID_NDK}/docs/CPU-ARCH-ABIS.html for the documentation.
#        "armeabi-v7a" - matches to the NDK ABI with the same name.
#           See ${ANDROID_NDK}/docs/CPU-ARCH-ABIS.html for the documentation.
#        "armeabi-v7a with NEON" - same as armeabi-v7a, but
#            sets NEON as floating-point unit
#        "armeabi-v7a with VFPV3" - same as armeabi-v7a, but
#            sets VFPV3 as floating-point unit (has 32 registers instead of 16).
#        "armeabi-v6 with VFP" - tuned for ARMv6 processors having VFP.
#        "x86" - matches to the NDK ABI with the same name.
#           See ${ANDROID_NDK}/docs/CPU-ARCH-ABIS.html for the documentation.
#
#    ANDROID_NATIVE_API_LEVEL=android-8 - level of Android API compile for.
#      Option is read-only when standalone toolchain used.
#
#    ANDROID_FORCE_ARM_BUILD=OFF - set true to generate 32-bit ARM instructions
#      instead of Thumb-1. Is not available for "x86" (inapplicable) and
#      "armeabi-v6 with VFP" (forced) ABIs.
#
#    ANDROID_NO_UNDEFINED=ON - set true to show all undefined symbols as linker
#      errors even if they are not used.
#
#    ANDROID_SO_UNDEFINED=OFF - set true to allow undefined symbols in shared
#      libraries. Automatically turned on for NDK r5x and r6x due to GLESv2
#      problems.
#
#    LIBRARY_OUTPUT_PATH_ROOT=${CMAKE_SOURCE_DIR} - where to output binary
#      files. See additional details below.
#
#    ANDROID_SET_OBSOLETE_VARIABLES=ON - it set, then toolchain defines some
#      obsolete variables which were set by previous versions of this file for
#      backward compatibility.
#
#
#  What?:
#    android-cmake toolchain searches for NDK/toolchain in the following order:
#      ANDROID_NDK - cmake parameter
#      ANDROID_NDK - environment variable
#      ANDROID_STANDALONE_TOOLCHAIN - cmake parameter
#      ANDROID_STANDALONE_TOOLCHAIN - environment variable
#      ANDROID_NDK - default locations
#      ANDROID_STANDALONE_TOOLCHAIN - default locations
#
#    Make sure to do the following in your scripts:
#      SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${my_cxx_flags}" )
#      SET( CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  ${my_cxx_flags}" )
#      The flags will be prepopulated with critical flags, so don't loose them.
#      Also be aware that toolchain also sets configuration-specific compiler
#      flags and linker flags.
#    
#    ANDROID and BUILD_ANDROID will be set to true, you may test any of these 
#    variables to make necessary Android-specific configuration changes.
#    
#    Also ARMEABI or ARMEABI_V7A or X86 will be set true, mutually exclusive.
#    NEON option will be set true if VFP is set to NEON.
#
#    LIBRARY_OUTPUT_PATH_ROOT should be set in cache to determine where Android
#    libraries will be installed.
#    Default is ${CMAKE_SOURCE_DIR}, and the android libs will always be
#    under the ${LIBRARY_OUTPUT_PATH_ROOT}/libs/${ANDROID_NDK_ABI_NAME}
#    (depending on the target ABI). This is convenient for Android packaging.
#
#  Change Log:
#   - initial version December 2010 Ethan Rublee ethan.ruble@gmail.com
#   - modified April 2011 Andrey Kamaev andrey.kamaev@itseez.com
#     [+] added possibility to build with NDK (without standalone toolchain)
#     [+] support cross-compilation on Windows (native, no cygwin support)
#     [+] added compiler option to force "char" type to be signed
#     [+] added toolchain option to compile to 32-bit ARM instructions
#     [+] added toolchain option to disable SWIG search
#     [+] added platform "armeabi-v7a with VFPV3"
#     [~] ARM_TARGETS renamed to ARM_TARGET
#     [+] EXECUTABLE_OUTPUT_PATH is set by toolchain (required on Windows)
#     [~] Fixed bug with ANDROID_API_LEVEL variable
#     [~] turn off SWIG search if it is not found first time
#   - modified May 2011 Andrey Kamaev andrey.kamaev@itseez.com
#     [~] ANDROID_LEVEL is renamed to ANDROID_API_LEVEL
#     [+] ANDROID_API_LEVEL is detected by toolchain if not specified
#     [~] added guard to prevent changing of output directories on the first
#         cmake pass
#     [~] toolchain exits with error if ARM_TARGET is not recognized
#   - modified June 2011 Andrey Kamaev andrey.kamaev@itseez.com
#     [~] default NDK path is updated for version r5c 
#     [+] variable CMAKE_SYSTEM_PROCESSOR is set based on ARM_TARGET
#     [~] toolchain install directory is added to linker paths
#     [-] removed SWIG-related stuff from toolchain
#     [+] added macro find_host_package, find_host_program to search
#         packages/programs on the host system
#     [~] fixed path to STL library
#   - modified July 2011 Andrey Kamaev andrey.kamaev@itseez.com
#     [~] fixed options caching
#     [~] search for all supported NDK versions
#     [~] allowed spaces in NDK path
#   - modified September 2011 Andrey Kamaev andrey.kamaev@itseez.com
#     [~] updated for NDK r6b
#   - modified November 2011 Andrey Kamaev andrey.kamaev@itseez.com
#     [*] rewritten for NDK r7
#     [+] x86 toolchain support (experimental)
#     [+] added "armeabi-v6 with VFP" ABI for ARMv6 processors.
#     [~] improved compiler and linker flags management
#     [+] support different build flags for Release and Debug configurations
#     [~] by default compiler flags the same as used by ndk-build (but only
#         where reasonable)
#     [~] ANDROID_NDK_TOOLCHAIN_ROOT is splitted to ANDROID_STANDALONE_TOOLCHAIN
#         and ANDROID_TOOLCHAIN_ROOT
#     [~] ARM_TARGET is renamed to ANDROID_ABI
#     [~] ARMEABI_NDK_NAME is renamed to ANDROID_NDK_ABI_NAME
#     [~] ANDROID_API_LEVEL is renamed to ANDROID_NATIVE_API_LEVEL
#   - modified January 2012 Andrey Kamaev andrey.kamaev@itseez.com
#     [+] added stlport_static support (experimental)
#     [+] added special check for cygwin
#     [+] filtered out hidden files (starting with .) while globbing inside NDK
#     [+] automatically applied GLESv2 linkage fix for NDK revisions 5-6
#     [+] added ANDROID_GET_ABI_RAWNAME to get NDK ABI names by CMake flags
#   - modified February 2012
#     [+] mostly fixed try_compile (it was always failing for Android)
# ------------------------------------------------------------------------------

# this one is important
set( CMAKE_SYSTEM_NAME Linux )
#this one not so much
set( CMAKE_SYSTEM_VERSION 1 )

set( ANDROID_SUPPORTED_NDK_VERSIONS -r7b -r7 -r6b -r6 -r5c -r5b -r5 "" )
if( CMAKE_HOST_WIN32 )
 file( TO_CMAKE_PATH "$ENV{PROGRAMFILES}" ANDROID_NDK_SEARCH_PATHS )
 set( ANDROID_NDK_SEARCH_PATHS "${ANDROID_NDK_SEARCH_PATHS}/android-ndk" "$ENV{SystemDrive}/NVPACK/android-ndk" )
else()
 set( ANDROID_NDK_SEARCH_PATHS /opt/android-ndk "$ENV{HOME}/NVPACK/android-ndk" )
endif()
set( ANDROID_STANDALONE_TOOLCHAIN_SEARCH_PATH /opt/android-toolchain )
set( ANDROID_SUPPORTED_ABIS_arm "armeabi-v7a;armeabi;armeabi-v7a with NEON;armeabi-v7a with VFPV3;armeabi-v6 with VFP" )
set( ANDROID_SUPPORTED_ABIS_x86 "x86" )

set( ANDROID_DEFAULT_NDK_API_LEVEL 8 )
set( ANDROID_DEFAULT_NDK_API_LEVEL_x86 9 )


macro( __LIST_FILTER listvar regex )
 if( ${listvar} )
  foreach( __val ${${listvar}} )
   if( __val MATCHES "${regex}" )
    list( REMOVE_ITEM ${listvar} "${__val}" )
   endif()
  endforeach()
 endif()
endmacro()

macro( __INIT_VARIABLE var_name )
 set( __test_path 0 )
 foreach( __var ${ARGN} )
  if( __var STREQUAL "PATH" )
   set( __test_path 1 )
   break()
  endif()
 endforeach()
 if( __test_path AND NOT EXISTS "${${var_name}}" )
  unset( ${var_name} CACHE )
 endif()
 if( "${${var_name}}" STREQUAL "" )
  set( __values 0 )
  foreach( __var ${ARGN} )
   if( __var STREQUAL "VALUES" )
    set( __values 1 )
   elseif( NOT __var STREQUAL "PATH" )
    set( __obsolete 0 )
    if( __var MATCHES "^OBSOLETE_.*$" )
     string( REPLACE "OBSOLETE_" "" __var "${__var}" )
     set( __obsolete 1 )
    endif()
    if( __var MATCHES "^ENV_.*$" )
     string( REPLACE "ENV_" "" __var "${__var}" )
     set( __value "$ENV{${__var}}" )
    elseif( DEFINED ${__var} )
     set( __value "${${__var}}" )
    else()
     if( __values )
      set( __value "${__var}" )
     else()
      set( __value "" )
     endif()
    endif()
    if( NOT "${__value}" STREQUAL "" )
     if( __test_path )
      if( EXISTS "${__value}" )
       set( ${var_name} "${__value}" )
       if( __obsolete )
        message( WARNING "Using value of obsolete variable ${__var} as initial value for ${var_name}. Please note, that ${__var} can be completely removed in future versions of the toolchain." )
       endif()
       break()
      endif()
     else()
      set( ${var_name} "${__value}" )
       if( __obsolete )
        message( WARNING "Using value of obsolete variable ${__var} as initial value for ${var_name}. Please note, that ${__var} can be completely removed in future versions of the toolchain." )
       endif()
      break()
     endif()
    endif()
   endif()
  endforeach()
  unset( __value )
  unset( __values )
  unset( __obsolete )
 endif()
 unset( __test_path )
endmacro()

macro( __DETECT_NATIVE_API_LEVEL _var _path )
 SET( __ndkApiLevelRegex "^[\t ]*#define[\t ]+__ANDROID_API__[\t ]+([0-9]+)[\t ]*$" )
 FILE( STRINGS ${_path} __apiFileContent REGEX "${__ndkApiLevelRegex}" )
 if( NOT __apiFileContent )
  message( SEND_ERROR "Could not get Android native API level. Probably you have specified invalid level value, or your copy of NDK/toolchain is broken." )
 endif()
 string( REGEX REPLACE "${__ndkApiLevelRegex}" "\\1" ${_var} "${__apiFileContent}" )
 unset( __apiFileContent )
 unset( __ndkApiLevelRegex )
endmacro()

macro( __DETECT_TOOLCHAIN_MACHINE_NAME _var _root )
 file( GLOB __gccExePath "${_root}/bin/*-gcc${TOOL_OS_SUFFIX}" )
 __LIST_FILTER( __gccExePath "bin/[.].*-gcc${TOOL_OS_SUFFIX}$" )
 list( LENGTH __gccExePath __gccExePathsCount )
 if( NOT __gccExePathsCount EQUAL 1 )
  message( WARNING "Could not uniquely determine machine name for compiler from ${_root}." )
  set( ${_var} "" )
 else()
  get_filename_component( __gccExeName "${__gccExePath}" NAME_WE )
  string( REPLACE "-gcc" "" ${_var} "${__gccExeName}" )
 endif()
 unset( __gccExePath )
 unset( __gccExePathsCount )
 unset( __gccExeName )
endmacro()

macro( __COPY_IF_DIFFERENT _source _destination )
 execute_process( COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${_source}" "${_destination}" RESULT_VARIABLE __fileCopyProcess )
 if( NOT __fileCopyProcess EQUAL 0 OR NOT EXISTS "${_destination}")
  message( SEND_ERROR "Failed copying of ${_source} to the ${_destination}" )
 endif()
 unset( __fileCopyProcess )
endmacro()


#stl version: by default gnustl_static will be used
set( ANDROID_USE_STLPORT FALSE CACHE BOOL "Experimental: use stlport_static instead of gnustl_static")
mark_as_advanced( ANDROID_USE_STLPORT )

#fight against cygwin
set( ANDROID_FORBID_SYGWIN TRUE CACHE BOOL "Prevent cmake from working under cygwin and using cygwin tools")
mark_as_advanced( ANDROID_FORBID_SYGWIN )
if( ANDROID_FORBID_SYGWIN )
 if( CYGWIN )
  message( FATAL_ERROR "Android NDK and android-cmake toolchain are not welcome Cygwin. It is unlikely that this cmake toolchain will work under cygwin. But if you want to try then you can set cmake variable ANDROID_FORBID_SYGWIN to FALSE and rerun cmake." )
 endif()

 if( CMAKE_HOST_WIN32 )
  #remove cygwin from PATH
  set( __new_path "$ENV{PATH}")
  __LIST_FILTER( __new_path "cygwin" )
  set(ENV{PATH} "${__new_path}")
  unset(__new_path)
 endif()
endif()

#detect current host platform
set( TOOL_OS_SUFFIX "" )
if( CMAKE_HOST_APPLE )
 set( ANDROID_NDK_HOST_SYSTEM_NAME "darwin-x86" )
elseif( CMAKE_HOST_WIN32 )
 set( ANDROID_NDK_HOST_SYSTEM_NAME "windows" )
 set( TOOL_OS_SUFFIX ".exe" )
elseif( CMAKE_HOST_UNIX )
 set( ANDROID_NDK_HOST_SYSTEM_NAME "linux-x86" )
else()
 message( FATAL_ERROR "Cross-compilation on your platform is not supported by this cmake toolchain" )
endif()

#see if we have path to Android NDK
__INIT_VARIABLE( ANDROID_NDK PATH ENV_ANDROID_NDK )
if( NOT ANDROID_NDK )
 #see if we have path to Android standalone toolchain
 __INIT_VARIABLE( ANDROID_STANDALONE_TOOLCHAIN PATH ENV_ANDROID_STANDALONE_TOOLCHAIN OBSOLETE_ANDROID_NDK_TOOLCHAIN_ROOT OBSOLETE_ENV_ANDROID_NDK_TOOLCHAIN_ROOT )

 if( NOT ANDROID_STANDALONE_TOOLCHAIN )
  #try to find Android NDK in one of the the default locations
  set( __ndkSearchPaths )
  foreach( __ndkSearchPath ${ANDROID_NDK_SEARCH_PATHS} )
   foreach( suffix ${ANDROID_SUPPORTED_NDK_VERSIONS} )
    list( APPEND __ndkSearchPaths "${__ndkSearchPath}${suffix}" )
   endforeach()
  endforeach()
  __INIT_VARIABLE( ANDROID_NDK PATH VALUES ${__ndkSearchPaths} )
  unset( __ndkSearchPaths )

  if( ANDROID_NDK )
   message( STATUS "Using default path for Android NDK: ${ANDROID_NDK}" )
   message( STATUS "  If you prefer to use a different location, please define a cmake or environment variable: ANDROID_NDK" )
  else()
   #try to find Android standalone toolchain in one of the the default locations
   __INIT_VARIABLE( ANDROID_STANDALONE_TOOLCHAIN PATH ANDROID_STANDALONE_TOOLCHAIN_SEARCH_PATH )

   if( ANDROID_STANDALONE_TOOLCHAIN )
    message( STATUS "Using default path for standalone toolchain ${ANDROID_STANDALONE_TOOLCHAIN}" )
    message( STATUS "  If you prefer to use a different location, please define the variable: ANDROID_STANDALONE_TOOLCHAIN" )
   endif( ANDROID_STANDALONE_TOOLCHAIN )
  endif( ANDROID_NDK )
 endif( NOT ANDROID_STANDALONE_TOOLCHAIN )
endif( NOT ANDROID_NDK )

#remember found paths
if( ANDROID_NDK )
 get_filename_component( ANDROID_NDK "${ANDROID_NDK}" ABSOLUTE )
 #try to detect change
 if( CMAKE_AR )
  string( LENGTH "${ANDROID_NDK}" __length )
  string( SUBSTRING "${CMAKE_AR}" 0 ${__length} __androidNdkPreviousPath )
  if( NOT __androidNdkPreviousPath STREQUAL ANDROID_NDK )
   message( FATAL_ERROR "It is not possible to change path to the NDK on subsequent run." )
  endif()
  unset( __androidNdkPreviousPath )
  unset( __length )
 endif()
 set( ANDROID_NDK "${ANDROID_NDK}" CACHE INTERNAL "Path of the Android NDK" )
 set( BUILD_WITH_ANDROID_NDK True )
elseif( ANDROID_STANDALONE_TOOLCHAIN )
 get_filename_component( ANDROID_STANDALONE_TOOLCHAIN "${ANDROID_STANDALONE_TOOLCHAIN}" ABSOLUTE )
 #try to detect change
 if( CMAKE_AR )
  string( LENGTH "${ANDROID_STANDALONE_TOOLCHAIN}" __length )
  string( SUBSTRING "${CMAKE_AR}" 0 ${__length} __androidStandaloneToolchainPreviousPath )
  if( NOT __androidStandaloneToolchainPreviousPath STREQUAL ANDROID_STANDALONE_TOOLCHAIN )
   message( FATAL_ERROR "It is not possible to change path to the Android standalone toolchain on subsequent run." )
  endif()
  unset( __androidStandaloneToolchainPreviousPath )
  unset( __length )
 endif()
 set( ANDROID_STANDALONE_TOOLCHAIN "${ANDROID_STANDALONE_TOOLCHAIN}" CACHE INTERNAL "Path of the Android standalone toolchain" )
 set( BUILD_WITH_STANDALONE_TOOLCHAIN True )
else()
 list(GET ANDROID_NDK_SEARCH_PATHS 0 ANDROID_NDK_SEARCH_PATH)
 message( FATAL_ERROR "Could not find neither Android NDK nor Android standalone toolcahin.
    You should either set an environment variable:
      export ANDROID_NDK=~/my-android-ndk
    or
      export ANDROID_STANDALONE_TOOLCHAIN=~/my-android-toolchain
    or put the toolchain or NDK in the default path:
      sudo ln -s ~/my-android-ndk ${ANDROID_NDK_SEARCH_PATH}
      sudo ln -s ~/my-android-toolchain ${ANDROID_STANDALONE_TOOLCHAIN_SEARCH_PATH}" )
endif()

#get all the details about standalone toolchain
if( BUILD_WITH_STANDALONE_TOOLCHAIN )
 __DETECT_NATIVE_API_LEVEL( ANDROID_SUPPORTED_NATIVE_API_LEVELS "${ANDROID_STANDALONE_TOOLCHAIN}/sysroot/usr/include/android/api-level.h" )
 set( __availableToolchains "standalone" )
 __DETECT_TOOLCHAIN_MACHINE_NAME( __availableToolchainMachines "${ANDROID_STANDALONE_TOOLCHAIN}" )
 if( NOT __availableToolchainMachines )
  message( FATAL_ERROR "Could not determine machine name of your toolchain. Probably your Android standalone toolchain is broken." )
 endif()
 if( __availableToolchainMachines MATCHES i686 )
  set( __availableToolchainArchs "x86" )
 elseif( __availableToolchainMachines MATCHES arm )
  set( __availableToolchainArchs "arm" )
 endif()
 if( ANDROID_COMPILER_VERSION )
  #do not run gcc every time because it is relatevely expencive
  set( __availableToolchainCompilerVersions "${ANDROID_COMPILER_VERSION}" )
 else()
  execute_process( COMMAND "${ANDROID_STANDALONE_TOOLCHAIN}/bin/${__availableToolchainMachines}-gcc${TOOL_OS_SUFFIX}" --version
   OUTPUT_VARIABLE __availableToolchainCompilerVersions OUTPUT_STRIP_TRAILING_WHITESPACE )
  string( REGEX MATCH "[0-9]+.[0-9]+.[0-9]+" __availableToolchainCompilerVersions "${__availableToolchainCompilerVersions}" )
 endif()
endif()

#get all the details about NDK
if( BUILD_WITH_ANDROID_NDK )
 file( GLOB ANDROID_SUPPORTED_NATIVE_API_LEVELS RELATIVE "${ANDROID_NDK}/platforms" "${ANDROID_NDK}/platforms/android-*" )
 string( REPLACE "android-" "" ANDROID_SUPPORTED_NATIVE_API_LEVELS "${ANDROID_SUPPORTED_NATIVE_API_LEVELS}" )
 file( GLOB __availableToolchains RELATIVE "${ANDROID_NDK}/toolchains" "${ANDROID_NDK}/toolchains/*" )
 __LIST_FILTER( __availableToolchains "^[.]" )
 set( __availableToolchainMachines "" )
 set( __availableToolchainArchs "" )
 set( __availableToolchainCompilerVersions "" )
 foreach( __toolchain ${__availableToolchains} )
  __DETECT_TOOLCHAIN_MACHINE_NAME( __machine "${ANDROID_NDK}/toolchains/${__toolchain}/prebuilt/${ANDROID_NDK_HOST_SYSTEM_NAME}" )
  if( __machine )
   string( REGEX MATCH "[0-9]+.[0-9]+.[0-9]+$" __version "${__toolchain}" )
   string( REGEX MATCH "^[^-]+" __arch "${__toolchain}" )
   list( APPEND __availableToolchainMachines ${__machine} )
   list( APPEND __availableToolchainArchs ${__arch} )
   list( APPEND __availableToolchainCompilerVersions ${__version} )
  else()
   list( REMOVE_ITEM __availableToolchains "${__toolchain}" )
  endif()
 endforeach()
 if( NOT __availableToolchains )
  message( FATAL_ERROR "Could not any working toolchain in the NDK. Probably your Android NDK is broken." )
 endif()
endif()

#build list of available ABIs
if( NOT ANDROID_SUPPORTED_ABIS )
 set( ANDROID_SUPPORTED_ABIS "" )
 set( __uniqToolchainArchNames ${__availableToolchainArchs} )
 list( REMOVE_DUPLICATES __uniqToolchainArchNames )
 list( SORT __uniqToolchainArchNames )
 foreach( __arch ${__uniqToolchainArchNames} )
  list( APPEND ANDROID_SUPPORTED_ABIS ${ANDROID_SUPPORTED_ABIS_${__arch}} )
 endforeach()
 unset( __uniqToolchainArchNames )
 if( NOT ANDROID_SUPPORTED_ABIS )
  message( FATAL_ERROR "No one of known Android ABIs is supported by this cmake toolchain." )
 endif()
endif()

#choose target ABI
__INIT_VARIABLE( ANDROID_ABI OBSOLETE_ARM_TARGET OBSOLETE_ARM_TARGETS VALUES ${ANDROID_SUPPORTED_ABIS} )
#verify that target ABI is supported
list( FIND ANDROID_SUPPORTED_ABIS "${ANDROID_ABI}" __androidAbiIdx )
if( __androidAbiIdx EQUAL -1 )
 string( REPLACE ";" "\", \"", PRINTABLE_ANDROID_SUPPORTED_ABIS  "${ANDROID_SUPPORTED_ABIS}" )
 message( FATAL_ERROR "Specified ANDROID_ABI = \"${ANDROID_ABI}\" is not supported by this cmake toolchain or your NDK/toolchain.
   Supported values are: \"${PRINTABLE_ANDROID_SUPPORTED_ABIS}\"
   " )
endif()
unset( __androidAbiIdx )

#remember target ABI
set( ANDROID_ABI "${ANDROID_ABI}" CACHE STRING "The target ABI for Android. If arm, then armeabi-v7a is recommended for hardware floating point." FORCE )

#set target ABI options
if( ANDROID_ABI STREQUAL "x86" )
 set( X86 true )
 set( ANDROID_NDK_ABI_NAME "x86" )
 set( ANDROID_ARCH_NAME "x86" )
 set( CMAKE_SYSTEM_PROCESSOR "i686" )
elseif( ANDROID_ABI STREQUAL "armeabi" )
 set( ARMEABI true )
 set( ANDROID_NDK_ABI_NAME "armeabi" )
 set( ANDROID_ARCH_NAME "arm" )
 set( CMAKE_SYSTEM_PROCESSOR "armv5te" )
elseif( ANDROID_ABI STREQUAL "armeabi-v6 with VFP" )
 set( ARMEABI_V6 true )
 set( ANDROID_NDK_ABI_NAME "armeabi" )
 set( ANDROID_ARCH_NAME "arm" )
 set( CMAKE_SYSTEM_PROCESSOR "armv6" )
 #need always fallback to older platform
 set( ARMEABI true )
elseif( ANDROID_ABI STREQUAL "armeabi-v7a")
 set( ARMEABI_V7A true )
 set( ANDROID_NDK_ABI_NAME "armeabi-v7a" )
 set( ANDROID_ARCH_NAME "arm" )
 set( CMAKE_SYSTEM_PROCESSOR "armv7-a" )
elseif( ANDROID_ABI STREQUAL "armeabi-v7a with VFPV3" )
 set( ARMEABI_V7A true )
 set( ANDROID_NDK_ABI_NAME "armeabi-v7a" )
 set( ANDROID_ARCH_NAME "arm" )
 set( CMAKE_SYSTEM_PROCESSOR "armv7-a" )
 set( VFPV3 true )
elseif( ANDROID_ABI STREQUAL "armeabi-v7a with NEON" )
 set( ARMEABI_V7A true )
 set( ANDROID_NDK_ABI_NAME "armeabi-v7a" )
 set( ANDROID_ARCH_NAME "arm" )
 set( CMAKE_SYSTEM_PROCESSOR "armv7-a" )
 set( VFPV3 true )
 set( NEON true )
else()
 message( SEND_ERROR "Unknown ANDROID_ABI=\"${ANDROID_ABI}\" is specified." )
endif()

set( ANDROID_SUPPORTED_ABIS ${ANDROID_SUPPORTED_ABIS_${ANDROID_ARCH_NAME}} CACHE INTERNAL "ANDROID_ABI can be changed only to one of these ABIs. Changing to any other ABI requires to reset cmake cache." )
if( CMAKE_VERSION VERSION_GREATER "2.8" )
 list( SORT ANDROID_SUPPORTED_ABIS_${ANDROID_ARCH_NAME} )
 set_property( CACHE ANDROID_ABI PROPERTY STRINGS ${ANDROID_SUPPORTED_ABIS_${ANDROID_ARCH_NAME}} )
endif()

if( ANDROID_ARCH_NAME STREQUAL "arm" AND NOT ARMEABI_V6 )
 __INIT_VARIABLE( ANDROID_FORCE_ARM_BUILD OBSOLETE_FORCE_ARM VALUES OFF )
 set( ANDROID_FORCE_ARM_BUILD ${ANDROID_FORCE_ARM_BUILD} CACHE BOOL "Use 32-bit ARM instructions instead of Thumb-1" FORCE )
 mark_as_advanced( ANDROID_FORCE_ARM_BUILD )
else()
 unset( ANDROID_FORCE_ARM_BUILD CACHE )
endif()

#choose toolchain
if( ANDROID_TOOLCHAIN_NAME )
 list( FIND __availableToolchains "${ANDROID_TOOLCHAIN_NAME}" __toolchainIdx )
 if( __toolchainIdx EQUAL -1 )
  message( FATAL_ERROR "Previously selected toolchain \"${ANDROID_TOOLCHAIN_NAME}\" is missing. You need to remove CMakeCache.txt and rerun cmake manually to change the toolchain" )
 endif()
 list( GET __availableToolchainArchs ${__toolchainIdx} __toolchainArch )
 if( NOT __toolchainArch STREQUAL ANDROID_ARCH_NAME )
  message( SEND_ERROR "Previously selected toolchain \"${ANDROID_TOOLCHAIN_NAME}\" is not able to compile binaries for the \"${ANDROID_ARCH_NAME}\" platform." )
 endif()
else()
 set( __toolchainIdx -1 )
 set( __applicableToolchains "" )
 set( __toolchainMaxVersion "0.0.0" )
 list( LENGTH __availableToolchains __availableToolchainsCount )
 math( EXPR __availableToolchainsCount "${__availableToolchainsCount}-1" )
 foreach( __idx RANGE ${__availableToolchainsCount} )
  list( GET __availableToolchainArchs ${__idx} __toolchainArch )
  if( __toolchainArch STREQUAL ANDROID_ARCH_NAME )
   list( GET __availableToolchainCompilerVersions ${__idx} __toolchainVersion )
   if( __toolchainVersion VERSION_GREATER __toolchainMaxVersion )
    set( __toolchainMaxVersion "${__toolchainVersion}" )
    set( __toolchainIdx ${__idx} )
   endif()
  endif()
 endforeach()
 unset( __availableToolchainsCount )
 unset( __toolchainMaxVersion )
 unset( __toolchainVersion )
endif()
unset( __toolchainArch )
if( __toolchainIdx EQUAL -1 )
 message( FATAL_ERROR "No one of available compiler toolchains is able to compile for ${ANDROID_ARCH_NAME} platform." )
endif()
list( GET __availableToolchains ${__toolchainIdx} ANDROID_TOOLCHAIN_NAME )
list( GET __availableToolchainMachines ${__toolchainIdx} ANDROID_TOOLCHAIN_MACHINE_NAME )
list( GET __availableToolchainCompilerVersions ${__toolchainIdx} ANDROID_COMPILER_VERSION )
set( ANDROID_TOOLCHAIN_NAME "${ANDROID_TOOLCHAIN_NAME}" CACHE INTERNAL "Name of toolchain used" )
set( ANDROID_COMPILER_VERSION "${ANDROID_COMPILER_VERSION}" CACHE INTERNAL "compiler version from selected toolchain" )
unset( __toolchainIdx )
unset( __availableToolchains )
unset( __availableToolchainMachines )
unset( __availableToolchainArchs )
unset( __availableToolchainCompilerVersions )

#choose native API level
__INIT_VARIABLE( ANDROID_NATIVE_API_LEVEL ENV_ANDROID_NATIVE_API_LEVEL ANDROID_API_LEVEL ENV_ANDROID_API_LEVEL ANDROID_DEFAULT_NDK_API_LEVEL_${ANDROID_ARCH_NAME} ANDROID_DEFAULT_NDK_API_LEVEL )
string( REGEX MATCH "[0-9]+" ANDROID_NATIVE_API_LEVEL "${ANDROID_NATIVE_API_LEVEL}" )
#validate
list( FIND ANDROID_SUPPORTED_NATIVE_API_LEVELS "${ANDROID_NATIVE_API_LEVEL}" __levelIdx )
if( __levelIdx EQUAL -1 )
 message( SEND_ERROR "Specified Android native API level (${ANDROID_NATIVE_API_LEVEL}) is not supported by your NDK/toolchain." )
endif()
unset( __levelIdx )
if( BUILD_WITH_ANDROID_NDK )
 __DETECT_NATIVE_API_LEVEL( __realApiLevel "${ANDROID_NDK}/platforms/android-${ANDROID_NATIVE_API_LEVEL}/arch-${ANDROID_ARCH_NAME}/usr/include/android/api-level.h" )
 if( NOT __realApiLevel EQUAL ANDROID_NATIVE_API_LEVEL )
  message( SEND_ERROR "Specified Android API level (${ANDROID_NATIVE_API_LEVEL}) does not match to the level found (${__realApiLevel}). Probably your copy of NDK is broken." )
 endif()
 unset( __realApiLevel )
endif()
set( ANDROID_NATIVE_API_LEVEL "${ANDROID_NATIVE_API_LEVEL}" CACHE STRING "Android API level for native code" FORCE )
if( CMAKE_VERSION VERSION_GREATER "2.8" )
 list( SORT ANDROID_SUPPORTED_NATIVE_API_LEVELS )
 set_property( CACHE ANDROID_NATIVE_API_LEVEL PROPERTY STRINGS ${ANDROID_SUPPORTED_NATIVE_API_LEVELS} )
endif()

#setup paths
if( BUILD_WITH_STANDALONE_TOOLCHAIN )
 set( ANDROID_TOOLCHAIN_ROOT "${ANDROID_STANDALONE_TOOLCHAIN}" )
 set( ANDROID_SYSROOT "${ANDROID_STANDALONE_TOOLCHAIN}/sysroot" )
 set( __stlLibPath "${ANDROID_STANDALONE_TOOLCHAIN}/${ANDROID_TOOLCHAIN_MACHINE_NAME}/lib" )
endif()
if( BUILD_WITH_ANDROID_NDK )
 set( ANDROID_TOOLCHAIN_ROOT "${ANDROID_NDK}/toolchains/${ANDROID_TOOLCHAIN_NAME}/prebuilt/${ANDROID_NDK_HOST_SYSTEM_NAME}" )
 set( ANDROID_SYSROOT "${ANDROID_NDK}/platforms/android-${ANDROID_NATIVE_API_LEVEL}/arch-${ANDROID_ARCH_NAME}" )
 if( ANDROID_USE_STLPORT )
  set( __stlIncludePath "${ANDROID_NDK}/sources/cxx-stl/stlport/stlport" )
  set( __stlLibPath "${ANDROID_NDK}/sources/cxx-stl/stlport/libs/${ANDROID_NDK_ABI_NAME}" )
 else()
  set( __stlIncludePath "${ANDROID_NDK}/sources/cxx-stl/gnu-libstdc++/include" )
  set( __stlLibPath "${ANDROID_NDK}/sources/cxx-stl/gnu-libstdc++/libs/${ANDROID_NDK_ABI_NAME}" )
 endif()
endif()

# specify the cross compiler
set( CMAKE_C_COMPILER   "${ANDROID_TOOLCHAIN_ROOT}/bin/${ANDROID_TOOLCHAIN_MACHINE_NAME}-gcc${TOOL_OS_SUFFIX}"     CACHE PATH "gcc" )
set( CMAKE_CXX_COMPILER "${ANDROID_TOOLCHAIN_ROOT}/bin/${ANDROID_TOOLCHAIN_MACHINE_NAME}-g++${TOOL_OS_SUFFIX}"     CACHE PATH "g++" )
set( CMAKE_STRIP        "${ANDROID_TOOLCHAIN_ROOT}/bin/${ANDROID_TOOLCHAIN_MACHINE_NAME}-strip${TOOL_OS_SUFFIX}"   CACHE PATH "strip" )
#there may be a way to make cmake deduce these TODO deduce the rest of the tools
set( CMAKE_AR           "${ANDROID_TOOLCHAIN_ROOT}/bin/${ANDROID_TOOLCHAIN_MACHINE_NAME}-ar${TOOL_OS_SUFFIX}"      CACHE PATH "archive" )
set( CMAKE_LINKER       "${ANDROID_TOOLCHAIN_ROOT}/bin/${ANDROID_TOOLCHAIN_MACHINE_NAME}-ld${TOOL_OS_SUFFIX}"      CACHE PATH "linker" )
set( CMAKE_NM           "${ANDROID_TOOLCHAIN_ROOT}/bin/${ANDROID_TOOLCHAIN_MACHINE_NAME}-nm${TOOL_OS_SUFFIX}"      CACHE PATH "nm" )
set( CMAKE_OBJCOPY      "${ANDROID_TOOLCHAIN_ROOT}/bin/${ANDROID_TOOLCHAIN_MACHINE_NAME}-objcopy${TOOL_OS_SUFFIX}" CACHE PATH "objcopy" )
set( CMAKE_OBJDUMP      "${ANDROID_TOOLCHAIN_ROOT}/bin/${ANDROID_TOOLCHAIN_MACHINE_NAME}-objdump${TOOL_OS_SUFFIX}" CACHE PATH "objdump" )
set( CMAKE_RANLIB       "${ANDROID_TOOLCHAIN_ROOT}/bin/${ANDROID_TOOLCHAIN_MACHINE_NAME}-ranlib${TOOL_OS_SUFFIX}"  CACHE PATH "ranlib" )

#export directories
set( ANDROID_SYSTEM_INCLUDE_DIRS "" )
set( ANDROID_SYSTEM_LIB_DIRS "" )

#setup output directories
set( LIBRARY_OUTPUT_PATH_ROOT ${CMAKE_SOURCE_DIR} CACHE PATH "root for library output, set this to change where android libs are installed to" )
set( CMAKE_INSTALL_PREFIX "${ANDROID_TOOLCHAIN_ROOT}/user" CACHE STRING "path for installing" )

set( DO_NOT_CHANGE_OUTPUT_PATHS_ON_FIRST_PASS OFF CACHE BOOL "" )
if( DO_NOT_CHANGE_OUTPUT_PATHS_ON_FIRST_PASS )
 if( EXISTS "${CMAKE_SOURCE_DIR}/jni/CMakeLists.txt" )
  set( EXECUTABLE_OUTPUT_PATH "${LIBRARY_OUTPUT_PATH_ROOT}/bin/${ANDROID_NDK_ABI_NAME}" CACHE PATH "Output directory for applications" )
 else()
  set( EXECUTABLE_OUTPUT_PATH "${LIBRARY_OUTPUT_PATH_ROOT}/bin" CACHE PATH "Output directory for applications" )
 endif()
 set( LIBRARY_OUTPUT_PATH "${LIBRARY_OUTPUT_PATH_ROOT}/libs/${ANDROID_NDK_ABI_NAME}" CACHE PATH "path for android libs" )
endif()
set( DO_NOT_CHANGE_OUTPUT_PATHS_ON_FIRST_PASS ON CACHE INTERNAL "" FORCE )

#includes
list( APPEND ANDROID_SYSTEM_INCLUDE_DIRS "${ANDROID_SYSROOT}/usr/include" )
if( __stlIncludePath AND EXISTS "${__stlIncludePath}" )
 list( APPEND ANDROID_SYSTEM_INCLUDE_DIRS "${__stlIncludePath}" )
endif()

#STL bits includes
if( __stlLibPath AND EXISTS "${__stlLibPath}/include" )
 list( APPEND ANDROID_SYSTEM_INCLUDE_DIRS "${__stlLibPath}/include" )
endif()
if( ANDROID_ARCH_NAME STREQUAL "arm" AND EXISTS "${ANDROID_TOOLCHAIN_ROOT}/${ANDROID_TOOLCHAIN_MACHINE_NAME}/include/c++/${ANDROID_COMPILER_VERSION}/${ANDROID_TOOLCHAIN_MACHINE_NAME}/${CMAKE_SYSTEM_PROCESSOR}/thumb/bits" )
 list( APPEND ANDROID_SYSTEM_INCLUDE_DIRS "${ANDROID_TOOLCHAIN_ROOT}/${ANDROID_TOOLCHAIN_MACHINE_NAME}/include/c++/${ANDROID_COMPILER_VERSION}/${ANDROID_TOOLCHAIN_MACHINE_NAME}/${CMAKE_SYSTEM_PROCESSOR}/thumb" )
elseif( EXISTS "${ANDROID_TOOLCHAIN_ROOT}/${ANDROID_TOOLCHAIN_MACHINE_NAME}/include/c++/${ANDROID_COMPILER_VERSION}/${ANDROID_TOOLCHAIN_MACHINE_NAME}/${CMAKE_SYSTEM_PROCESSOR}/bits" )
 list( APPEND ANDROID_SYSTEM_INCLUDE_DIRS "${ANDROID_TOOLCHAIN_ROOT}/${ANDROID_TOOLCHAIN_MACHINE_NAME}/include/c++/${ANDROID_COMPILER_VERSION}/${ANDROID_TOOLCHAIN_MACHINE_NAME}/${CMAKE_SYSTEM_PROCESSOR}" )
elseif( ANDROID_ARCH_NAME STREQUAL "arm" AND EXISTS "${ANDROID_TOOLCHAIN_ROOT}/${ANDROID_TOOLCHAIN_MACHINE_NAME}/include/c++/${ANDROID_COMPILER_VERSION}/${ANDROID_TOOLCHAIN_MACHINE_NAME}/thumb/bits" )
 list( APPEND ANDROID_SYSTEM_INCLUDE_DIRS "${ANDROID_TOOLCHAIN_ROOT}/${ANDROID_TOOLCHAIN_MACHINE_NAME}/include/c++/${ANDROID_COMPILER_VERSION}/${ANDROID_TOOLCHAIN_MACHINE_NAME}/thumb" )
elseif( EXISTS "${ANDROID_TOOLCHAIN_ROOT}/${ANDROID_TOOLCHAIN_MACHINE_NAME}/include/c++/${ANDROID_COMPILER_VERSION}/${ANDROID_TOOLCHAIN_MACHINE_NAME}/bits" )
 list( APPEND ANDROID_SYSTEM_INCLUDE_DIRS "${ANDROID_TOOLCHAIN_ROOT}/${ANDROID_TOOLCHAIN_MACHINE_NAME}/include/c++/${ANDROID_COMPILER_VERSION}/${ANDROID_TOOLCHAIN_MACHINE_NAME}" )
endif()

#flags and definitions
if(ANDROID_SYSROOT MATCHES "[ ;\"]")
 set( ANDROID_CXX_FLAGS "--sysroot=\"${ANDROID_SYSROOT}\"" )
 # quotes will break try_compile and compiler identification
 message(WARNING "Your Android system root has non-alphanumeric symbols. It can break compiler features detection and the whole build.")
else()
 set( ANDROID_CXX_FLAGS "--sysroot=${ANDROID_SYSROOT}" )
endif()

remove_definitions( -DANDROID )
add_definitions( -DANDROID )

# Force set compilers because standard identification works badly for us
include( CMakeForceCompiler )
CMAKE_FORCE_C_COMPILER( "${CMAKE_C_COMPILER}" GNU )
CMAKE_FORCE_CXX_COMPILER( "${CMAKE_CXX_COMPILER}" GNU )
set( CMAKE_SIZEOF_VOID_P 4 )
set( CMAKE_C_SIZEOF_DATA_PTR 4 )
set( CMAKE_CXX_SIZEOF_DATA_PTR 4 )

# NDK flags
if( ARMEABI OR ARMEABI_V7A )
 # NDK also defines -ffunction-sections -funwind-tables but they result in worse OpenCV performance
 set( CMAKE_CXX_FLAGS "-fPIC -Wno-psabi" )
 set( CMAKE_C_FLAGS "-fPIC -Wno-psabi" )
 if( ANDROID_USE_STLPORT )
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti -fno-exceptions" )
  set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-rtti -fno-exceptions" )
 else()
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frtti -fexceptions" )
  set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions" )
 endif()
 remove_definitions( -D__ARM_ARCH_5__ -D__ARM_ARCH_5T__ -D__ARM_ARCH_5E__ -D__ARM_ARCH_5TE__ )
 add_definitions( -D__ARM_ARCH_5__ -D__ARM_ARCH_5T__ -D__ARM_ARCH_5E__ -D__ARM_ARCH_5TE__ )
 # extra arm-specific flags
 set( ANDROID_CXX_FLAGS "${ANDROID_CXX_FLAGS} -fsigned-char" )
elseif( X86 )
 set( CMAKE_CXX_FLAGS "-ffunction-sections -funwind-tables" )
 set( CMAKE_C_FLAGS "-ffunction-sections -funwind-tables" )
 if( ANDROID_USE_STLPORT )
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti -fno-exceptions" )
  set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-rtti -fno-exceptions" )
 else()
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frtti -fexceptions" )
  set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions" )
 endif()
else()
 set( CMAKE_CXX_FLAGS "" )
 set( CMAKE_C_FLAGS "" )
endif()

#release and debug flags
if( ARMEABI OR ARMEABI_V7A )
 if( NOT ANDROID_FORCE_ARM_BUILD AND NOT ARMEABI_V6 )
  #It is recommended to use the -mthumb compiler flag to force the generation
  #of 16-bit Thumb-1 instructions (the default being 32-bit ARM ones).
  # O3 instead of O2/Os in release mode - like cmake sets for desktop gcc
  set( CMAKE_CXX_FLAGS_RELEASE "-mthumb -O3" )
  set( CMAKE_C_FLAGS_RELEASE   "-mthumb -O3" )
  set( CMAKE_CXX_FLAGS_DEBUG "-marm -Os -finline-limit=64" )
  set( CMAKE_C_FLAGS_DEBUG   "-marm -Os -finline-limit=64" )
 else()
  # always compile ARMEABI_V6 in arm mode; otherwise there is no difference from ARMEABI
  # O3 instead of O2/Os in release mode - like cmake sets for desktop gcc
  set( CMAKE_CXX_FLAGS_RELEASE "-marm -O3 -fstrict-aliasing" )
  set( CMAKE_C_FLAGS_RELEASE   "-marm -O3 -fstrict-aliasing" )
  set( CMAKE_CXX_FLAGS_DEBUG "-marm -O0 -finline-limit=300" )
  set( CMAKE_C_FLAGS_DEBUG   "-marm -O0 -finline-limit=300" )
 endif()
elseif( X86 )
 set( CMAKE_CXX_FLAGS_RELEASE "-O3 -fstrict-aliasing" )
 set( CMAKE_C_FLAGS_RELEASE   "-O3 -fstrict-aliasing" )
 set( CMAKE_CXX_FLAGS_DEBUG "-O0 -finline-limit=300" )
 set( CMAKE_C_FLAGS_DEBUG   "-O0 -finline-limit=300" )
endif()
set( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fomit-frame-pointer -DNDEBUG" )
set( CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE}   -fomit-frame-pointer -DNDEBUG" )
set( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-strict-aliasing -fno-omit-frame-pointer -DDEBUG -D_DEBUG" )
set( CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG}   -fno-strict-aliasing -fno-omit-frame-pointer -DDEBUG -D_DEBUG" )

#ABI-specific flags
if( ARMEABI_V7A )
 set( ANDROID_CXX_FLAGS "${ANDROID_CXX_FLAGS} -march=armv7-a -mfloat-abi=softfp" )
 if( NEON )
  set( ANDROID_CXX_FLAGS "${ANDROID_CXX_FLAGS} -mfpu=neon" )
 elseif( VFPV3 )
  set( ANDROID_CXX_FLAGS "${ANDROID_CXX_FLAGS} -mfpu=vfpv3" )
 else()
  set( ANDROID_CXX_FLAGS "${ANDROID_CXX_FLAGS} -mfpu=vfp" )
 endif()
elseif( ARMEABI_V6 )
 set( ANDROID_CXX_FLAGS "${ANDROID_CXX_FLAGS} -march=armv6 -mfloat-abi=softfp -mfpu=vfp" )
elseif( ARMEABI )
 set( ANDROID_CXX_FLAGS "${ANDROID_CXX_FLAGS} -march=armv5te -mtune=xscale -msoft-float" )
elseif( X86 )
 set( ANDROID_CXX_FLAGS "${ANDROID_CXX_FLAGS}" )#sse?
endif()

#linker flags
list( APPEND ANDROID_SYSTEM_LIB_DIRS "${CMAKE_BINARY_DIR}/systemlibs/${ANDROID_NDK_ABI_NAME}" "${CMAKE_INSTALL_PREFIX}/libs/${ANDROID_NDK_ABI_NAME}" )
set( LINKER_FLAGS "" )
#STL
if( ANDROID_USE_STLPORT )
 if( EXISTS "${__stlLibPath}/libstlport_static.a" )
  __COPY_IF_DIFFERENT( "${__stlLibPath}/libstlport_static.a" "${CMAKE_BINARY_DIR}/systemlibs/${ANDROID_NDK_ABI_NAME}/libstlport_static.a" )
 endif()
 if( EXISTS "${CMAKE_BINARY_DIR}/systemlibs/${ANDROID_NDK_ABI_NAME}/libstlport_static.a" )
  set( LINKER_FLAGS "${LINKER_FLAGS} -Wl,--start-group -lstlport_static" )
 endif()
else( ANDROID_USE_STLPORT )
 if( EXISTS "${__stlLibPath}/libgnustl_static.a" )
  __COPY_IF_DIFFERENT( "${__stlLibPath}/libgnustl_static.a" "${CMAKE_BINARY_DIR}/systemlibs/${ANDROID_NDK_ABI_NAME}/libstdc++.a" )
 elseif( ANDROID_ARCH_NAME STREQUAL "arm" AND EXISTS "${__stlLibPath}/${CMAKE_SYSTEM_PROCESSOR}/thumb/libstdc++.a" )
  __COPY_IF_DIFFERENT( "${__stlLibPath}/${CMAKE_SYSTEM_PROCESSOR}/thumb/libstdc++.a" "${CMAKE_BINARY_DIR}/systemlibs/${ANDROID_NDK_ABI_NAME}/libstdc++.a" )
 elseif( ANDROID_ARCH_NAME STREQUAL "arm" AND EXISTS "${__stlLibPath}/${CMAKE_SYSTEM_PROCESSOR}/libstdc++.a" )
  __COPY_IF_DIFFERENT( "${__stlLibPath}/${CMAKE_SYSTEM_PROCESSOR}/libstdc++.a" "${CMAKE_BINARY_DIR}/systemlibs/${ANDROID_NDK_ABI_NAME}/libstdc++.a" )
 elseif( ANDROID_ARCH_NAME STREQUAL "arm" AND EXISTS "${__stlLibPath}/thumb/libstdc++.a" )
  __COPY_IF_DIFFERENT( "${__stlLibPath}/thumb/libstdc++.a" "${CMAKE_BINARY_DIR}/systemlibs/${ANDROID_NDK_ABI_NAME}/libstdc++.a" )
 elseif( EXISTS "${__stlLibPath}/libstdc++.a" )
  __COPY_IF_DIFFERENT( "${__stlLibPath}/libstdc++.a" "${CMAKE_BINARY_DIR}/systemlibs/${ANDROID_NDK_ABI_NAME}/libstdc++.a" )
 endif()
 if( EXISTS "${CMAKE_BINARY_DIR}/systemlibs/${ANDROID_NDK_ABI_NAME}/libstdc++.a" )
  set( LINKER_FLAGS "${LINKER_FLAGS} -lstdc++" )
 endif()

 #gcc exception & rtti support
 if( EXISTS "${__stlLibPath}/libsupc++.a" )
  __COPY_IF_DIFFERENT( "${__stlLibPath}/libsupc++.a" "${CMAKE_BINARY_DIR}/systemlibs/${ANDROID_NDK_ABI_NAME}/libsupc++.a" )
 elseif( ANDROID_ARCH_NAME STREQUAL "arm" AND EXISTS "${ANDROID_TOOLCHAIN_ROOT}/${ANDROID_TOOLCHAIN_MACHINE_NAME}/lib/${CMAKE_SYSTEM_PROCESSOR}/thumb/libsupc++.a" )
  __COPY_IF_DIFFERENT( "${ANDROID_TOOLCHAIN_ROOT}/${ANDROID_TOOLCHAIN_MACHINE_NAME}/lib/${CMAKE_SYSTEM_PROCESSOR}/thumb/libsupc++.a" "${CMAKE_BINARY_DIR}/systemlibs/${ANDROID_NDK_ABI_NAME}/libsupc++.a" )
 elseif( ANDROID_ARCH_NAME STREQUAL "arm" AND EXISTS "${ANDROID_TOOLCHAIN_ROOT}/${ANDROID_TOOLCHAIN_MACHINE_NAME}/lib/${CMAKE_SYSTEM_PROCESSOR}/libsupc++.a" )
  __COPY_IF_DIFFERENT( "${ANDROID_TOOLCHAIN_ROOT}/${ANDROID_TOOLCHAIN_MACHINE_NAME}/lib/${CMAKE_SYSTEM_PROCESSOR}/libsupc++.a" "${CMAKE_BINARY_DIR}/systemlibs/${ANDROID_NDK_ABI_NAME}/libsupc++.a" )
 elseif( ANDROID_ARCH_NAME STREQUAL "arm" AND EXISTS "${ANDROID_TOOLCHAIN_ROOT}/${ANDROID_TOOLCHAIN_MACHINE_NAME}/lib/thumb/libsupc++.a" )
  __COPY_IF_DIFFERENT( "${ANDROID_TOOLCHAIN_ROOT}/${ANDROID_TOOLCHAIN_MACHINE_NAME}/lib/thumb/libsupc++.a" "${CMAKE_BINARY_DIR}/systemlibs/${ANDROID_NDK_ABI_NAME}/libsupc++.a" )
 elseif( EXISTS "${ANDROID_TOOLCHAIN_ROOT}/${ANDROID_TOOLCHAIN_MACHINE_NAME}/lib/libsupc++.a" )
  __COPY_IF_DIFFERENT( "${ANDROID_TOOLCHAIN_ROOT}/${ANDROID_TOOLCHAIN_MACHINE_NAME}/lib/libsupc++.a" "${CMAKE_BINARY_DIR}/systemlibs/${ANDROID_NDK_ABI_NAME}/libsupc++.a" )
 endif()
 if( EXISTS "${CMAKE_BINARY_DIR}/systemlibs/${ANDROID_NDK_ABI_NAME}/libsupc++.a" )
  set( LINKER_FLAGS "${LINKER_FLAGS} -lsupc++" )
 endif()
endif( ANDROID_USE_STLPORT )

#cleanup for STL search
unset( __stlIncludePath )
unset( __stlLibPath )

#other linker flags
__INIT_VARIABLE( ANDROID_NO_UNDEFINED OBSOLETE_NO_UNDEFINED VALUES ON )
set( ANDROID_NO_UNDEFINED ${ANDROID_NO_UNDEFINED} CACHE BOOL "Show all undefined symbols as linker errors" FORCE )
mark_as_advanced( ANDROID_NO_UNDEFINED )
if( ANDROID_NO_UNDEFINED )
 set( LINKER_FLAGS "-Wl,--no-undefined ${LINKER_FLAGS}" )
endif()

if (ANDROID_NDK MATCHES "-r[56].?$")
 #libGLESv2.so in NDK's prior to r7 refers to exteranal symbols. So this flag option is required for all projects using OpenGL from native.
 __INIT_VARIABLE( ANDROID_SO_UNDEFINED VALUES ON )
else()
 __INIT_VARIABLE( ANDROID_SO_UNDEFINED VALUES OFF )
endif()
set( ANDROID_SO_UNDEFINED ${ANDROID_SO_UNDEFINED} CACHE BOOL "Allows or disallows undefined symbols in shared libraries" FORCE )
mark_as_advanced( ANDROID_SO_UNDEFINED )
if( ANDROID_SO_UNDEFINED )
 set( LINKER_FLAGS "${LINKER_FLAGS} -Wl,-allow-shlib-undefined" )
endif()

if( ARMEABI_V7A )
 # this is *required* to use the following linker flags that routes around
 # a CPU bug in some Cortex-A8 implementations:
 set( LINKER_FLAGS "-Wl,--fix-cortex-a8 ${LINKER_FLAGS}" )
endif()

#cache flags
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "c++ flags" )
set( CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "c flags" )
set( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}" CACHE STRING "c++ Release flags" )
set( CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}" CACHE STRING "c Release flags" )
set( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}" CACHE STRING "c++ Debug flags" )
set( CMAKE_C_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}" CACHE STRING "c Debug flags" )
set( CMAKE_SHARED_LINKER_FLAGS "${LINKER_FLAGS}" CACHE STRING "linker flags" )
set( CMAKE_MODULE_LINKER_FLAGS "${LINKER_FLAGS}" CACHE STRING "linker flags" )
set( CMAKE_EXE_LINKER_FLAGS "-Wl,--gc-sections -Wl,-z,nocopyreloc ${LINKER_FLAGS}" CACHE STRING "linker flags" )

include_directories( SYSTEM ${ANDROID_SYSTEM_INCLUDE_DIRS} )
link_directories( ${ANDROID_SYSTEM_LIB_DIRS} )

#finish flags
set( ANDROID_CXX_FLAGS "${ANDROID_CXX_FLAGS}" CACHE INTERNAL "Extra Android falgs")
set( CMAKE_CXX_FLAGS "${ANDROID_CXX_FLAGS} ${CMAKE_CXX_FLAGS}" )
set( CMAKE_C_FLAGS   "${ANDROID_CXX_FLAGS} ${CMAKE_C_FLAGS}" )

#set these global flags for cmake client scripts to change behavior
set( ANDROID True )
set( BUILD_ANDROID True )

# where is the target environment 
set( CMAKE_FIND_ROOT_PATH "${ANDROID_TOOLCHAIN_ROOT}/bin" "${ANDROID_TOOLCHAIN_ROOT}/${ANDROID_TOOLCHAIN_MACHINE_NAME}" "${ANDROID_SYSROOT}" "${CMAKE_INSTALL_PREFIX}" "${CMAKE_INSTALL_PREFIX}/share" )

# only search for libraries and includes in the ndk toolchain
set( CMAKE_FIND_ROOT_PATH_MODE_PROGRAM ONLY )
set( CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY )
set( CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY )

#macro to find packages on the host OS
macro( find_host_package )
 set( CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER )
 set( CMAKE_FIND_ROOT_PATH_MODE_LIBRARY NEVER )
 set( CMAKE_FIND_ROOT_PATH_MODE_INCLUDE NEVER )
 if( CMAKE_HOST_WIN32 )
  SET( WIN32 1 )
  SET( UNIX )
 elseif( CMAKE_HOST_APPLE )
  SET( APPLE 1 )
  SET( UNIX )
 endif()
 find_package( ${ARGN} )
 SET( WIN32 )
 SET( APPLE )
 SET( UNIX 1 )
 set( CMAKE_FIND_ROOT_PATH_MODE_PROGRAM ONLY )
 set( CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY )
 set( CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY )
endmacro()
#macro to find programs on the host OS
macro( find_host_program )
 set( CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER )
 set( CMAKE_FIND_ROOT_PATH_MODE_LIBRARY NEVER )
 set( CMAKE_FIND_ROOT_PATH_MODE_INCLUDE NEVER )
 if( CMAKE_HOST_WIN32 )
  SET( WIN32 1 )
  SET( UNIX )
 elseif( CMAKE_HOST_APPLE )
  SET( APPLE 1 )
  SET( UNIX )
 endif()
 find_program( ${ARGN} )
 SET( WIN32 )
 SET( APPLE )
 SET( UNIX 1 )
 set( CMAKE_FIND_ROOT_PATH_MODE_PROGRAM ONLY )
 set( CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY )
 set( CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY )
endmacro()

macro( ANDROID_GET_ABI_RAWNAME TOOLCHAIN_FLAG VAR )
 if( "${TOOLCHAIN_FLAG}" STREQUAL "ARMEABI" )
  set( ${VAR} "armeabi" )
 elseif( "${TOOLCHAIN_FLAG}" STREQUAL "ARMEABI_V7A" )
    set( ${VAR} "armeabi-v7a" )
 elseif( "${TOOLCHAIN_FLAG}" STREQUAL "X86" )
    set( ${VAR} "x86" )
 else()
    set( ${VAR} "unknown" )
 endif()
endmacro()

set( ANDROID_SET_OBSOLETE_VARIABLES ON CACHE BOOL "Define obsolete Andrid-specific cmake variables" )
mark_as_advanced( ANDROID_SET_OBSOLETE_VARIABLES )
if( ANDROID_SET_OBSOLETE_VARIABLES )
 set( ANDROID_API_LEVEL ${ANDROID_NATIVE_API_LEVEL} )
 set( ARM_TARGET "${ANDROID_ABI}" )
 set( ARMEABI_NDK_NAME "${ANDROID_NDK_ABI_NAME}" )
endif()

# Variables controlling behavior or set by cmake toolchain:
#   ANDROID_ABI : "armeabi-v7a" (default), "armeabi", "armeabi-v7a with NEON", "armeabi-v7a with VFPV3", "armeabi-v6 with VFP", "x86"
#   ANDROID_FORCE_ARM_BUILD : ON/OFF
#   ANDROID_NATIVE_API_LEVEL : 3,4,5,8,9,14 (depends on NDK version)
#   ANDROID_NO_UNDEFINED : ON/OFF
#   ANDROID_SO_UNDEFINED : OFF/ON  (default depends on NDK version)
#   ANDROID_SET_OBSOLETE_VARIABLES : ON/OFF
#   LIBRARY_OUTPUT_PATH_ROOT : <any path>
#   ANDROID_USE_STLPORT : OFF/ON - EXPERIMENTAL!!! 
#   ANDROID_FORBID_SYGWIN : ON/OFF
# Can be set only at the first run:
#   ANDROID_NDK
#   ANDROID_STANDALONE_TOOLCHAIN
# Obsolete:
#   ANDROID_API_LEVEL : superseded by ANDROID_NATIVE_API_LEVEL
#   ARM_TARGET : superseded by ANDROID_ABI
#   ARM_TARGETS : superseded by ANDROID_ABI (can be set only)
#   ANDROID_NDK_TOOLCHAIN_ROOT : superseded by ANDROID_STANDALONE_TOOLCHAIN (can be set only)
#   ANDROID_LEVEL : superseded by ANDROID_NATIVE_API_LEVEL (completely removed)
#
# Primary read-only variables:
#   ANDROID : always TRUE
#   ARMEABI : TRUE for arm v6 and older devices
#   ARMEABI_V6 : TRUE for arm v6
#   ARMEABI_V7A : TRUE for arm v7a
#   NEON : TRUE if NEON unit is enabled
#   VFPV3 : TRUE if VFP version 3 is enabled
#   X86 : TRUE if configured for x86
#   BUILD_ANDROID : always TRUE
#   BUILD_WITH_ANDROID_NDK : TRUE if NDK is used
#   BUILD_WITH_STANDALONE_TOOLCHAIN : TRUE if standalone toolchain is used
#   ANDROID_NDK_HOST_SYSTEM_NAME : "windows", "linux-x86" or "darwin-x86" depending on host platform
#   ANDROID_NDK_ABI_NAME : "armeabi", "armeabi-v7a" or "x86" depending on ANDROID_ABI
#   ANDROID_ARCH_NAME : "arm" or "x86" depending on ANDROID_ABI
#   TOOL_OS_SUFFIX : "" or ".exe" depending on host platform
#   ANDROID_SYSTEM_INCLUDE_DIRS
#   ANDROID_SYSTEM_LIB_DIRS
# Obsolete:
#   ARMEABI_NDK_NAME : superseded by ANDROID_NDK_ABI_NAME
#
# Secondary (less stable) read-only variables:
#   ANDROID_COMPILER_VERSION : GCC version used
#   ANDROID_CXX_FLAGS : C/C++ compiler flags required by Android platform
#   ANDROID_SUPPORTED_ABIS : list of currently allowed values for ANDROID_ABI
#   ANDROID_SYSROOT : path to the compiler sysroot
#   ANDROID_TOOLCHAIN_NAME : "standalone", "arm-linux-androideabi-4.4.3" or "x86-4.4.3" or something similar.
#   ANDROID_TOOLCHAIN_MACHINE_NAME : "arm-linux-androideabi", "arm-eabi" or "i686-android-linux"
#   ANDROID_TOOLCHAIN_ROOT : path to the top level of toolchain (standalone or placed inside NDK)
#   ANDROID_SUPPORTED_NATIVE_API_LEVELS : list of native API levels found inside NDK
#
# Defaults:
#   ANDROID_DEFAULT_NDK_API_LEVEL
#   ANDROID_DEFAULT_NDK_API_LEVEL_${ARCH}
#   ANDROID_NDK_SEARCH_PATHS
#   ANDROID_STANDALONE_TOOLCHAIN_SEARCH_PATH
#   ANDROID_SUPPORTED_ABIS_${ARCH}
#   ANDROID_SUPPORTED_NDK_VERSIONS

