# ----------------------------------------------------------------------------
#  Android CMake toolchain file, for use with the ndk r5
#  See home page: http://code.google.com/p/android-cmake/
#
#  Usage Linux:
#   $ export ANDROID_NDK=/<absolute path to NDK>
#   $ cmake -DCMAKE_TOOLCHAIN_FILE=<path to this file>/android.toolchain.cmake ..
#   $ make
#
#  Usage Linux (using standalone toolchain):
#   $ export ANDROID_NDK_TOOLCHAIN_ROOT=/<absolute path to standalone toolchain>
#   $ cmake -DCMAKE_TOOLCHAIN_FILE=<path to this file>/android.toolchain.cmake ..
#   $ make
#
#  Usage Windows:
#     You need native port of make to build your project.
#     For example this one: http://gnuwin32.sourceforge.net/packages/make.htm
#
#   $ SET ANDROID_NDK=C:\<absolute path to NDK>\android-ndk-r5c
#   $ cmake.exe -G"Unix Makefiles" -DCMAKE_TOOLCHAIN_FILE=<path to this file>\android.toolchain.cmake -DCMAKE_MAKE_PROGRAM=C:\<absolute path to make>\make.exe ..
#   $ C:\<absolute path to make>\make.exe
#
#
#  Toolchain options (can be set as cmake parameters: -D<option_name>=<value>):
#    ANDROID_NDK=/opt/android-ndk-r5c - path to NDK root.
#      Can be set as environment variable.
#
#    ANDROID_NDK_TOOLCHAIN_ROOT=/opt/android-toolchain - path to standalone toolchain.
#      Option is not used if full NDK is found. Can be set as environment variable.
#
#    ANDROID_API_LEVEL=android-8 - level of android API to use.
#      Option is ignored when build uses stanalone toolchain.
#
#    ARM_TARGET=armeabi-v7a - type of floating point support.
#      Other possible values are: "armeabi", "armeabi-v7a with NEON", "armeabi-v7a with VFPV3"
#
#    FORCE_ARM=false - set true to generate 32-bit ARM instructions instead of Thumb-1.
#
#    NO_UNDEFINED=true - set true to show all undefined symbols will as linker errors even if they are not used.
#
#
#  Toolcahin will search for NDK/toolchain in following order:
#    ANDROID_NDK - cmake parameter
#    ANDROID_NDK - environment variable
#    ANDROID_NDK - default location
#    ANDROID_NDK_TOOLCHAIN_ROOT - cmake parameter
#    ANDROID_NDK_TOOLCHAIN_ROOT - environment variable
#    ANDROID_NDK_TOOLCHAIN_ROOT - default location
#
#
#  What?:
#     Make sure to do the following in your scripts:
#       SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${my_cxx_flags}")
#       SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  ${my_cxx_flags}")
#       The flags will be prepopulated with critical flags, so don't loose them.
#    
#     ANDROID and BUILD_ANDROID will be set to true, you may test these 
#     variables to make necessary changes.
#    
#     Also ARMEABI and ARMEABI_V7A will be set true, mutually exclusive. V7A is
#     for floating point. NEON option will be set true if fpu is set to neon.
#
#     LIBRARY_OUTPUT_PATH_ROOT should be set in cache to determine where android
#     libraries will be installed.
#        default is ${CMAKE_SOURCE_DIR} , and the android libs will always be
#        under ${LIBRARY_OUTPUT_PATH_ROOT}/libs/armeabi* depending on target.
#        this will be convenient for android linking
#
#     Base system is Linux, but you may need to change things 
#     for android compatibility.
#   
#
#   - initial version December 2010 Ethan Rublee ethan.ruble@gmail.com
#   - modified April 2011 Andrey Kamaev andrey.kamaev@itseez.com
#     [+] added possibility to build with NDK (without standalone toolchain)
#     [+] support croos compilation on Windows (native, no cygwin support)
#     [+] added compiler option to force "char" type to be signed
#     [+] added toolchain option to compile to 32-bit ARM instructions
#     [+] added toolchain option to disable SWIG search
#     [+] added platform "armeabi-v7a with VFPV3"
#     [~] ARM_TARGETS renamed to ARM_TARGET
#   - modified April 2011 Andrey Kamaev andrey.kamaev@itseez.com
#     [+] EXECUTABLE_OUTPUT_PATH is set by toolchain (required on Windows)
#     [~] Fixed bug with ANDROID_API_LEVEL variable
#     [~] turn off SWIG search if it is not found first time
#   - modified May 2011 Andrey Kamaev andrey.kamaev@itseez.com
#     [~] ANDROID_LEVEL is renamed to ANDROID_API_LEVEL
#     [+] ANDROID_API_LEVEL is detected by toolchain if not specified
#     [~] added guard to prevent changing of output directories on first cmake pass
#     [~] toolchain exits with error if ARM_TARGET is not recognized
#   - modified June 2011 Andrey Kamaev andrey.kamaev@itseez.com
#     [~] default NDK path is updated for version r5c 
#     [+] variable CMAKE_SYSTEM_PROCESSOR is set based on ARM_TARGET
#     [~] toolchain install directory is added to linker paths
#     [-] removed SWIG-related stuff from toolchain
#     [+] added macro find_host_package, find_host_program to search packages/programs on host system
#     [~] fixed path to STL library
# ----------------------------------------------------------------------------



# this one is important
set( CMAKE_SYSTEM_NAME Linux )
#this one not so much
set( CMAKE_SYSTEM_VERSION 1 )

set( ANDROID_NDK_DEFAULT_SEARCH_PATH /opt/android-ndk-r5c )
set( ANDROID_NDK_TOOLCHAIN_DEFAULT_SEARCH_PATH /opt/android-toolchain )
set( TOOL_OS_SUFFIX "" )

macro( __TOOLCHAIN_DETECT_API_LEVEL _path )
 SET( _expected ${ARGV1} )
 if( NOT EXISTS ${_path} )
  message( FATAL_ERROR "Could not verify Android API level. Probably you have specified invalid level value, or your copy of NDK/toolchain is broken." )
 endif()
 SET( API_LEVEL_REGEX "^[\t ]*#define[\t ]+__ANDROID_API__[\t ]+([0-9]+)[\t ]*$" )
 FILE( STRINGS ${_path} API_FILE_CONTENT REGEX "${API_LEVEL_REGEX}")
 if( NOT API_FILE_CONTENT )
  message( FATAL_ERROR "Could not verify Android API level. Probably you have specified invalid level value, or your copy of NDK/toolchain is broken." )
 endif()
 string( REGEX REPLACE "${API_LEVEL_REGEX}" "\\1" ANDROID_LEVEL_FOUND "${API_FILE_CONTENT}" )
 if( DEFINED _expected )
  if( NOT ${ANDROID_LEVEL_FOUND} EQUAL ${_expected} )
   message( FATAL_ERROR "Specified Android API level does not match level found. Probably your copy of NDK/toolchain is broken." )
  endif()
 endif()
 set( ANDROID_API_LEVEL ${ANDROID_LEVEL_FOUND} CACHE STRING "android API level" FORCE )
endmacro()

#set path for android NDK -- look
if( NOT DEFINED ANDROID_NDK )
 set( ANDROID_NDK $ENV{ANDROID_NDK} )
endif()

if( NOT EXISTS ${ANDROID_NDK} )
 if( EXISTS ${ANDROID_NDK_DEFAULT_SEARCH_PATH} )
  set ( ANDROID_NDK ${ANDROID_NDK_DEFAULT_SEARCH_PATH} )
  message( STATUS "Using default path for android NDK ${ANDROID_NDK}" )
  message( STATUS "If you prefer to use a different location, please define the variable: ANDROID_NDK" )
 endif()
endif()

if( EXISTS ${ANDROID_NDK} )
 set( ANDROID_NDK ${ANDROID_NDK} CACHE PATH "root of the android ndk" FORCE )
 
 if( APPLE )
  set( NDKSYSTEM "darwin-x86" )
 elseif( WIN32 )
  set( NDKSYSTEM "windows" )
  set( TOOL_OS_SUFFIX ".exe" )
 elseif( UNIX )
  set( NDKSYSTEM "linux-x86" )
 else()
  message( FATAL_ERROR "Your platform is not supported" )
 endif()

 set( ANDROID_API_LEVEL $ENV{ANDROID_API_LEVEL} )
 string( REGEX REPLACE "[\t ]*android-([0-9]+)[\t ]*" "\\1" ANDROID_API_LEVEL "${ANDROID_API_LEVEL}" )
 string( REGEX REPLACE "[\t ]*([0-9]+)[\t ]*" "\\1" ANDROID_API_LEVEL "${ANDROID_API_LEVEL}" )

 set( PossibleAndroidLevels "3;4;5;8;9" )
 set( ANDROID_API_LEVEL ${ANDROID_API_LEVEL} CACHE STRING "android API level" )
 set_property( CACHE ANDROID_API_LEVEL PROPERTY STRINGS ${PossibleAndroidLevels} )
 
 if( NOT ANDROID_API_LEVEL GREATER 2 )
  set( ANDROID_API_LEVEL 8)
  message( STATUS "Using default android API level android-${ANDROID_API_LEVEL}" )
  message( STATUS "If you prefer to use a different API level, please define the variable: ANDROID_API_LEVEL" )
 endif()

 set( ANDROID_NDK_TOOLCHAIN_ROOT "${ANDROID_NDK}/toolchains/arm-linux-androideabi-4.4.3/prebuilt/${NDKSYSTEM}" )
 set( ANDROID_NDK_SYSROOT "${ANDROID_NDK}/platforms/android-${ANDROID_API_LEVEL}/arch-arm" )
 
 __TOOLCHAIN_DETECT_API_LEVEL( "${ANDROID_NDK_SYSROOT}/usr/include/android/api-level.h" ${ANDROID_API_LEVEL} )
 
 #message( STATUS "Using android NDK from ${ANDROID_NDK}" )
 set( BUILD_WITH_ANDROID_NDK True )
else()
 #try to find toolchain
 if( NOT DEFINED ANDROID_NDK_TOOLCHAIN_ROOT )
  set( ANDROID_NDK_TOOLCHAIN_ROOT $ENV{ANDROID_NDK_TOOLCHAIN_ROOT} )
 endif()
 
 if( NOT EXISTS ${ANDROID_NDK_TOOLCHAIN_ROOT} )
  set( ANDROID_NDK_TOOLCHAIN_ROOT ${ANDROID_NDK_TOOLCHAIN_DEFAULT_SEARCH_PATH} )
  message( STATUS "Using default path for toolchain ${ANDROID_NDK_TOOLCHAIN_ROOT}" )
  message( STATUS "If you prefer to use a different location, please define the variable: ANDROID_NDK_TOOLCHAIN_ROOT" )
 endif()

 set( ANDROID_NDK_TOOLCHAIN_ROOT ${ANDROID_NDK_TOOLCHAIN_ROOT} CACHE PATH "root of the Android NDK standalone toolchain" FORCE )
 set( ANDROID_NDK_SYSROOT "${ANDROID_NDK_TOOLCHAIN_ROOT}/sysroot" )

 if( NOT EXISTS ${ANDROID_NDK_TOOLCHAIN_ROOT} )
  message( FATAL_ERROR "neither ${ANDROID_NDK} nor ${ANDROID_NDK_TOOLCHAIN_ROOT} does not exist!
    You should either set an environment variable:
      export ANDROID_NDK=~/my-android-ndk
    or
      export ANDROID_NDK_TOOLCHAIN_ROOT=~/my-android-toolchain
    or put the toolchain or NDK in the default path:
      sudo ln -s ~/my-android-ndk ${ANDROID_NDK_DEFAULT_SEARCH_PATH}
      sudo ln -s ~/my-android-toolchain ${ANDROID_NDK_TOOLCHAIN_DEFAULT_SEARCH_PATH}" )
 endif()
 
 __TOOLCHAIN_DETECT_API_LEVEL( "${ANDROID_NDK_SYSROOT}/usr/include/android/api-level.h" )

 #message( STATUS "Using android NDK standalone toolchain from ${ANDROID_NDK_TOOLCHAIN_ROOT}" )
 set( BUILD_WITH_ANDROID_NDK_TOOLCHAIN True )
endif()

# specify the cross compiler
set( CMAKE_C_COMPILER   ${ANDROID_NDK_TOOLCHAIN_ROOT}/bin/arm-linux-androideabi-gcc${TOOL_OS_SUFFIX}     CACHE PATH "gcc" FORCE )
set( CMAKE_CXX_COMPILER ${ANDROID_NDK_TOOLCHAIN_ROOT}/bin/arm-linux-androideabi-g++${TOOL_OS_SUFFIX}     CACHE PATH "g++" FORCE )
#there may be a way to make cmake deduce these TODO deduce the rest of the tools
set( CMAKE_AR           ${ANDROID_NDK_TOOLCHAIN_ROOT}/bin/arm-linux-androideabi-ar${TOOL_OS_SUFFIX}      CACHE PATH "archive" FORCE )
set( CMAKE_LINKER       ${ANDROID_NDK_TOOLCHAIN_ROOT}/bin/arm-linux-androideabi-ld${TOOL_OS_SUFFIX}      CACHE PATH "linker" FORCE )
set( CMAKE_NM           ${ANDROID_NDK_TOOLCHAIN_ROOT}/bin/arm-linux-androideabi-nm${TOOL_OS_SUFFIX}      CACHE PATH "nm" FORCE )
set( CMAKE_OBJCOPY      ${ANDROID_NDK_TOOLCHAIN_ROOT}/bin/arm-linux-androideabi-objcopy${TOOL_OS_SUFFIX} CACHE PATH "objcopy" FORCE )
set( CMAKE_OBJDUMP      ${ANDROID_NDK_TOOLCHAIN_ROOT}/bin/arm-linux-androideabi-objdump${TOOL_OS_SUFFIX} CACHE PATH "objdump" FORCE )
set( CMAKE_STRIP        ${ANDROID_NDK_TOOLCHAIN_ROOT}/bin/arm-linux-androideabi-strip${TOOL_OS_SUFFIX}   CACHE PATH "strip" FORCE )
set( CMAKE_RANLIB       ${ANDROID_NDK_TOOLCHAIN_ROOT}/bin/arm-linux-androideabi-ranlib${TOOL_OS_SUFFIX}  CACHE PATH "ranlib" FORCE )

#setup build targets, mutually exclusive
set( PossibleArmTargets "armeabi;armeabi-v7a;armeabi-v7a with NEON;armeabi-v7a with VFPV3" )
set( ARM_TARGET "armeabi-v7a" CACHE STRING "the arm target for android, recommend armeabi-v7a for floating point support and NEON." )
set_property( CACHE ARM_TARGET PROPERTY STRINGS ${PossibleArmTargets} )

#compatibility junk for previous version of toolchain
if( DEFINED ARM_TARGETS AND NOT DEFINED ARM_TARGET )
 SET( ARM_TARGET "${ARM_TARGETS}" )
endif()

#set these flags for client use
if( ARM_TARGET STREQUAL "armeabi" )
 set( ARMEABI true )
 set( ARMEABI_NDK_NAME "armeabi" CACHE STRING "NDK eabi name" FORCE)
 set( NEON false )
 set( CMAKE_SYSTEM_PROCESSOR "armv5te" )
else()
 if( ARM_TARGET STREQUAL "armeabi-v7a with NEON" )
  set( NEON true )
  set( VFPV3 true )
 elseif( ARM_TARGET STREQUAL "armeabi-v7a with VFPV3" )
  set( VFPV3 true )
 elseif( NOT ARM_TARGET STREQUAL "armeabi-v7a")
  message( FATAL_ERROR "Unsupported ARM_TARGET=${ARM_TARGET} is specified.
Supported values are: \"armeabi\", \"armeabi-v7a\", \"armeabi-v7a with NEON\", \"armeabi-v7a with VFPV3\"
" )
 endif()
 set( ARMEABI_V7A true )
 set( ARMEABI_NDK_NAME "armeabi-v7a" CACHE STRING "NDK eabi name" FORCE)
 set( CMAKE_SYSTEM_PROCESSOR "armv7-a" )
endif()

#setup output directories
set( LIBRARY_OUTPUT_PATH_ROOT ${CMAKE_SOURCE_DIR} CACHE PATH "root for library output, set this to change where android libs are installed to" )

SET( DO_NOT_CHANGE_OUTPUT_PATHS_ON_FIRST_PASS OFF CACHE BOOL "")
if( DO_NOT_CHANGE_OUTPUT_PATHS_ON_FIRST_PASS )
 if( EXISTS ${CMAKE_SOURCE_DIR}/jni/CMakeLists.txt )
  set( EXECUTABLE_OUTPUT_PATH ${LIBRARY_OUTPUT_PATH_ROOT}/bin/${ARMEABI_NDK_NAME} CACHE PATH "Output directory for applications")
 else()
  set( EXECUTABLE_OUTPUT_PATH ${LIBRARY_OUTPUT_PATH_ROOT}/bin CACHE PATH "Output directory for applications")
 endif()
 set( LIBRARY_OUTPUT_PATH ${LIBRARY_OUTPUT_PATH_ROOT}/libs/${ARMEABI_NDK_NAME} CACHE PATH "path for android libs")
 set( CMAKE_INSTALL_PREFIX ${ANDROID_NDK_TOOLCHAIN_ROOT}/user CACHE STRING "path for installing" )
endif()
SET( DO_NOT_CHANGE_OUTPUT_PATHS_ON_FIRST_PASS ON CACHE INTERNAL "" FORCE)

# where is the target environment 
set( CMAKE_FIND_ROOT_PATH ${ANDROID_NDK_TOOLCHAIN_ROOT}/bin ${ANDROID_NDK_TOOLCHAIN_ROOT}/arm-linux-androideabi ${ANDROID_NDK_SYSROOT} ${CMAKE_INSTALL_PREFIX} ${CMAKE_INSTALL_PREFIX}/share )

if( BUILD_WITH_ANDROID_NDK )
 set( STL_PATH "${ANDROID_NDK}/sources/cxx-stl/gnu-libstdc++" )
 set( STL_LIBRARIES_PATH "${STL_PATH}/libs/${ARMEABI_NDK_NAME}" )
 include_directories( "${STL_PATH}/include" "${STL_LIBRARIES_PATH}/include" )
# if ( NOT ARMEABI AND NOT FORCE_ARM )
#  set( STL_LIBRARIES_PATH "${ANDROID_NDK_TOOLCHAIN_ROOT}/arm-linux-androideabi/lib/${CMAKE_SYSTEM_PROCESSOR}/thumb" )
# endif()
endif()

if( BUILD_WITH_ANDROID_NDK_TOOLCHAIN )
 set( STL_LIBRARIES_PATH "${ANDROID_NDK_TOOLCHAIN_ROOT}/arm-linux-androideabi/lib" )
 if( NOT ARMEABI )
  set( STL_LIBRARIES_PATH "${STL_LIBRARIES_PATH}/${CMAKE_SYSTEM_PROCESSOR}" )
 endif()
 if( NOT FORCE_ARM )
  set( STL_LIBRARIES_PATH "${STL_LIBRARIES_PATH}/thumb" )
 endif()
 #for some reason this is needed? TODO figure out why...
 include_directories( ${ANDROID_NDK_TOOLCHAIN_ROOT}/arm-linux-androideabi/include/c++/4.4.3/arm-linux-androideabi )
endif()

# only search for libraries and includes in the ndk toolchain
set( CMAKE_FIND_ROOT_PATH_MODE_PROGRAM ONLY )
set( CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY )
set( CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY )

set( CMAKE_CXX_FLAGS "-fPIC -DANDROID -Wno-psabi -fsigned-char" )
set( CMAKE_C_FLAGS "-fPIC -DANDROID -Wno-psabi -fsigned-char" )

set( FORCE_ARM OFF CACHE BOOL "Use 32-bit ARM instructions instead of Thumb-1" )
if( NOT FORCE_ARM )
 #It is recommended to use the -mthumb compiler flag to force the generation
 #of 16-bit Thumb-1 instructions (the default being 32-bit ARM ones).
 set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mthumb" )
 set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mthumb" )
else()
 set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -marm" )
 set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -marm" )
endif()

if( BUILD_WITH_ANDROID_NDK )
 set( CMAKE_CXX_FLAGS "--sysroot=${ANDROID_NDK_SYSROOT} ${CMAKE_CXX_FLAGS}" )
 set( CMAKE_C_FLAGS "--sysroot=${ANDROID_NDK_SYSROOT} ${CMAKE_C_FLAGS}" )
endif()

if( ARMEABI_V7A )
 #these are required flags for android armv7-a
 set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv7-a -mfloat-abi=softfp" )
 set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv7-a -mfloat-abi=softfp" )
 if( NEON )
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon" )
  set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mfpu=neon" )
 elseif( VFPV3 )
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=vfpv3" )
  set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mfpu=vfpv3" )
 endif()
endif()

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "c++ flags" )
set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS}" CACHE STRING "c flags" )
      
#-Wl,-L${LIBCPP_LINK_DIR},-lstdc++,-lsupc++
#-L${LIBCPP_LINK_DIR} -lstdc++ -lsupc++
#Also, this is *required* to use the following linker flags that routes around
#a CPU bug in some Cortex-A8 implementations:
set( LINKER_FLAGS "-Wl,--fix-cortex-a8 -L${STL_LIBRARIES_PATH} -L${CMAKE_INSTALL_PREFIX}/libs/${ARMEABI_NDK_NAME} -lstdc++ -lsupc++ " )

set( NO_UNDEFINED ON CACHE BOOL "Don't all undefined symbols" )
if( NO_UNDEFINED )
 set( LINKER_FLAGS "-Wl,--no-undefined ${LINKER_FLAGS}" )
endif()

set( CMAKE_SHARED_LINKER_FLAGS "${LINKER_FLAGS}" CACHE STRING "linker flags" FORCE )
set( CMAKE_MODULE_LINKER_FLAGS "${LINKER_FLAGS}" CACHE STRING "linker flags" FORCE )
set( CMAKE_EXE_LINKER_FLAGS "${LINKER_FLAGS}" CACHE STRING "linker flags" FORCE )

#set these global flags for cmake client scripts to change behavior
set( ANDROID True )
set( BUILD_ANDROID True )

#macro to find packages on the host OS
macro(find_host_package)
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
 SET( UNIX 1)
 set( CMAKE_FIND_ROOT_PATH_MODE_PROGRAM ONLY )
 set( CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY )
 set( CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY )
endmacro()
#macro to find programs on the host OS
macro(find_host_program)
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
 SET( UNIX 1)
 set( CMAKE_FIND_ROOT_PATH_MODE_PROGRAM ONLY )
 set( CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY )
 set( CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY )
endmacro()
