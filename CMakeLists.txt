# ----------------------------------------------------------------------------
#  Root CMake file for OpenCV
#
#    From the off-tree build directory, invoke:
#      $ cmake <PATH_TO_OPENCV_ROOT>
#
#
#   - OCT-2008: Initial version <joseluisblancoc@gmail.com>
#
# ----------------------------------------------------------------------------

set(CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS true)

# Following block can broke build in case of cross-compilng
# but CMAKE_CROSSCOMPILING variable will be set only on project(OpenCV) command
# so we will try to detect crosscompiling by presense of CMAKE_TOOLCHAIN_FILE
if(NOT CMAKE_TOOLCHAIN_FILE)
    # Add these standard paths to the search paths for FIND_LIBRARY
    # to find libraries from these locations first
    if(UNIX)
        set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} /lib /usr/lib)
    endif()

    # it _must_ go before PROJECT(OpenCV) in order to work
    if(WIN32)
        set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory")
    else()
        set(CMAKE_INSTALL_PREFIX "/usr/local" CACHE PATH "Installation Directory")
    endif()

    if(MSVC)
        set(CMAKE_USE_RELATIVE_PATHS ON CACHE INTERNAL "" FORCE)
    endif()
endif(NOT CMAKE_TOOLCHAIN_FILE)

# --------------------------------------------------------------
# Top level OpenCV project
# --------------------------------------------------------------
cmake_minimum_required(VERSION 2.4)
project(OpenCV)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Configs" FORCE)
if(DEFINED CMAKE_BUILD_TYPE)
    set_property( CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${CMAKE_CONFIGURATION_TYPES} )
endif()

set(CMAKE_C_FLAGS_MINSIZEREL "" CACHE INTERNAL "" FORCE)
set(CMAKE_C_FLAGS_RELWITHDEBINFO "" CACHE INTERNAL "" FORCE)
set(CMAKE_CXX_FLAGS_MINSIZEREL "" CACHE INTERNAL "" FORCE)
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "" CACHE INTERNAL "" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_MINSIZEREL "" CACHE INTERNAL "" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO "" CACHE INTERNAL "" FORCE)
set(CMAKE_MODULE_LINKER_FLAGS_MINSIZEREL "" CACHE INTERNAL "" FORCE)
set(CMAKE_MODULE_LINKER_FLAGS_RELWITHDEBINFO "" CACHE INTERNAL "" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_MINSIZEREL "" CACHE INTERNAL "" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO "" CACHE INTERNAL "" FORCE)
set(CMAKE_VERBOSE OFF CACHE BOOL "Verbose mode")
if(CMAKE_VERBOSE)
    set(CMAKE_VERBOSE_MAKEFILE 1)
endif()

# --------------------------------------------------------------
# Indicate CMake 2.7 and above that we don't want to mix relative
#  and absolute paths in linker lib lists.
# Run "cmake --help-policy CMP0003" for more information.
# --------------------------------------------------------------
if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif()

# ----------------------------------------------------------------------------
# Build static or dynamic libs?
# Default: dynamic libraries
# ----------------------------------------------------------------------------
set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries (.dll/.so) instead of static ones (.lib/.a)")

# ----------------------------------------------------------------------------
# Include debug info into debug libs?
# Default: yes
# ----------------------------------------------------------------------------
set(BUILD_WITH_DEBUG_INFO ON CACHE BOOL "Include debug info into debug libs")

# ----------------------------------------------------------------------------
#  Get actual OpenCV version number from sources
# ----------------------------------------------------------------------------
SET(OPENCV_VERSION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/modules/core/include/opencv2/core/version.hpp")
FILE(STRINGS "${OPENCV_VERSION_FILE}" OPENCV_VERSION_PARTS REGEX "#define CV_.+OR_VERSION[ ]+[0-9]+" )
string(REGEX REPLACE ".+CV_MAJOR_VERSION[ ]+([0-9]+).*" "\\1" OPENCV_VERSION_MAJOR "${OPENCV_VERSION_PARTS}")
string(REGEX REPLACE ".+CV_MINOR_VERSION[ ]+([0-9]+).*" "\\1" OPENCV_VERSION_MINOR "${OPENCV_VERSION_PARTS}")
string(REGEX REPLACE ".+CV_SUBMINOR_VERSION[ ]+([0-9]+).*" "\\1" OPENCV_VERSION_PATCH "${OPENCV_VERSION_PARTS}")
set(OPENCV_VERSION "${OPENCV_VERSION_MAJOR}.${OPENCV_VERSION_MINOR}.${OPENCV_VERSION_PATCH}")

set(OPENCV_SOVERSION "${OPENCV_VERSION_MAJOR}.${OPENCV_VERSION_MINOR}")

# create a dependency on version file
# we will never use output of the following command but cmake will rerun if version file changes
configure_file("${OPENCV_VERSION_FILE}" "${CMAKE_BINARY_DIR}/junk/version.junk" COPYONLY)

if(WIN32)
    # Postfix of DLLs:
    set(OPENCV_DLLVERSION "${OPENCV_VERSION_MAJOR}${OPENCV_VERSION_MINOR}${OPENCV_VERSION_PATCH}")
    set(OPENCV_DEBUG_POSTFIX d)
else()
    # Postfix of so's:
    set(OPENCV_DLLVERSION "")
    set(OPENCV_DEBUG_POSTFIX)
endif()

#name mangling
set(OPENCV_INCLUDE_PREFIX include)
if(UNIX AND NOT ANDROID AND BUILD_SHARED_LIBS)
    option(OPENCV_MANGLED_INSTALL_PATHS "Enables mangled install paths, that help with side by side installs." False)
    if(OPENCV_MANGLED_INSTALL_PATHS)
      set(OPENCV_INCLUDE_PREFIX include/opencv-${OPENCV_VERSION})
    endif()
endif()

# SUBDIRECTORIES:
# Save libs and executables in the same place
set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/lib" CACHE PATH "Output directory for libraries" )
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin" CACHE PATH "Output directory for applications" )

if(ANDROID OR WIN32)
    set(OPENCV_DOC_INSTALL_PATH doc)
elseif(OPENCV_MANGLED_INSTALL_PATHS)
    set(OPENCV_DOC_INSTALL_PATH share/OpenCV-${OPENCV_VERSION}/doc)
else()
    set(OPENCV_DOC_INSTALL_PATH share/OpenCV/doc)
endif()

if(ANDROID)
    set(OPENCV_LIB_INSTALL_PATH libs/${ARMEABI_NDK_NAME})
else()
    set(OPENCV_LIB_INSTALL_PATH lib${LIB_SUFFIX})
endif()

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${OPENCV_LIB_INSTALL_PATH}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# search packages for host system instead of packages for target system
# in case of cross compilation thess macro should be defined by toolchain file
if(NOT COMMAND find_host_package)
    macro(find_host_package)
        find_package(${ARGN})
    endmacro()
endif()
if(NOT COMMAND find_host_program)
    macro(find_host_program)
        find_program(${ARGN})
    endmacro()
endif()

# ----------------------------------------------------------------------------
# Use statically or dynamically linked CRT?
# Default: dynamic
# ----------------------------------------------------------------------------
if(WIN32 AND NOT BUILD_SHARED_LIBS)
  option (BUILD_WITH_STATIC_CRT "Enables use of staticaly linked CRT" ON)
endif()
  
if(MSVC)
    if(BUILD_WITH_STATIC_CRT)
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
    else(BUILD_WITH_STATIC_CRT)
        foreach(flag_var
                CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE
                CMAKE_C_FLAGS_MINSIZEREL CMAKE_C_FLAGS_RELWITHDEBINFO
                CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
                CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
           if(${flag_var} MATCHES "/MT")
              string(REGEX REPLACE "/MT" "/MD" ${flag_var} "${${flag_var}}")
           endif()
           if(${flag_var} MATCHES "/MTd")
              string(REGEX REPLACE "/MTd" "/MDd" ${flag_var} "${${flag_var}}")
           endif()
        endforeach(flag_var)
    endif(BUILD_WITH_STATIC_CRT)

    if(NOT BUILD_WITH_DEBUG_INFO)
        string(REPLACE "/debug" "" CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG}")
        string(REPLACE "/DEBUG" "" CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG}")
        string(REPLACE "/INCREMENTAL:YES" "/INCREMENTAL:NO" CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG}")
        string(REPLACE "/INCREMENTAL " "/INCREMENTAL:NO " CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG}")

        string(REPLACE "/debug" "" CMAKE_MODULE_LINKER_FLAGS_DEBUG "${CMAKE_MODULE_LINKER_FLAGS_DEBUG}")
        string(REPLACE "/DEBUG" "" CMAKE_MODULE_LINKER_FLAGS_DEBUG "${CMAKE_MODULE_LINKER_FLAGS_DEBUG}")
        string(REPLACE "/INCREMENTAL:YES" "/INCREMENTAL:NO" CMAKE_MODULE_LINKER_FLAGS_DEBUG "${CMAKE_MODULE_LINKER_FLAGS_DEBUG}")
        string(REPLACE "/INCREMENTAL " "/INCREMENTAL:NO " CMAKE_MODULE_LINKER_FLAGS_DEBUG "${CMAKE_MODULE_LINKER_FLAGS_DEBUG}")

        string(REPLACE "/debug" "" CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS_DEBUG}")
        string(REPLACE "/DEBUG" "" CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS_DEBUG}")
        string(REPLACE "/INCREMENTAL:YES" "/INCREMENTAL:NO" CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS_DEBUG}")
        string(REPLACE "/INCREMENTAL " "/INCREMENTAL:NO " CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS_DEBUG}")

        string(REPLACE "/Zi" "" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
        string(REPLACE "/Zi" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
    endif()

endif(MSVC)

# ----------------------------------------------------------------------------
#  Variables for cvconfig.h.cmake
# ----------------------------------------------------------------------------
set(PACKAGE "opencv")
set(PACKAGE_BUGREPORT "opencvlibrary-devel@lists.sourceforge.net")
set(PACKAGE_NAME "opencv")
set(PACKAGE_STRING "${PACKAGE} ${OPENCV_VERSION}")
set(PACKAGE_TARNAME "${PACKAGE}")
set(PACKAGE_VERSION "${OPENCV_VERSION}")

# ----------------------------------------------------------------------------
#  Autodetect if we are in a SVN repository
# ----------------------------------------------------------------------------
find_host_program(SVNVERSION_PATH svnversion)
mark_as_advanced(force SVNVERSION_PATH)
if(SVNVERSION_PATH)
    message(STATUS "Extracting svn version, please wait...")
    execute_process(COMMAND ${SVNVERSION_PATH} -n ${OpenCV_SOURCE_DIR} OUTPUT_VARIABLE SVNVERSION_RESULT)

    if(SVNVERSION_RESULT MATCHES "exported")
        # This is NOT a svn repository:
        set(OPENCV_SVNVERSION "")
        message(STATUS "SVNVERSION: exported")
    else()
        set(OPENCV_SVNVERSION " svn:${SVNVERSION_RESULT}")
        message(STATUS "SVNVERSION: ${OPENCV_SVNVERSION}")
    endif()
else()
    # We don't have svnversion:
    set(OPENCV_SVNVERSION "")
endif()

# ----------------------------------------------------------------------------
# Detect Microsoft compiler:
# ----------------------------------------------------------------------------
if(CMAKE_CL_64)
    set(MSVC64 1)
endif()

# ----------------------------------------------------------------------------
# Detect GNU version:
# ----------------------------------------------------------------------------
if(CMAKE_COMPILER_IS_GNUCXX)
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} --version
                  OUTPUT_VARIABLE CMAKE_OPENCV_GCC_VERSION_FULL
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
    
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} -v
                  ERROR_VARIABLE CMAKE_OPENCV_GCC_INFO_FULL
                  OUTPUT_STRIP_TRAILING_WHITESPACE)           
        
    # Typical output in CMAKE_OPENCV_GCC_VERSION_FULL: "c+//0 (whatever) 4.2.3 (...)"
    # Look for the version number
    string(REGEX MATCH "[0-9]+.[0-9]+.[0-9]+" CMAKE_GCC_REGEX_VERSION "${CMAKE_OPENCV_GCC_VERSION_FULL}")

    # Split the three parts:
    string(REGEX MATCHALL "[0-9]+" CMAKE_OPENCV_GCC_VERSIONS "${CMAKE_GCC_REGEX_VERSION}")

    list(GET CMAKE_OPENCV_GCC_VERSIONS 0 CMAKE_OPENCV_GCC_VERSION_MAJOR)
    list(GET CMAKE_OPENCV_GCC_VERSIONS 1 CMAKE_OPENCV_GCC_VERSION_MINOR)

    set(CMAKE_OPENCV_GCC_VERSION ${CMAKE_OPENCV_GCC_VERSION_MAJOR}${CMAKE_OPENCV_GCC_VERSION_MINOR})
    math(EXPR CMAKE_OPENCV_GCC_VERSION_NUM "${CMAKE_OPENCV_GCC_VERSION_MAJOR}*100 + ${CMAKE_OPENCV_GCC_VERSION_MINOR}")
    message(STATUS "Detected version of GNU GCC: ${CMAKE_OPENCV_GCC_VERSION} (${CMAKE_OPENCV_GCC_VERSION_NUM})")

    if(WIN32)
        execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpmachine
                  OUTPUT_VARIABLE CMAKE_OPENCV_GCC_TARGET_MACHINE
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(CMAKE_OPENCV_GCC_TARGET_MACHINE MATCHES "64")
            set(MINGW64 1)
        endif()
    endif()
endif()

# ----------------------------------------------------------------------------
# Detect Intel ICC compiler -- for -fPIC in 3rdparty ( UNIX ONLY ):
#  see  include/opencv/cxtypes.h file for related   ICC & CV_ICC defines.
# NOTE: The system needs to determine if the '-fPIC' option needs to be added
#  for the 3rdparty static libs being compiled.  The CMakeLists.txt files
#  in 3rdparty use the CV_ICC definition being set here to determine if
#  the -fPIC flag should be used.
# ----------------------------------------------------------------------------
if(UNIX)
    if  (__ICL)
        set(CV_ICC   __ICL)
    elseif(__ICC)
        set(CV_ICC   __ICC)
    elseif(__ECL)
        set(CV_ICC   __ECL)
    elseif(__ECC)
        set(CV_ICC   __ECC)
    elseif(__INTEL_COMPILER)
        set(CV_ICC   __INTEL_COMPILER)
    elseif(CMAKE_C_COMPILER MATCHES "icc")
        set(CV_ICC   icc_matches_c_compiler)
    endif()
endif()

# ----------------------------------------------------------------------------
#       CHECK FOR SYSTEM LIBRARIES, OPTIONS, ETC..
# ----------------------------------------------------------------------------

# Build/install (or not) some apps:
# ===================================================
set(BUILD_EXAMPLES OFF CACHE BOOL "Build all examples")
set(INSTALL_C_EXAMPLES OFF CACHE BOOL "Install C examples")
set(INSTALL_PYTHON_EXAMPLES OFF CACHE BOOL "Install Python examples")
if(ANDROID)
    set(INSTALL_ANDROID_EXAMPLES OFF CACHE BOOL "Install Android examples")
endif()

# Build tests:
# ===================================================
set(BUILD_TESTS ON CACHE BOOL "Build tests")

# Build 3rdparty libraries under unix
# ===================================================
if(WIN32)
    set(OPENCV_BUILD_3RDPARTY_LIBS TRUE  CACHE BOOL "Build 3rd party libraries")
else()
    set(OPENCV_BUILD_3RDPARTY_LIBS FALSE CACHE BOOL "Build 3rd party libraries")
endif()

include(OpenCVPCHSupport.cmake REQUIRED)
include(OpenCVModule.cmake REQUIRED)
if(ANDROID)
    include(OpenCVAndroidProject.cmake REQUIRED)
endif()

if(PCHSupport_FOUND)
    SET(USE_PRECOMPILED_HEADERS ON CACHE BOOL "Use precompiled headers")
else()
    SET(USE_PRECOMPILED_HEADERS OFF CACHE BOOL "Use precompiled headers" FORCE)
endif()

if(UNIX)
    include(OpenCVFindPkgConfig.cmake OPTIONAL)
    include(CheckFunctionExists)
    include(CheckIncludeFile)
endif()

if(CMAKE_COMPILER_IS_GNUCXX)
    set(ENABLE_PROFILING OFF CACHE BOOL "Enable profiling in the GCC compiler (Add flags: -g -pg)")
    set(USE_OMIT_FRAME_POINTER ON CACHE BOOL "Enable -fomit-frame-pointer for GCC")

    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES amd64.*|x86_64.*)
        set(X86_64 1)
    endif()
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES i686.*|i386.*|x86.*)
        set(X86 1)
    endif()
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES powerpc.*)
        set(ENABLE_POWERPC ON CACHE BOOL "Enable PowerPC for GCC")
    endif ()

    if(X86 OR X86_64)
        # enable everything, since the available set of instructions is checked at runtime
        #IF ("${CMAKE_GCC_REGEX_VERSION}" VERSION_GREATER 4.5)
            SET(_USE_FAST_MATH OFF)
        #ELSE()
        #    SET(_USE_FAST_MATH ON)
        #ENDIF()
        set(USE_FAST_MATH ${_USE_FAST_MATH} CACHE BOOL "Enable -ffast-math (not recommended for GCC 4.6.x)")
        set(ENABLE_SSE ON CACHE BOOL "Enable SSE instructions")
        set(ENABLE_SSE2 ON CACHE BOOL "Enable SSE2 instructions")
        set(ENABLE_SSE3 OFF CACHE BOOL "Enable SSE3 instructions")
        set(ENABLE_SSSE3 OFF CACHE BOOL "Enable SSSE3 instructions")
        set(ENABLE_SSE41 OFF CACHE BOOL "Enable SSE4.1 instructions")
        set(ENABLE_SSE42 OFF CACHE BOOL "Enable SSE4.2 instructions")
    endif()
endif()

if(MSVC)
    set(ENABLE_SSE ON CACHE BOOL "Enable SSE instructions for MSVC")
    set(ENABLE_SSE2 ON CACHE BOOL "Enable SSE2 instructions for MSVC")
    if(CMAKE_C_COMPILER MATCHES "icc")
        set(ENABLE_SSE3 OFF CACHE BOOL "Enable SSE3 instructions for ICC")
        set(ENABLE_SSE4_1 OFF CACHE BOOL "Enable SSE4.1 instructions for ICC")
    endif()
endif()

# allow fine grained control over which libraries not to link, even if
# they are available on the system
# ====================================================================
set(WITH_PNG ON CACHE BOOL "Include PNG support")
set(WITH_JPEG ON CACHE BOOL "Include JPEG support")
set(WITH_JASPER ON CACHE BOOL "Include JPEG2K support")
set(WITH_TIFF ON CACHE BOOL "Include TIFF support")
set(WITH_OPENEXR ON CACHE BOOL "Include ILM support via OpenEXR")

if(UNIX)
    set(WITH_FFMPEG ON CACHE BOOL "Include FFMPEG support")
    if(NOT APPLE)
        set(WITH_UNICAP OFF CACHE BOOL "Include Unicap support (GPL)")
        set(WITH_GTK ON CACHE BOOL "Include GTK support")
        set(WITH_GSTREAMER ON CACHE BOOL "Include Gstreamer support")
        set(WITH_V4L ON CACHE BOOL "Include Video 4 Linux support")
        set(WITH_XINE OFF CACHE BOOL "Include Xine support (GPL)")
    endif()
    set(WITH_PVAPI ON CACHE BOOL "Include Prosilica GigE support")
    set(WITH_1394 ON CACHE BOOL "Include IEEE1394 support")
endif()

if(APPLE)
    set(WITH_CARBON OFF CACHE BOOL "Use Carbon for UI instead of Cocoa")
    set(WITH_QUICKTIME OFF CACHE BOOL "Use QuickTime for Video I/O insted of QTKit")
endif()

set(WITH_TBB OFF CACHE BOOL "Include Intel TBB support")
set(WITH_IPP OFF CACHE BOOL "Include Intel IPP support")
set(WITH_EIGEN ON CACHE BOOL "Include Eigen2/Eigen3 support")
set(WITH_CUDA ON CACHE BOOL "Include NVidia Cuda Runtime support")

set(WITH_OPENNI OFF CACHE BOOL "Include OpenNI support")
set(WITH_XIMEA OFF CACHE BOOL "Include XIMEA cameras support")

# ===================================================
# Macros that checks if module have been installed.
# After it adds module to build and define
# constants passed as second arg
# ===================================================

macro(CHECK_MODULE module_name define)
    set(${define} 0)
    if(PKG_CONFIG_FOUND)
        set(ALIAS               ALIASOF_${module_name})
        set(ALIAS_FOUND                 ${ALIAS}_FOUND)
        set(ALIAS_INCLUDE_DIRS   ${ALIAS}_INCLUDE_DIRS)
        set(ALIAS_LIBRARY_DIRS   ${ALIAS}_LIBRARY_DIRS)
        set(ALIAS_LIBRARIES         ${ALIAS}_LIBRARIES)

        PKG_CHECK_MODULES(${ALIAS} ${module_name})

        if (${ALIAS_FOUND})
            set(${define} 1)
            foreach(P "${ALIAS_INCLUDE_DIRS}")
                if (${P})
                    list(APPEND HIGHGUI_INCLUDE_DIRS ${${P}})
                endif()
            endforeach()

            foreach(P "${ALIAS_LIBRARY_DIRS}")
                if (${P})
                    list(APPEND HIGHGUI_LIBRARY_DIRS ${${P}})
                endif()
            endforeach()

            list(APPEND HIGHGUI_LIBRARIES ${${ALIAS_LIBRARIES}})
        endif()
    endif()
endmacro()

if(UNIX)
    if(NOT APPLE)
      if(WITH_GTK)
        CHECK_MODULE(gtk+-2.0 HAVE_GTK)
        CHECK_MODULE(gthread-2.0 HAVE_GTHREAD)
      else()
        set(HAVE_GTK FALSE)
        set(HAVE_GTHREAD FALSE)
      endif()
      if(WITH_GSTREAMER)
        CHECK_MODULE(gstreamer-base-0.10 HAVE_GSTREAMER)
        CHECK_MODULE(gstreamer-app-0.10 HAVE_GSTREAMER)
        CHECK_MODULE(gstreamer-video-0.10 HAVE_GSTREAMER)
      else()
        set(HAVE_GSTREAMER FALSE)
      endif()
    endif()

    if(WITH_UNICAP)
      CHECK_MODULE(libunicap HAVE_UNICAP_)
      CHECK_MODULE(libucil HAVE_UNICAP_UCIL)
      if(HAVE_UNICAP_ AND HAVE_UNICAP_UCIL)
        set(HAVE_UNICAP 1)
      endif()
    else()
      set(HAVE_UNICAP FALSE)
    endif()

    if(WITH_PVAPI)
      find_path(PVAPI_INCLUDE_PATH "PvApi.h"
                PATHS "/usr/local/include" "/usr/include"
                DOC "The path to PvAPI header")
      if(PVAPI_INCLUDE_PATH)
        set(HAVE_PVAPI 1)
      endif()
    endif()

    set(HAVE_FFMPEG 0)
    if(WITH_FFMPEG)
      CHECK_MODULE(libavcodec HAVE_FFMPEG_CODEC)
      CHECK_MODULE(libavformat HAVE_FFMPEG_FORMAT)
      CHECK_MODULE(libavutil HAVE_FFMPEG_UTIL)
      CHECK_MODULE(libswscale HAVE_FFMPEG_SWSCALE)
      CHECK_INCLUDE_FILE(libavformat/avformat.h HAVE_GENTOO_FFMPEG)
      CHECK_INCLUDE_FILE(ffmpeg/avformat.h HAVE_FFMPEG_FFMPEG)
      if(NOT HAVE_GENTOO_FFMPEG AND NOT HAVE_FFMPEG_FFMPEG)
        if(EXISTS /usr/include/ffmpeg/libavformat/avformat.h OR HAVE_FFMPEG_SWSCALE)
          set(HAVE_GENTOO_FFMPEG 1)
        endif()
      endif()
      if(HAVE_FFMPEG_CODEC AND HAVE_FFMPEG_FORMAT AND HAVE_FFMPEG_UTIL)
        if(HAVE_FFMPEG_SWSCALE OR NOT HAVE_GENTOO_FFMPEG)
            set(HAVE_FFMPEG 1)
        endif()
      endif()
      # Find the bzip2 library because it is required on some systems
      FIND_LIBRARY(BZIP2_LIBRARIES NAMES bz2 bzip2)
      if(NOT BZIP2_LIBRARIES)
        # Do an other trial
        FIND_FILE(BZIP2_LIBRARIES NAMES libbz2.so.1 PATHS /lib)
      endif()
    endif()

    if(WITH_1394)
      CHECK_MODULE(libdc1394-2 HAVE_DC1394_2)
      if(NOT HAVE_DC1394_2)
        CHECK_MODULE(libdc1394 HAVE_DC1394)
      endif()
    else()
      set(HAVE_DC1394_2 FALSE)
      set(HAVE_DC1394 FALSE)
    endif()

    if(NOT OPENCV_BUILD_3RDPARTY_LIBS)
        include(FindZLIB)
        if(WITH_PNG)
            include(FindPNG)
            if(PNG_FOUND)
                CHECK_INCLUDE_FILE(${PNG_PNG_INCLUDE_DIR}/png.h HAVE_PNG_H)
                CHECK_INCLUDE_FILE(${PNG_PNG_INCLUDE_DIR}/libpng/png.h HAVE_LIBPNG_PNG_H)
            endif()
        else()
            set(PNG_FOUND FALSE)
        endif()
        if(WITH_TIFF)
            include(FindTIFF)
        else()
            set(TIFF_FOUND FALSE)
        endif()
        if(WITH_JASPER)
            include(FindJasper)
        else()
            set(JASPER_FOUND FALSE)
        endif()
        if(WITH_JPEG)
            include(FindJPEG)
        else()
            set(JPEG_FOUND FALSE)
        endif()
    endif()

    if(NOT APPLE)
        CHECK_INCLUDE_FILE(alloca.h HAVE_ALLOCA_H)
        CHECK_FUNCTION_EXISTS(alloca HAVE_ALLOCA)
        CHECK_INCLUDE_FILE(unistd.h HAVE_UNISTD_H)
        CHECK_INCLUDE_FILE(pthread.h HAVE_LIBPTHREAD)

        if(WITH_XINE)
            CHECK_MODULE(libxine HAVE_XINE)
        else()
            set(HAVE_XINE FALSE)
        endif()
        if(WITH_V4L)
            CHECK_MODULE(libv4l1 HAVE_LIBV4L)
            CHECK_INCLUDE_FILE(linux/videodev.h HAVE_CAMV4L)
            CHECK_INCLUDE_FILE(linux/videodev2.h HAVE_CAMV4L2)
        else()
            set(HAVE_LIBV4L FALSE)
            set(HAVE_CAMV4L FALSE)
            set(HAVE_CAMV4L2 FALSE)
        endif()

        if(ANDROID)
            set(OPENCV_LINKER_LIBS dl m log)
        elseif(${CMAKE_SYSTEM_NAME} MATCHES "FreeBSD" OR ${CMAKE_SYSTEM_NAME} MATCHES "NetBSD")
            set(OPENCV_LINKER_LIBS m pthread)
        else()
            set(OPENCV_LINKER_LIBS dl m pthread rt)
        endif()
    else()
        add_definitions(-DHAVE_ALLOCA -DHAVE_ALLOCA_H -DHAVE_LIBPTHREAD -DHAVE_UNISTD_H)
    endif()
endif()

if(WITH_PNG AND NOT PNG_FOUND)
    set(PNG_LIBRARIES libpng)
endif()

if(WITH_JPEG AND NOT JPEG_FOUND)
    set(JPEG_LIBRARIES libjpeg)
endif()

if(WITH_TIFF AND NOT TIFF_FOUND)
    set(TIFF_LIBRARIES libtiff)
endif()

if(WITH_JASPER AND NOT JASPER_FOUND)
    set(JASPER_LIBRARIES libjasper)
endif()

if(NOT ZLIB_FOUND)
    set(ZLIB_LIBRARY zlib)
endif()

#message(STATUS "Graphic libraries: ${PNG_LIBRARIES} ${JPEG_LIBRARIES} ${TIFF_LIBRARIES} ${JASPER_LIBRARIES}")

if(WITH_OPENEXR)
    include(OpenCVFindOpenEXR.cmake)
endif()

set(BUILD_DOCS ON CACHE BOOL "Build OpenCV Documentation")

if(BUILD_DOCS)
  include(OpenCVFindLATEX.cmake REQUIRED)
endif()

set(BUILD_NEW_PYTHON_SUPPORT ON CACHE BOOL "Build with Python support")

if (WIN32)
    if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        set(BUILD_NEW_PYTHON_SUPPORT OFF)
    endif()
endif()

if(ANDROID)
    set(BUILD_NEW_PYTHON_SUPPORT OFF)
endif()

# Always try to find python
# ===================================================
find_host_package(PythonInterp)
find_host_package(PythonLibs)

# cmake 2.4 (at least on Ubuntu 8.04 (hardy)) don't define PYTHONLIBS_FOUND
if(NOT PYTHONLIBS_FOUND AND PYTHON_INCLUDE_PATH)
    set(PYTHONLIBS_FOUND ON)
endif()

execute_process(COMMAND ${PYTHON_EXECUTABLE} --version
      ERROR_VARIABLE PYTHON_VERSION_FULL
      OUTPUT_STRIP_TRAILING_WHITESPACE)

string(REGEX MATCH "[0-9]+.[0-9]+" PYTHON_VERSION_MAJOR_MINOR "${PYTHON_VERSION_FULL}")
if(CMAKE_HOST_UNIX)
    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "from distutils.sysconfig import *; print get_python_lib()"
                    RESULT_VARIABLE PYTHON_CVPY_PROCESS
                    OUTPUT_VARIABLE PYTHON_STD_PACKAGES_PATH
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    if("${PYTHON_STD_PACKAGES_PATH}" MATCHES "site-packages")
        set(PYTHON_PACKAGES_PATH lib/python${PYTHON_VERSION_MAJOR_MINOR}/site-packages CACHE PATH "Where to install the python packages.")
    else() #debian based assumed, install to the dist-packages.
        set(PYTHON_PACKAGES_PATH lib/python${PYTHON_VERSION_MAJOR_MINOR}/dist-packages CACHE PATH "Where to install the python packages.")
    endif()
endif()
if(CMAKE_HOST_WIN32)
    get_filename_component(PYTHON_PATH "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Python\\PythonCore\\${PYTHON_VERSION_MAJOR_MINOR}\\InstallPath]" ABSOLUTE CACHE)
    set(PYTHON_PACKAGES_PATH "${PYTHON_PATH}/Lib/site-packages")
endif()

IF ("${PYTHON_VERSION_MAJOR_MINOR}" VERSION_GREATER 2.4)
    SET(PYTHON_EXECUTABLE ${PYTHON_EXECUTABLE} -B)
ENDIF()

# Attempt to discover the NumPy include directory. If this succeeds, then build python API with NumPy
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import os; os.environ['DISTUTILS_USE_SDK']='1'; import numpy.distutils; print numpy.distutils.misc_util.get_numpy_include_dirs()[0]"
            RESULT_VARIABLE PYTHON_NUMPY_PROCESS
            OUTPUT_VARIABLE PYTHON_NUMPY_INCLUDE_DIRS
            OUTPUT_STRIP_TRAILING_WHITESPACE)

if(PYTHON_NUMPY_PROCESS EQUAL 0)
    set(PYTHON_USE_NUMPY 1)
    add_definitions(-D PYTHON_USE_NUMPY=1)
    include_directories(AFTER ${PYTHON_NUMPY_INCLUDE_DIRS})
    message(STATUS "    Use NumPy headers from: ${PYTHON_NUMPY_INCLUDE_DIRS}")
else()
    set(PYTHON_USE_NUMPY 0)
endif()

execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import sphinx; print sphinx.__version__"
            RESULT_VARIABLE SPHINX_PROCESS
            OUTPUT_VARIABLE SPHINX_VERSION
            OUTPUT_STRIP_TRAILING_WHITESPACE)

set(HAVE_SPHINX 0)
if(SPHINX_PROCESS EQUAL 0)
    find_host_program(SPHINX_BUILD sphinx-build)
    if(SPHINX_BUILD)
        set(HAVE_SPHINX 1)
        message(STATUS "    Found Sphinx ${SPHINX_VERSION}: ${SPHINX_BUILD}")
    endif()
endif()

# Java support
# ===================================================
if (PYTHON_EXECUTABLE AND ANDROID AND ANDROID_API_LEVEL GREATER 7)
    option(BUILD_JAVA_SUPPORT "Build with Java support" TRUE)
endif()

if (BUILD_JAVA_SUPPORT)
    file(TO_CMAKE_PATH "$ENV{ANT_DIR}" ANT_DIR_ENV_PATH)
    file(TO_CMAKE_PATH "$ENV{ProgramFiles}" ProgramFiles_ENV_PATH)

    if(ANDROID)
        file(TO_CMAKE_PATH "$ENV{ANDROID_SDK}" ANDROID_SDK_ENV_PATH)
        
        #find android SDK
        find_host_program(ANDROID_EXECUTABLE
            NAMES android.bat android
            PATHS "${ANDROID_SDK_ENV_PATH}/tools/"
                  "${ProgramFiles_ENV_PATH}/Android/android-sdk/tools/"
                  "/opt/android-sdk/tools/"
                  "/opt/android-sdk-linux_x86/tools/"
                  "/opt/android-sdk-mac_x86/tools/"
                  "/opt/android-sdk-linux_86/tools/"
                  "/opt/android-sdk-mac_86/tools/"
            )
        if(ANDROID_EXECUTABLE)
            message(STATUS "    Found android tool: ${ANDROID_EXECUTABLE}")

            get_filename_component(ANDROID_SDK_TOOLS_PATH "${ANDROID_EXECUTABLE}" PATH)

            #read source.properties
            if (EXISTS "${ANDROID_SDK_TOOLS_PATH}/source.properties")
                file(STRINGS "${ANDROID_SDK_TOOLS_PATH}/source.properties" ANDROID_SDK_TOOLS_SOURCE_PROPERTIES_LINES REGEX "^[ ]*[^#].*$")
                foreach(line ${ANDROID_SDK_TOOLS_SOURCE_PROPERTIES_LINES})
                    string(REPLACE "\\:" ":" line ${line})
                    string(REPLACE "=" ";" line ${line})
                    list(GET line 0 line_name)
                    list(GET line 1 line_value)
                    string(REPLACE "." "_" line_name ${line_name})
                    SET(ANDROID_TOOLS_${line_name} "${line_value}")
                    MARK_AS_ADVANCED(ANDROID_TOOLS_${line_name})
                endforeach()
            endif()
            if (NOT ANDROID_TOOLS_Pkg_Revision)
                SET(ANDROID_TOOLS_Pkg_Revision "Unknown")
                MARK_AS_ADVANCED(ANDROID_TOOLS_Pkg_Revision)
            endif()
            if (NOT ANDROID_TOOLS_Pkg_Desc)
                SET(ANDROID_TOOLS_Pkg_Desc "Android SDK Tools, revision ${ANDROID_TOOLS_Pkg_Revision}.")
                if (NOT ANDROID_TOOLS_Pkg_Revision GREATER 11)
                    SET(ANDROID_TOOLS_Pkg_Desc "${ANDROID_TOOLS_Pkg_Desc} It is recommended to update your SDK tools to revision 12 or newer.")
                endif()
                MARK_AS_ADVANCED(ANDROID_TOOLS_Pkg_Desc)
            endif()

            #get installed targets
            execute_process(COMMAND ${ANDROID_EXECUTABLE} list target
                RESULT_VARIABLE ANDROID_PROCESS
                OUTPUT_VARIABLE ANDROID_SDK_TARGETS_FULL
                ERROR_VARIABLE ANDROID_PROCESS_ERRORS
                OUTPUT_STRIP_TRAILING_WHITESPACE)
            string(REGEX MATCHALL "(^|\n)id: [0-9]+ or \"([^\n]+[0-9+])\"(\n|$)" ANDROID_SDK_TARGETS_FULL "${ANDROID_SDK_TARGETS_FULL}")

            SET(ANDROID_SDK_TARGETS)
            if (ANDROID_PROCESS EQUAL 0)
                foreach(line ${ANDROID_SDK_TARGETS_FULL})
                    string(REGEX REPLACE "(^|\n)id: [0-9]+ or \"([^\n]+[0-9+])\"(\n|$)" "\\2" line "${line}")
                    list(APPEND ANDROID_SDK_TARGETS "${line}")
                endforeach()
            endif()

            # detect ANDROID_SDK_TARGET if no target is provided by user
            if (NOT ANDROID_SDK_TARGET)
                set(desired_android_target_level ${ANDROID_API_LEVEL})
                if (desired_android_target_level LESS 8)
                    set(desired_android_target_level 8)
                endif()
                if (ANDROID_PROCESS EQUAL 0)
                    math(EXPR desired_android_target_level_1 "${desired_android_target_level}-1")

                    foreach(target ${ANDROID_SDK_TARGETS})
                        string(REGEX MATCH "[0-9]+$" target_level "${target}")
                        if (target_level GREATER desired_android_target_level_1)
                           set(ANDROID_SDK_TARGET "${target}")
                           break()
                        endif()
                    endforeach()
                else()
                    set(ANDROID_SDK_TARGET android-${desired_android_target_level})
                    message(WARNING "Could not retrieve list of installed Android targets. Will try to use \"${ANDROID_SDK_TARGET}\" target")
                endif()
            endif()

            SET(ANDROID_SDK_TARGET ${ANDROID_SDK_TARGET} CACHE STRING "SDK target for Android tests and samples")
            if (ANDROID_PROCESS EQUAL 0)
                set_property( CACHE ANDROID_SDK_TARGET PROPERTY STRINGS ${ANDROID_SDK_TARGETS} )
            endif()
            string(REGEX MATCH "[0-9]+$" ANDROID_SDK_TARGET_LEVEL "${ANDROID_SDK_TARGET}")
        endif()
    endif()

    #find apache ant
    find_host_program(ANT_EXECUTABLE NAMES ant.bat ant
        PATHS "${ANT_DIR_ENV_PATH}/bin"
              "${ProgramFiles_ENV_PATH}/apache-ant/bin"
        )
    if(ANT_EXECUTABLE)
        execute_process(COMMAND ${ANT_EXECUTABLE} -version
            OUTPUT_VARIABLE ANT_VERSION_FULL
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        string(REGEX MATCH "[0-9]+.[0-9]+.[0-9]+" ANT_VERSION "${ANT_VERSION_FULL}")

        message(STATUS "    Found apache ant ${ANT_VERSION}: ${ANT_EXECUTABLE}")
    endif()

    if (ANDROID_EXECUTABLE AND ANT_EXECUTABLE AND (ANT_VERSION VERSION_GREATER 1.7) AND (ANDROID_SDK_TARGET_LEVEL GREATER 7))
        SET(CAN_BUILD_ANDROID_PROJECTS TRUE)
    else()
        SET(CAN_BUILD_ANDROID_PROJECTS FALSE)
    endif()
endif()

if(CAN_BUILD_ANDROID_PROJECTS)
    SET(BUILD_ANDROID_EXAMPLES TRUE CACHE BOOL "Build examples for Android platform")
endif()

#YV
############################### QT ################################

set(WITH_QT OFF CACHE BOOL "Build with Qt Backend support")
set(WITH_QT_OPENGL OFF CACHE BOOL "Add OpenGL extension to Qt")

set(HAVE_QT 0)
set(HAVE_QT_OPENGL 0)

if (WITH_QT)
    find_package(Qt4)
    if (QT4_FOUND)
        set(HAVE_QT 1)
        add_definitions(-DHAVE_QT)#We need to define te macro this way, using cvconfig.h.cmake does not work
        find_package (OpenGL QUIET)

        #if (NOT WIN32)
        if (WITH_QT_OPENGL)
            if (QT_QTOPENGL_FOUND AND OPENGL_FOUND)
                set(HAVE_QT_OPENGL 1)
                add_definitions(-DHAVE_QT_OPENGL)
                #link_directories("${OPENGL_LIBRARIES}")
                set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} ${OPENGL_LIBRARIES})
            endif()
        endif()
        #endif()
    endif()
endif()

############################### TBB ################################

if (WITH_TBB)
    if (UNIX AND NOT APPLE AND NOT ANDROID)
        PKG_CHECK_MODULES(TBB tbb)

        if (TBB_FOUND)
            set(HAVE_TBB 1)
            if(NOT ${TBB_INCLUDE_DIRS} STREQUAL "")
                include_directories(${TBB_INCLUDE_DIRS})
            endif()
            link_directories(${TBB_LIBRARY_DIRS})
            set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} ${TBB_LIBRARIES})        
        endif()
    endif()    
    
    if (NOT HAVE_TBB)
        set(TBB_DEFAULT_INCLUDE_DIRS "/opt/intel/tbb" "/usr/local/include" "/usr/include" "C:/Program Files/Intel/TBB" "C:/Program Files (x86)/Intel/TBB" "C:/Program Files (x86)/TBB" "${CMAKE_INSTALL_PREFIX}/include")
        
        find_path(TBB_INCLUDE_DIR "tbb/tbb.h" PATHS ${TBB_DEFAULT_INCLUDE_DIRS} DOC "The path to TBB headers")
        if (TBB_INCLUDE_DIR)
            if (UNIX)
                set(TBB_LIB_DIR "${TBB_INCLUDE_DIR}/../lib" CACHE PATH "Full path of TBB library directory")
                link_directories("${TBB_LIB_DIR}")
            endif()
            if (APPLE)
                set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} libtbb.dylib)
            elseif (ANDROID)
                set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} tbbmalloc tbb)
                add_definitions(-DTBB_USE_GCC_BUILTINS)
            elseif (UNIX)
                set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} tbb)
            elseif (WIN32)
                if (CMAKE_COMPILER_IS_GNUCXX)
                    set(TBB_LIB_DIR "${TBB_INCLUDE_DIR}/../lib" CACHE PATH "Full path of TBB library directory")
                    link_directories("${TBB_LIB_DIR}")
                    set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} tbb)
                else()
                    get_filename_component(_TBB_LIB_PATH "${TBB_INCLUDE_DIR}/../lib" ABSOLUTE)                          
                   
                    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES amd64*|x86_64* OR MSVC64)
                        set(_TBB_LIB_PATH "${_TBB_LIB_PATH}/intel64")
                    elseif(${CMAKE_SYSTEM_PROCESSOR} MATCHES x86*|i386*|i686*)
                        set(_TBB_LIB_PATH "${_TBB_LIB_PATH}/ia32")
                    endif()

                    if (MSVC80)
                       set(_TBB_LIB_PATH "${_TBB_LIB_PATH}/vc8")
                    elseif(MSVC90)
                       set(_TBB_LIB_PATH "${_TBB_LIB_PATH}/vc9")                    
                    elseif(MSVC10)
                        set(_TBB_LIB_PATH "${_TBB_LIB_PATH}/vc10")
                    endif()
                    set(TBB_LIB_DIR "${_TBB_LIB_PATH}" CACHE PATH "Full path of TBB library directory")
                    link_directories("${TBB_LIB_DIR}")
                endif()
            endif()

            set(HAVE_TBB 1)
            if(NOT "${TBB_INCLUDE_DIR}" STREQUAL "")
                include_directories("${TBB_INCLUDE_DIR}")
            endif()
        endif()
    endif()
endif(WITH_TBB)

#Threading Framework -- temporary decision for ARM-s instead of TBB
if (NOT HAVE_TBB)
    file(GLOB THREADING_FRAMEWORK_HEADER "${OpenCV_SOURCE_DIR}/modules/core/include/opencv2/core/threading_framework.hpp")
    file(GLOB THREADING_FRAMEWORK_SOURCE "${OpenCV_SOURCE_DIR}/modules/core/src/threading_framework.cpp")
    if(THREADING_FRAMEWORK_HEADER AND THREADING_FRAMEWORK_SOURCE)
        set(HAVE_THREADING_FRAMEWORK 1)
    endif()
endif()

############################ Intel IPP #############################
set(IPP_FOUND)

if(WITH_IPP)
    include(OpenCVFindIPP.cmake)
endif()

if(IPP_FOUND)
    add_definitions(-DHAVE_IPP)
    include_directories(${IPP_INCLUDE_DIRS})
    link_directories(${IPP_LIBRARY_DIRS})
    set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} ${IPP_LIBRARIES})
endif()


############################### CUDA ################################

if(WITH_CUDA)
    find_package(CUDA 4.0)
    
    if(CUDA_FOUND)
        set(HAVE_CUDA 1)
        message(STATUS "CUDA detected: " ${CUDA_VERSION})

        set(CUDA_ARCH_BIN "1.1 1.2 1.3 2.0 2.1(2.0)" CACHE STRING "Specify 'real' GPU architectures to build binaries for, BIN(PTX) format is supported")
        set(CUDA_ARCH_PTX "2.0" CACHE STRING "Specify 'virtual' PTX architectures to build PTX intermediate code for")
        
        string(REGEX REPLACE "\\." "" ARCH_BIN_NO_POINTS "${CUDA_ARCH_BIN}")
        string(REGEX REPLACE "\\." "" ARCH_PTX_NO_POINTS "${CUDA_ARCH_PTX}")
        
        # Ckeck if user specified 1.0 compute capability: we don't support it
        string(REGEX MATCH "1.0" HAS_ARCH_10 "${CUDA_ARCH_BIN} ${CUDA_ARCH_PTX}")
        set(CUDA_ARCH_BIN_OR_PTX_10 0)
        if(NOT ${HAS_ARCH_10} STREQUAL "")
            set(CUDA_ARCH_BIN_OR_PTX_10 1)
        endif()
        
        # NVCC flags to be set
        set(NVCC_FLAGS_EXTRA "")      
        
        # These vars will be passed into the templates
        set(OPENCV_CUDA_ARCH_BIN "") 
        set(OPENCV_CUDA_ARCH_PTX "") 
        set(OPENCV_CUDA_ARCH_FEATURES "")        
        
        # Tell NVCC to add binaries for the specified GPUs
        string(REGEX MATCHALL "[0-9()]+" ARCH_LIST "${ARCH_BIN_NO_POINTS}")
        foreach(ARCH IN LISTS ARCH_LIST)
            if (ARCH MATCHES "([0-9]+)\\(([0-9]+)\\)")
                # User explicitly specified PTX for the concrete BIN
                set(NVCC_FLAGS_EXTRA ${NVCC_FLAGS_EXTRA} -gencode arch=compute_${CMAKE_MATCH_2},code=sm_${CMAKE_MATCH_1})
                set(OPENCV_CUDA_ARCH_BIN "${OPENCV_CUDA_ARCH_BIN} ${CMAKE_MATCH_1}")
                set(OPENCV_CUDA_ARCH_FEATURES "${OPENCV_CUDA_ARCH_FEATURES} ${CMAKE_MATCH_2}")
            else()
                # User didn't explicitly specify PTX for the concrete BIN, we assume PTX=BIN                
                set(NVCC_FLAGS_EXTRA ${NVCC_FLAGS_EXTRA} -gencode arch=compute_${ARCH},code=sm_${ARCH})
                set(OPENCV_CUDA_ARCH_BIN "${OPENCV_CUDA_ARCH_BIN} ${ARCH}")
                set(OPENCV_CUDA_ARCH_FEATURES "${OPENCV_CUDA_ARCH_FEATURES} ${ARCH}")
            endif()
        endforeach()
        
        # Tell NVCC to add PTX intermediate code for the specified architectures
        string(REGEX MATCHALL "[0-9]+" ARCH_LIST "${ARCH_PTX_NO_POINTS}")
        foreach(ARCH IN LISTS ARCH_LIST)
            set(NVCC_FLAGS_EXTRA ${NVCC_FLAGS_EXTRA} -gencode arch=compute_${ARCH},code=compute_${ARCH})
            set(OPENCV_CUDA_ARCH_PTX "${OPENCV_CUDA_ARCH_PTX} ${ARCH}")
            set(OPENCV_CUDA_ARCH_FEATURES "${OPENCV_CUDA_ARCH_FEATURES} ${ARCH}")
        endforeach()
        
        # These vars will be processed in other scripts
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${NVCC_FLAGS_EXTRA})        
        set(OpenCV_CUDA_CC "${NVCC_FLAGS_EXTRA}")
        
        message(STATUS "CUDA NVCC target flags: ${CUDA_NVCC_FLAGS}")
    else()
        unset(CUDA_ARCH_BIN CACHE)
        unset(CUDA_ARCH_PTX CACHE)  
    endif() 
endif()


############################### OpenNI ################################
set(HAVE_OPENNI FALSE)
set(HAVE_OPENNI_PRIME_SENSOR_MODULE FALSE)

if(WITH_OPENNI)
    include(OpenCVFindOpenNI.cmake)
endif()

############################### XIMEA ################################
set(HAVE_XIMEA FALSE)
 
if(WITH_XIMEA)
    include(OpenCVFindXimea.cmake)
endif()

if(XIMEA_FOUND)
    set(HAVE_XIMEA TRUE)
endif()

############################## Eigen ##############################

if(WITH_EIGEN)
    find_path(EIGEN_INCLUDE_PATH "Eigen/Core"
            PATHS "/usr/local/include/eigen2" "/opt/include/eigen2" "/usr/include/eigen2"
            "/usr/local/include/eigen3" "/opt/include/eigen3" "/usr/include/eigen3"
            DOC "The path to Eigen2/Eigen3 headers")
    if(EIGEN_INCLUDE_PATH)
        include_directories(${EIGEN_INCLUDE_PATH})
        set(HAVE_EIGEN 1)
    endif()
endif()


################## Extra HighGUI libs on Windows ###################

if(WIN32)
    set(WITH_VIDEOINPUT ON CACHE BOOL "Build HighGUI with DirectShow support")
    set(HIGHGUI_LIBRARIES ${HIGHGUI_LIBRARIES} comctl32 gdi32 ole32)

    if(WITH_VIDEOINPUT)
        set(HAVE_VIDEOINPUT 1)
    endif()
    
    if (MSVC)
        set(HIGHGUI_LIBRARIES ${HIGHGUI_LIBRARIES} vfw32)
    endif()

    if(MINGW)
        if(MINGW64)
            set(HIGHGUI_LIBRARIES ${HIGHGUI_LIBRARIES} msvfw32 avifil32 avicap32 winmm)
        else()
            set(HIGHGUI_LIBRARIES ${HIGHGUI_LIBRARIES} vfw32 winmm)
        endif()  
    endif()
endif()

############## Android source tree for native camera ###############
if(ANDROID AND ANDROID_API_LEVEL GREATER 7)
    option(WITH_ANDROID_CAMERA "Build with native Android camera support" TRUE)

    SET (ANDROID_SOURCE_TREE "ANDROID_SOURCE_TREE-NOTFOUND" CACHE PATH 
"Path to Android source tree.
Set this variable to path to your Android sources to compile
libnative_camera_rx.x.x.so for your Android")
    SET(BUILD_ANDROID_CAMERA_WRAPPER OFF)
    if (ANDROID_SOURCE_TREE)
        FILE(STRINGS "${ANDROID_SOURCE_TREE}/development/sdk/platform_source.properties" ANDROID_VERSION REGEX "Platform\\.Version=[0-9]+\\.[0-9]+\\.[0-9]+" )
        string(REGEX REPLACE "Platform\\.Version=([0-9]+\\.[0-9]+\\.[0-9]+)" "\\1" ANDROID_VERSION "${ANDROID_VERSION}")
        if(NOT "${ANDROID_VERSION}" STREQUAL "")
            SET(BUILD_ANDROID_CAMERA_WRAPPER ON)
        endif()
    endif()
    MARK_AS_ADVANCED(ANDROID_SOURCE_TREE)
endif()

# ----------------------------------------------------------------------------
#                   UPDATE CONFIG FILES & SCRIPTS:
#
#  CONFIGURE_FILE(InputFile OutputFile [COPYONLY] [ESCAPE_QUOTES] [@ONLY])
#  If @ONLY is specified, only variables of the form @VAR@ will be
#  replaces and ${VAR} will be ignored.
#
#  A directory will be created for each platform so the "cvconfig.h" file is
#  not overwritten if cmake generates code in the same path.
# ----------------------------------------------------------------------------
add_definitions(-DHAVE_CVCONFIG_H)

set(OPENCV_CONFIG_FILE_INCLUDE_DIR "${CMAKE_BINARY_DIR}/" CACHE PATH "Where to create the platform-dependant cvconfig.h")

message(STATUS "Parsing 'cvconfig.h.cmake'")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cvconfig.h.cmake" "${OPENCV_CONFIG_FILE_INCLUDE_DIR}/cvconfig.h")

# ---------------------------------------------------------------------------
# The C+//0 include & link directories:
# ---------------------------------------------------------------------------
include_directories("."
    "${OPENCV_CONFIG_FILE_INCLUDE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/opencv"
    )

# ----------------------------------------------------------------------------
#           Set the maximum level of warnings:
# ----------------------------------------------------------------------------
# Should be set to true for development
set(OPENCV_WARNINGS_ARE_ERRORS OFF CACHE BOOL "Treat warnings as errors")
if (WIN32 AND ${CMAKE_GENERATOR} MATCHES "(MinGW)|(MSYS)")
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG" CACHE STRING "")
endif()

set(EXTRA_C_FLAGS "")
set(EXTRA_C_FLAGS_RELEASE "")
set(EXTRA_C_FLAGS_DEBUG "")
set(EXTRA_EXE_LINKER_FLAGS "")
set(EXTRA_EXE_LINKER_FLAGS_RELEASE "")
set(EXTRA_EXE_LINKER_FLAGS_DEBUG "")

if(MSVC)
    set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} /D _CRT_SECURE_NO_DEPRECATE /D _CRT_NONSTDC_NO_DEPRECATE /D _SCL_SECURE_NO_WARNINGS")
    # 64-bit portability warnings, in MSVC8
    if(MSVC80)
        set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} /Wp64")
    endif()
    #if(MSVC90)
    #    set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} /D _BIND_TO_CURRENT_CRT_VERSION=1 /D _BIND_TO_CURRENT_VCLIBS_VERSION=1")
    #endif()

    if(BUILD_WITH_DEBUG_INFO)
        set(EXTRA_EXE_LINKER_FLAGS_RELEASE "${EXTRA_EXE_LINKER_FLAGS_RELEASE} /debug")
    endif()

    # Remove unreferenced functions: function level linking
    set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} /Gy")
    if(BUILD_WITH_DEBUG_INFO)
       set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} /Zi")
    endif()
endif()

if(CMAKE_COMPILER_IS_GNUCXX)
    # High level of warnings.
    set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -Wall")

    # The -Wno-long-long is required in 64bit systems when including sytem headers.
    if(X86_64)
        set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -Wno-long-long")
    endif()

    # We need pthread's
    if(UNIX AND NOT ANDROID)
        set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -pthread")
    endif()

    if(OPENCV_WARNINGS_ARE_ERRORS)
        set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -Werror")
    endif()

    if(X86 AND NOT MINGW64 AND NOT X86_64 AND NOT APPLE)
        set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -march=i686")
    endif()

    # Other optimizations
    if(USE_OMIT_FRAME_POINTER)
        set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -fomit-frame-pointer")
    endif()
    if(USE_FAST_MATH)
        set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -ffast-math")
    endif()
    if(ENABLE_POWERPC)
        set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -mcpu=G3 -mtune=G5")
    endif()
    if(ENABLE_SSE)
        set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -msse")
    endif()
    if(ENABLE_SSE2)
        set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -msse2")
    endif()
    # SSE3 and further should be disabled under MingW because it generates compiler errors
    if(NOT MINGW)
        if(ENABLE_SSE3)
            set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -msse3")
        endif()
        
        if(${CMAKE_OPENCV_GCC_VERSION_NUM} GREATER 402)
            set(HAVE_GCC43_OR_NEWER 1)
        endif()
        if(${CMAKE_OPENCV_GCC_VERSION_NUM} GREATER 401)
            set(HAVE_GCC42_OR_NEWER 1)
        endif()
        
        if(HAVE_GCC42_OR_NEWER OR APPLE)
            if(ENABLE_SSSE3)
                set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -mssse3")
            endif()
            if(HAVE_GCC43_OR_NEWER)
                if(ENABLE_SSE41)
                    set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -msse4.1")
                endif()
                if(ENABLE_SSE42)
                    set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -msse4.2")
                endif()
            endif()
        endif()
    endif()

    if(X86 OR X86_64)
        if(NOT APPLE AND ${CMAKE_SIZEOF_VOID_P} EQUAL 4)
            set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -mfpmath=387")
        endif()
    endif()

    # Profiling?
    if(ENABLE_PROFILING)
        set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -pg -g")
    elseif(NOT APPLE)
        # Remove unreferenced functions: function level linking
        set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -ffunction-sections")
    endif()

    set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -DNDEBUG")
    set(EXTRA_C_FLAGS_DEBUG "${EXTRA_C_FLAGS_DEBUG} -O0 -DDEBUG -D_DEBUG")
    if(BUILD_WITH_DEBUG_INFO)
       set(EXTRA_C_FLAGS_DEBUG "${EXTRA_C_FLAGS_DEBUG} -ggdb3")
    endif()
endif()

if(MSVC)
    # 64-bit MSVC compiler uses SSE/SSE2 by default
    if(NOT MSVC64)
       if(ENABLE_SSE)
          set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} /arch:SSE")
       endif()
       if(ENABLE_SSE2)
          set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} /arch:SSE2")
       endif()
    endif()
    if(ENABLE_SSE3)
       set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} /arch:SSE3")
    endif()
    if(ENABLE_SSE4_1)
       set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} /arch:SSE4.1")
    endif()
    if (ENABLE_SSE OR ENABLE_SSE2 OR ENABLE_SSE3 OR ENABLE_SSE4_1)
       set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} /Oi")
    endif()
endif()

# Extra link libs if the user selects building static libs:
if(NOT BUILD_SHARED_LIBS AND CMAKE_COMPILER_IS_GNUCXX AND NOT ANDROID)
    # Android does not need these settings because they are already set by toolchain file
    set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} stdc++)
    set(OPENCV_EXTRA_C_FLAGS "-fPIC ${OPENCV_EXTRA_C_FLAGS}")
endif()

# Add user supplied extra options (optimization, etc...)
# ==========================================================
set(OPENCV_EXTRA_C_FLAGS "${OPENCV_EXTRA_C_FLAGS}" CACHE STRING "Extra compiler options")
set(OPENCV_EXTRA_C_FLAGS_RELEASE "${OPENCV_EXTRA_C_FLAGS_RELEASE}" CACHE STRING "Extra compiler options for Release build")
set(OPENCV_EXTRA_C_FLAGS_DEBUG "${OPENCV_EXTRA_C_FLAGS_DEBUG}" CACHE STRING "Extra compiler options for Debug build")
set(OPENCV_EXTRA_EXE_LINKER_FLAGS "${OPENCV_EXTRA_EXE_LINKER_FLAGS}" CACHE STRING "Extra linker flags" FORCE)
set(OPENCV_EXTRA_EXE_LINKER_FLAGS_RELEASE "${OPENCV_EXTRA_EXE_LINKER_FLAGS_RELEASE}" CACHE STRING "Extra linker flags for Release build" FORCE)
set(OPENCV_EXTRA_EXE_LINKER_FLAGS_DEBUG "${OPENCV_EXTRA_EXE_LINKER_FLAGS_DEBUG}" CACHE STRING "Extra linker flags for Debug build" FORCE)

#combine all "extra" options
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${EXTRA_C_FLAGS} ${OPENCV_EXTRA_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_C_FLAGS} ${OPENCV_EXTRA_C_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${EXTRA_C_FLAGS_RELEASE} ${OPENCV_EXTRA_C_FLAGS_RELEASE}")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${EXTRA_C_FLAGS_RELEASE} ${OPENCV_EXTRA_C_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${EXTRA_C_FLAGS_DEBUG} ${OPENCV_EXTRA_C_FLAGS_DEBUG}")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${EXTRA_C_FLAGS_DEBUG} ${OPENCV_EXTRA_C_FLAGS_DEBUG}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${EXTRA_EXE_LINKER_FLAGS} ${OPENCV_EXTRA_EXE_LINKER_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} ${EXTRA_EXE_LINKER_FLAGS_RELEASE} ${OPENCV_EXTRA_EXE_LINKER_FLAGS_RELEASE}")
set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} ${EXTRA_EXE_LINKER_FLAGS_DEBUG} ${OPENCV_EXTRA_EXE_LINKER_FLAGS_DEBUG}")

# In case of Makefiles if the user does not setup CMAKE_BUILD_TYPE, assume it's Release:
if (${CMAKE_GENERATOR} MATCHES ".*Makefiles" AND "${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE Release)
endif()

if (WIN32 AND MSVC)
    # avoid warnings from MSVC about overriding the /W* option
    # we replace /W3 with /W4 only for C++ files,
    # since all the 3rd-party libraries OpenCV uses are in C,
    # and we do not care about their warnings.
    string(REPLACE "/W3" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE "/W3" "/W4" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    string(REPLACE "/W3" "/W4" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")

    # allow extern "C" functions throw exceptions
    string(REPLACE "/EHsc" "/EHsc-" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    string(REPLACE "/EHsc" "/EHsc-" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
    string(REPLACE "/EHsc" "/EHsc-" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
    string(REPLACE "/EHsc" "/EHsc-" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE "/EHsc" "/EHsc-" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    string(REPLACE "/EHsc" "/EHsc-" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")

    string(REPLACE "/Zm1000" " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE "/Zm1000" " " CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
endif()

if("${CMAKE_CONFIGURE_LDFLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_CONFIGURE_LDFLAGS}")
endif("${CMAKE_CONFIGURE_LDFLAGS}")

# --------------------------------------------------------------------------------------------
#  Installation for CMake Module:  OpenCVConfig.cmake
#  Part 1/3: ${BIN_DIR}/OpenCVConfig.cmake              -> For use *without* "make install"
#  Part 2/3: ${BIN_DIR}/unix-install/OpenCVConfig.cmake -> For use with "make install"
#  Part 3/3: ${BIN_DIR}/win-install/OpenCVConfig.cmake  -> For use within binary installers/packages
# -------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------
#  Part 1/3: ${BIN_DIR}/OpenCVConfig.cmake              -> For use *without* "make install"
# -------------------------------------------------------------------------------------------
set(CMAKE_INCLUDE_DIRS_CONFIGCMAKE "\"${OPENCV_CONFIG_FILE_INCLUDE_DIR}\" \"${CMAKE_CURRENT_SOURCE_DIR}/include\" \"${CMAKE_CURRENT_SOURCE_DIR}/include/opencv\"")
set(CMAKE_OPENCV2_INCLUDE_DIRS_CONFIGCMAKE "${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_LIB_DIRS_CONFIGCMAKE "${LIBRARY_OUTPUT_PATH}")
set(CMAKE_3RDPARTY_LIB_DIRS_CONFIGCMAKE "\"${CMAKE_BINARY_DIR}/3rdparty/${OPENCV_LIB_INSTALL_PATH}\"")

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/OpenCVConfig.cmake.in" "${CMAKE_BINARY_DIR}/OpenCVConfig.cmake" IMMEDIATE @ONLY)
#support for version checking when finding opencv. find_package(OpenCV 2.3.1 EXACT) should now work.
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/OpenCVConfig-version.cmake.in" "${CMAKE_BINARY_DIR}/OpenCVConfig-version.cmake" IMMEDIATE @ONLY)
# --------------------------------------------------------------------------------------------
#  Part 2/3: ${BIN_DIR}/unix-install/OpenCVConfig.cmake -> For use *with* "make install"
# -------------------------------------------------------------------------------------------
set(CMAKE_INCLUDE_DIRS_CONFIGCMAKE "\"\${OpenCV_INSTALL_PATH}/${OPENCV_INCLUDE_PREFIX}/opencv" "\${OpenCV_INSTALL_PATH}/${OPENCV_INCLUDE_PREFIX}\"")

set(CMAKE_OPENCV2_INCLUDE_DIRS_CONFIGCMAKE "")
if(ANDROID)
    set(CMAKE_LIB_DIRS_CONFIGCMAKE "\"\${OpenCV_INSTALL_PATH}/libs/\${ARMEABI_NDK_NAME}\"")
    set(CMAKE_3RDPARTY_LIB_DIRS_CONFIGCMAKE "\"\${OpenCV_INSTALL_PATH}/share/OpenCV/3rdparty/libs/\${ARMEABI_NDK_NAME}\"")
else()
    set(CMAKE_LIB_DIRS_CONFIGCMAKE "\"\${OpenCV_INSTALL_PATH}/${OPENCV_LIB_INSTALL_PATH}\"")
    set(CMAKE_3RDPARTY_LIB_DIRS_CONFIGCMAKE "\"\${OpenCV_INSTALL_PATH}/share/OpenCV/3rdparty/${OPENCV_LIB_INSTALL_PATH}\"")
    if(OPENCV_MANGLED_INSTALL_PATHS)
      set(CMAKE_3RDPARTY_LIB_DIRS_CONFIGCMAKE "\"\${OpenCV_INSTALL_PATH}/share/OpenCV-${OPENCV_VERSION}/3rdparty/${OPENCV_LIB_INSTALL_PATH}\"")
    endif()
endif()

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/OpenCVConfig.cmake.in" "${CMAKE_BINARY_DIR}/unix-install/OpenCVConfig.cmake" IMMEDIATE @ONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/OpenCVConfig-version.cmake.in" "${CMAKE_BINARY_DIR}/unix-install/OpenCVConfig-version.cmake" IMMEDIATE @ONLY)

if(UNIX)
  #http://www.vtk.org/Wiki/CMake/Tutorials/Packaging reference
  # For a command "find_package(<name> [major[.minor]] [EXACT] [REQUIRED|QUIET])"
  # cmake will look in the following dir on unix:
  #                <prefix>/(share|lib)/cmake/<name>*/                     (U)
  #                <prefix>/(share|lib)/<name>*/                           (U)
  #                <prefix>/(share|lib)/<name>*/(cmake|CMake)/             (U)
  if(OPENCV_MANGLED_INSTALL_PATHS)
    install(FILES ${CMAKE_BINARY_DIR}/unix-install/OpenCVConfig.cmake DESTINATION share/OpenCV-${OPENCV_VERSION}/)
    install(FILES ${CMAKE_BINARY_DIR}/unix-install/OpenCVConfig-version.cmake DESTINATION share/OpenCV-${OPENCV_VERSION}/)
  else()
    install(FILES "${CMAKE_BINARY_DIR}/unix-install/OpenCVConfig.cmake" DESTINATION share/OpenCV/)
    install(FILES ${CMAKE_BINARY_DIR}/unix-install/OpenCVConfig-version.cmake DESTINATION share/OpenCV/)
  endif()
endif()

if(ANDROID)
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/android/android.toolchain.cmake" DESTINATION share/OpenCV)
endif()

# --------------------------------------------------------------------------------------------
#  Part 3/3: ${BIN_DIR}/win-install/OpenCVConfig.cmake  -> For use within binary installers/packages
# -------------------------------------------------------------------------------------------
if(WIN32)
    set(CMAKE_INCLUDE_DIRS_CONFIGCMAKE "\"\${OpenCV_CONFIG_PATH}/include\" \"\${OpenCV_CONFIG_PATH}/include/opencv\"")
    set(CMAKE_OPENCV2_INCLUDE_DIRS_CONFIGCMAKE "")
    set(CMAKE_LIB_DIRS_CONFIGCMAKE "\"\${OpenCV_CONFIG_PATH}/${OPENCV_LIB_INSTALL_PATH}\"")
    set(CMAKE_3RDPARTY_LIB_DIRS_CONFIGCMAKE "\"\${OpenCV_CONFIG_PATH}/share/OpenCV/3rdparty/${OPENCV_LIB_INSTALL_PATH}\"")

    exec_program(mkdir ARGS "-p \"${CMAKE_BINARY_DIR}/win-install/\"" OUTPUT_VARIABLE RET_VAL)
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/OpenCVConfig.cmake.in" "${CMAKE_BINARY_DIR}/win-install/OpenCVConfig.cmake" IMMEDIATE @ONLY)

    # Install the OpenCVConfig.cmake file which has the right paths pointing to the install directory 
    install(FILES "${CMAKE_BINARY_DIR}/win-install/OpenCVConfig.cmake" DESTINATION "${CMAKE_INSTALL_PREFIX}/") 
endif()

# --------------------------------------------------------------------------------------------
#  Installation for Android ndk-build makefile:  OpenCV.mk
#  Part 1/2: ${BIN_DIR}/OpenCV.mk              -> For use *without* "make install"
#  Part 2/2: ${BIN_DIR}/unix-install/OpenCV.mk -> For use with "make install"
# -------------------------------------------------------------------------------------------
if(ANDROID)
    if(BUILD_SHARED_LIBS)
        SET(OPENCV_LIBTYPE_CONFIGMAKE "SHARED")
    else()
        SET(OPENCV_LIBTYPE_CONFIGMAKE "STATIC")
    endif()
    if(BUILD_ANDROID_CAMERA_WRAPPER)
        set(CMAKE_CAMERA_LIBS_CONFIGCMAKE "native_camera_r${ANDROID_VERSION}")
    elseif(WITH_ANDROID_CAMERA)
        SET(CMAKE_CAMERA_LIBS_CONFIGCMAKE "")
        file(GLOB CMAKE_CAMERA_LIBS "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/lib/${ARMEABI_NDK_NAME}/libnative_camera_r*.so")
        foreach(cam_lib ${CMAKE_CAMERA_LIBS})
            get_filename_component(cam_lib "${cam_lib}" NAME)
            string(REGEX REPLACE "lib(native_camera_r[0-9]+\\.[0-9]+\\.[0-9]+)\\.so" "\\1" cam_lib "${cam_lib}")
            SET(CMAKE_CAMERA_LIBS_CONFIGCMAKE "${CMAKE_CAMERA_LIBS_CONFIGCMAKE} ${cam_lib}")
        endforeach()
    endif()
endif(ANDROID)

# -------------------------------------------------------------------------------------------
#  Part 1/2: ${BIN_DIR}/OpenCV.mk              -> For use *without* "make install"
# -------------------------------------------------------------------------------------------
if(ANDROID)
    set(CMAKE_INCLUDE_DIRS_CONFIGCMAKE "\"${OPENCV_CONFIG_FILE_INCLUDE_DIR}\" \"${CMAKE_CURRENT_SOURCE_DIR}/include\" \"${CMAKE_CURRENT_SOURCE_DIR}/include/opencv\"")
    set(CMAKE_BASE_INCLUDE_DIR_CONFIGCMAKE "\"${CMAKE_CURRENT_SOURCE_DIR}\"")
    set(CMAKE_LIBS_DIR_CONFIGCMAKE "\$(OPENCV_THIS_DIR)")

    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/OpenCV.mk.in" "${CMAKE_BINARY_DIR}/OpenCV.mk" IMMEDIATE @ONLY)
endif(ANDROID)

# -------------------------------------------------------------------------------------------
#  Part 2/2: ${BIN_DIR}/unix-install/OpenCV.mk -> For use with "make install"
# -------------------------------------------------------------------------------------------
if(ANDROID)
    set(CMAKE_INCLUDE_DIRS_CONFIGCMAKE "\"\$(LOCAL_PATH)/\$(OPENCV_THIS_DIR)/../../include/opencv\" \"\$(LOCAL_PATH)/\$(OPENCV_THIS_DIR)/../../include\"")
    set(CMAKE_BASE_INCLUDE_DIR_CONFIGCMAKE "")
    set(CMAKE_LIBS_DIR_CONFIGCMAKE "\$(OPENCV_THIS_DIR)/../..")

    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/OpenCV.mk.in" "${CMAKE_BINARY_DIR}/unix-install/OpenCV.mk" IMMEDIATE @ONLY)
    install(FILES ${CMAKE_BINARY_DIR}/unix-install/OpenCV.mk DESTINATION share/OpenCV/)
endif(ANDROID)

# --------------------------------------------------------------------------------------------
#according to man pkg-config
#  The package name specified on the pkg-config command line is defined to
#      be the name of the metadata file, minus the .pc extension. If a library
#      can install multiple versions simultaneously, it must give each version
#      its own name (for example, GTK 1.2 might have the package  name  "gtk+"
#      while GTK 2.0 has "gtk+-2.0").
#  Part 2/2: ${BIN_DIR}/unix-install/opencv.pc -> For use *with* "make install"
# prefix=/usr
# exec_prefix=${prefix}
# libdir=${exec_prefix}/lib
# includedir=${prefix}/include/opencv
# -------------------------------------------------------------------------------------------
set(prefix      ${CMAKE_INSTALL_PREFIX})
set(exec_prefix "\${prefix}")
set(libdir      "\${exec_prefix}/lib${LIB_SUFFIX}")
set(includedir  "\${prefix}/${OPENCV_INCLUDE_PREFIX}")
set(VERSION     ${OPENCV_VERSION})

set(OPENCV_PC_FILE_NAME opencv.pc)
#need to be explicit with naming the pc file and version number for side by side installs to work.
if(OPENCV_MANGLED_INSTALL_PATHS)
  SET(OpenCV_LIB_COMPONENTS opencv_contrib opencv_legacy opencv_objdetect opencv_calib3d opencv_features2d opencv_video opencv_highgui opencv_ml opencv_imgproc opencv_flann opencv_core )
  #be explicit about the library names.
  set(OpenCV_LIB_COMPONENTS_ )
  foreach( CVLib ${OpenCV_LIB_COMPONENTS})
    set(OpenCV_LIB_COMPONENTS_ "${OpenCV_LIB_COMPONENTS_} ${libdir}/lib${CVLib}.so.${OPENCV_VERSION}" )
  endforeach()
  set(OpenCV_LIB_COMPONENTS ${OpenCV_LIB_COMPONENTS_})
  set(OPENCV_PC_FILE_NAME "opencv-${OPENCV_VERSION}.pc")
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/opencv-XXX.pc.cmake.in" "${CMAKE_BINARY_DIR}/unix-install/${OPENCV_PC_FILE_NAME}" @ONLY IMMEDIATE)
else()
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/opencv.pc.cmake.in" "${CMAKE_BINARY_DIR}/unix-install/${OPENCV_PC_FILE_NAME}" @ONLY IMMEDIATE)
endif()

if(UNIX AND NOT ANDROID)
  install(FILES ${CMAKE_BINARY_DIR}/unix-install/${OPENCV_PC_FILE_NAME} DESTINATION ${OPENCV_LIB_INSTALL_PATH}/pkgconfig)
endif()

# ----------------------------------------------------------------------------
#   Uninstall target, for "make uninstall"
# ----------------------------------------------------------------------------
CONFIGURE_FILE(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
  IMMEDIATE @ONLY)

ADD_CUSTOM_TARGET(uninstall "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake")

#-----------------------------------
# Source package:
#-----------------------------------
set(BUILD_PACKAGE ON CACHE BOOL "Enables 'make package_source' command")

if(BUILD_PACKAGE)
    set(TARBALL_NAME "${CMAKE_PROJECT_NAME}-${OPENCV_VERSION_MAJOR}.${OPENCV_VERSION_MINOR}.${OPENCV_VERSION_PATCH}")
    if (NOT WIN32)
        add_custom_target(package_source
        #TODO: maybe we should not remove dll's
        COMMAND tar --transform 's,^,${TARBALL_NAME}/,S' -cjpf ${CMAKE_CURRENT_BINARY_DIR}/${TARBALL_NAME}.tar.bz2 --exclude-vcs --exclude="*.pyc" --exclude="*.vcproj" --exclude="*/lib/*" --exclude="*.dll" ./
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    else()
        add_custom_target(package_source
        COMMAND zip -9 -r ${CMAKE_CURRENT_BINARY_DIR}/${TARBALL_NAME}.zip . -x '*/.svn/*' '*.vcproj' '*.pyc'
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    endif()
endif()


#-----------------------------------
# Solution folders:
#-----------------------------------

if(${CMAKE_VERSION} VERSION_GREATER "2.8.0")
    set(ENABLE_SOLUTION_FOLDERS OFF CACHE BOOL "Solution folder in Visual Studio or in other IDEs")
endif()

if(ENABLE_SOLUTION_FOLDERS)
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)
    set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "CMakeTargets")
endif()

#-----------------------------------
# Subdirectories:
#-----------------------------------
add_subdirectory(include)
add_subdirectory(modules)
add_subdirectory(doc)
add_subdirectory(data)
add_subdirectory(3rdparty)

if(BUILD_EXAMPLES OR BUILD_ANDROID_EXAMPLES OR INSTALL_PYTHON_EXAMPLES)
    add_subdirectory(samples)
endif()

# ----------------------------------------------------------------------------
#   Summary:
# ----------------------------------------------------------------------------
macro(status text)
    SET(status_cond)
    SET(status_then)
    SET(status_else)
    
    SET(status_current_name "cond")
    foreach(arg ${ARGN})
       if(arg STREQUAL "THEN")
           SET(status_current_name "then")
       elseif(arg STREQUAL "ELSE")
           SET(status_current_name "else")
       else()
           LIST(APPEND status_${status_current_name} ${arg})
       endif()
    endforeach()

    if(DEFINED status_cond)
        SET(status_placeholder_length 32)
        string(RANDOM LENGTH ${status_placeholder_length} ALPHABET " " status_placeholder)
        string(LENGTH "${text}" status_text_length)
        if (status_text_length LESS status_placeholder_length)
            string(SUBSTRING "${text}${status_placeholder}" 0 ${status_placeholder_length} status_text)
        elseif (DEFINED status_then OR DEFINED status_else)
            message(STATUS "${text}")
            SET(status_text "${status_placeholder}")
        else()
            SET(status_text "${text}")
        endif()

        if (DEFINED status_then OR DEFINED status_else)
            if(${status_cond})
                string(REPLACE ";" " " status_then "${status_then}")
                message(STATUS "${status_text}" "${status_then}")
            else()
                string(REPLACE ";" " " status_else "${status_else}")
                message(STATUS "${status_text}" "${status_else}")
            endif()
        else()
            string(REPLACE ";" " " status_cond "${status_cond}")
            message(STATUS "${status_text}" "${status_cond}")
        endif()
     else()
         message(STATUS "${text}")
     endif()
endmacro()

status("")
status("General configuration for opencv ${OPENCV_VERSION} =====================================")
status("")
status("    Built as dynamic libs?:" BUILD_SHARED_LIBS THEN YES ELSE NO)
status("    Compiler:"               CMAKE_COMPILER THEN "${CMAKE_COMPILER}" ELSE "${CMAKE_CXX_COMPILER}")
status("    C++ flags (Release):"    ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE})
status("    C++ flags (Debug):"      ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG})
if(WIN32)
    status("    Linker flags (Release):" ${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS_RELEASE})
    status("    Linker flags (Debug):"   ${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS_DEBUG})
else()
    status("    Linker flags (Release):" ${CMAKE_SHARED_LINKER_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS_RELEASE})
    status("    Linker flags (Debug):"   ${CMAKE_SHARED_LINKER_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS_DEBUG})
endif()

if(ANDROID)
    status("    Floating point type:" ${ARM_TARGET})
    status("    Native API level:" android-${ANDROID_API_LEVEL})
    status("    SDK target:" "${ANDROID_SDK_TARGET}")
endif()

#YV
status("")
status("  GUI: ")

if (HAVE_QT)
    status("    QT 4.x:"            HAVE_QT        THEN YES ELSE NO)
    status("    QT OpenGL support:" HAVE_QT_OPENGL THEN YES ELSE NO)
else()
    if(WIN32)
        status("    Win32 UI:" YES)
    else()
        if(APPLE)
            if(WITH_CARBON)
                status("    Carbon:" YES)
            else()
                status("    Cocoa:"  YES)
            endif()
        else()
            status("    GTK+ 2.x:" HAVE_GTK     THEN YES ELSE NO)
            status("    GThread:"  HAVE_GTHREAD THEN YES ELSE NO)
        endif()
    endif()
endif()

# media
status("")
status("  Media I/O: ")
status("    ZLib:"      ZLIB_FOUND                         THEN YES             ELSE build)
status("    JPEG:"      NOT WITH_JPEG OR JPEG_FOUND        THEN ${JPEG_FOUND}   ELSE build)
status("    PNG:"       NOT WITH_PNG OR PNG_FOUND          THEN ${PNG_FOUND}    ELSE build)
status("    TIFF:"      NOT WITH_TIFF OR TIFF_FOUND        THEN ${TIFF_FOUND}   ELSE build)
status("    JPEG 2000:" NOT WITH_JASPER OR JASPER_FOUND    THEN ${JASPER_FOUND} ELSE build)
status("    OpenEXR:"   WITH_OPENEXR AND OPENEXR_FOUND     THEN YES             ELSE NO)

status("    OpenNI:"    HAVE_OPENNI                        THEN YES             ELSE NO)
status("    OpenNI PrimeSensor Modules:"
                        HAVE_OPENNI_PRIME_SENSOR_MODULE    THEN YES             ELSE NO)
status("    XIMEA:"     HAVE_XIMEA  THEN YES ELSE NO)

# video
status("")
if(UNIX AND NOT APPLE)
    status("  Video I/O:")
    status("    DC1394 1.x:"     HAVE_DC1394         THEN YES ELSE NO)
    status("    DC1394 2.x:"     HAVE_DC1394_2       THEN YES ELSE NO)
    status("    FFMPEG:"         HAVE_FFMPEG         THEN YES ELSE NO)
    status("      codec:"        HAVE_FFMPEG_CODEC   THEN YES ELSE NO)
    status("      format:"       HAVE_FFMPEG_FORMAT  THEN YES ELSE NO)
    status("      util:"         HAVE_FFMPEG_UTIL    THEN YES ELSE NO)
    status("      swscale:"      HAVE_FFMPEG_SWSCALE THEN YES ELSE NO)
    status("      gentoo-style:" HAVE_GENTOO_FFMPEG  THEN YES ELSE NO)
    status("    GStreamer:"      HAVE_GSTREAMER      THEN YES ELSE NO)
    status("    UniCap:"         HAVE_UNICAP         THEN YES ELSE NO)
    status("    PvAPI:"          HAVE_PVAPI          THEN YES ELSE NO)
    status("    V4L/V4L2:"       HAVE_LIBV4L         THEN Using libv4l ELSE ${HAVE_CAMV4L}/${HAVE_CAMV4L2})
    status("    Xine:"           HAVE_XINE           THEN YES ELSE NO)

    if(ANDROID)
        if(WITH_ANDROID_CAMERA)
            status("    AndroidNativeCamera:" BUILD_ANDROID_CAMERA_WRAPPER THEN "build for Android ${ANDROID_VERSION}" ELSE "use prebuilt libraries")
        else()
            status("    AndroidNativeCamera:" "NO (native camera requires Android API level 8 or higher)")
        endif()
    endif()
elseif(APPLE)
    status("  Video I/O:"        WITH_QUICKTIME      THEN QuickTime ELSE QTKit)
elseif(WIN32)
    status("  Video I/O:"        HAVE_VIDEOINPUT     THEN DirectShow ELSE NO)
endif()

# Other third-party libraries
status("")
status("  Other third-party libraries:")

if(WITH_IPP AND IPP_FOUND)
    status("    Use IPP:" "${IPP_LATEST_VERSION_STR} [${IPP_LATEST_VERSION_MAJOR}.${IPP_LATEST_VERSION_MINOR}.${IPP_LATEST_VERSION_BUILD}]")
    status("         at:" "${IPP_ROOT_DIR}")
else()
    status("    Use IPP:"   WITH_IPP AND NOT IPP_FOUND THEN IPP not found ELSE NO)
endif()

status("    Use TBB:" HAVE_TBB THEN YES ELSE NO)

if(UNIX)
    status("    Use ThreadingFramework:" HAVE_THREADING_FRAMEWORK THEN YES ELSE NO)
endif()

status("    Use Cuda:"  HAVE_CUDA  THEN YES ELSE NO)
status("    Use Eigen:" HAVE_EIGEN THEN YES ELSE NO)

# interfaces to other languages
status("")
status("  Interfaces:")
status("    Python:"              BUILD_NEW_PYTHON_SUPPORT THEN YES ELSE NO)
status("    Python interpreter:"  PYTHON_EXECUTABLE        THEN "${PYTHON_EXECUTABLE} (ver ${PYTHON_VERSION_MAJOR_MINOR})" ELSE NO)
status("    Python numpy:"        PYTHON_USE_NUMPY         THEN YES ELSE "NO (Python wrappers will not be generated)")
if(ANDROID AND ANDROID_API_LEVEL LESS 8)
    status("    Java:" "NO (Java API requires Android API level 8 or higher)")
else()
    status("    Java:" BUILD_JAVA_SUPPORT THEN YES ELSE NO)
endif()

if(ANDROID)
    status("    android tool:"    ANDROID_EXECUTABLE       THEN "${ANDROID_EXECUTABLE} (${ANDROID_TOOLS_Pkg_Desc})" ELSE NO)
    status("    ant:"             ANT_EXECUTABLE           THEN "${ANT_EXECUTABLE} (ver ${ANT_VERSION})" ELSE NO)
endif()

# documentation
status("")
status("  Documentation:")
status("    Sphinx:"              HAVE_SPHINX              THEN "${SPHINX_BUILD} (ver ${SPHINX_VERSION})" ELSE NO)
status("    PdfLaTeX compiler:"   PDFLATEX_COMPILER        THEN "${PDFLATEX_COMPILER}" ELSE NO)
if (BUILD_DOCS AND HAVE_SPHINX)
    status("    Build Documentation:" PDFLATEX_COMPILER THEN YES ELSE "YES (only HTML without math formulas)")
else()
    status("    Build Documentation:" NO)
endif()

# samples and tests
status("")
status("  Tests and samples:")
status("    Tests:"    BUILD_TESTS     THEN YES ELSE NO)
status("    Examples:" BUILD_EXAMPLES  THEN YES ELSE NO)

if(ANDROID)
   status("    Android tests:" BUILD_TESTS AND CAN_BUILD_ANDROID_PROJECTS THEN YES ELSE NO)
   status("    Android examples:" BUILD_ANDROID_EXAMPLES THEN YES ELSE NO)
endif()

# auxiliary
status("")
status("  Install path:" "${CMAKE_INSTALL_PREFIX}")
status("")
status("  cvconfig.h is in:" "${OPENCV_CONFIG_FILE_INCLUDE_DIR}")
status("-----------------------------------------------------------------")
status("")

# warn in the case of in-source build
if("${CMAKE_CURRENT_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_BINARY_DIR}")
    message(WARNING "The source directory is the same as binary directory. \"make clean\" may damage the source tree")
endif()
