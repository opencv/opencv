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
# Add these standard paths to the search paths for FIND_LIBRARY
# to find libraries from these locations first
if(UNIX)
  set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} /lib /usr/lib)
endif()
# it _must_ go before PROJECT(OpenCV) in order to work
if (NOT CMAKE_INSTALL_PREFIX)
    if (WIN32)
        set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR} CACHE INTERNAL "" FORCE)
    elseif()
        set(CMAKE_INSTALL_PREFIX "/usr" CACHE INTERNAL "" FORCE)
    endif()
endif()

SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Configs" FORCE)
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

if (CMAKE_VERBOSE)
        set (CMAKE_VERBOSE_MAKEFILE 1)
endif()

project(OpenCV)

cmake_minimum_required(VERSION 2.4)

if(MSVC)
    set(CMAKE_USE_RELATIVE_PATHS ON CACHE INTERNAL "" FORCE)
endif()

#set(CMAKE_C_COMPILER "/opt/BullseyeCoverage/bin/gcc")
#set(CMAKE_CXX_COMPILER "/opt/BullseyeCoverage/bin/g++")
#set(CMAKE_CXX_COMPILER_INIT "/opt/BullseyeCoverage/bin/gcc")

# --------------------------------------------------------------
# Indicate CMake 2.7 and above that we don't want to mix relative
#  and absolute paths in linker lib lists.
# Run "cmake --help-policy CMP0003" for more information.
# --------------------------------------------------------------
if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif()

# ----------------------------------------------------------------------------
#  Current version number:
# ----------------------------------------------------------------------------
set(OPENCV_VERSION "2.1.1")

string(REGEX MATCHALL "[0-9]" OPENCV_VERSION_PARTS "${OPENCV_VERSION}")

list(GET OPENCV_VERSION_PARTS 0 OPENCV_VERSION_MAJOR)
list(GET OPENCV_VERSION_PARTS 1 OPENCV_VERSION_MINOR)
list(GET OPENCV_VERSION_PARTS 2 OPENCV_VERSION_PATCH)

set(OPENCV_SOVERSION "${OPENCV_VERSION_MAJOR}.${OPENCV_VERSION_MINOR}")

if(WIN32)
    # Postfix of DLLs:
    set(OPENCV_DLLVERSION "${OPENCV_VERSION_MAJOR}${OPENCV_VERSION_MINOR}${OPENCV_VERSION_PATCH}")
    set(OPENCV_DEBUG_POSTFIX d)
else()
    # Postfix of so's:
    #set(OPENCV_DLLVERSION "${OPENCV_VERSION_MAJOR}${OPENCV_VERSION_MINOR}${OPENCV_VERSION_PATCH}")
    set(OPENCV_DLLVERSION "")
    set(OPENCV_DEBUG_POSTFIX)
endif()


# ----------------------------------------------------------------------------
# Build static or dynamic libs?
# ----------------------------------------------------------------------------
# Default: dynamic libraries:
SET(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries (.dll/.so) instead of static ones (.lib/.a)")
IF(BUILD_SHARED_LIBS)
    SET(OPENCV_BUILD_SHARED_LIB 1) # For cvconfig.h, etc.
ELSE(BUILD_SHARED_LIBS)
    SET(OPENCV_BUILD_SHARED_LIB 0)
ENDIF(BUILD_SHARED_LIBS)

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
find_program(SVNVERSION_PATH svnversion)
mark_as_advanced(force SVNVERSION_PATH)
if(SVNVERSION_PATH)
    message(STATUS "Extracting svn version, please wait...")
    execute_process(COMMAND ${SVNVERSION_PATH} -n ${CMAKE_SOURCE_DIR} OUTPUT_VARIABLE SVNVERSION_RESULT)

    if(SVNVERSION_RESULT MATCHES "exported")
       # This is NOT a svn repository:
       set(OPENCV_SVNVERSION "")
    else()
        set(OPENCV_SVNVERSION " svn:${SVNVERSION_RESULT}")
    endif()

    message(STATUS "SVNVERSION: ${OPENCV_SVNVERSION}")
else()
    # We don't have svnversion:
    set(OPENCV_SVNVERSION "")
endif()

# ----------------------------------------------------------------------------
# Detect GNU version:
# ----------------------------------------------------------------------------
if(CMAKE_COMPILER_IS_GNUCXX)
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} --version
                  OUTPUT_VARIABLE CMAKE_OPENCV_GCC_VERSION_FULL
                  OUTPUT_STRIP_TRAILING_WHITESPACE)

    # Typical output in CMAKE_OPENCV_GCC_VERSION_FULL: "c+//0 (whatever) 4.2.3 (...)"
    #  Look for the version number
    string(REGEX MATCH "[0-9].[0-9].[0-9]" CMAKE_GCC_REGEX_VERSION "${CMAKE_OPENCV_GCC_VERSION_FULL}")

    # Split the three parts:
    string(REGEX MATCHALL "[0-9]" CMAKE_OPENCV_GCC_VERSIONS "${CMAKE_GCC_REGEX_VERSION}")

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

# Build tests:
# ===================================================
set(BUILD_TESTS ON CACHE BOOL "Build tests")

# Build 3rdparty libraries under unix
# ===================================================
if(WIN32 OR APPLE)
    set(OPENCV_BUILD_3RDPARTY_LIBS TRUE  CACHE BOOL "Build 3rd party libraries")
else()
    set(OPENCV_BUILD_3RDPARTY_LIBS FALSE CACHE BOOL "Build 3rd party libraries")
endif()

include(OpenCVPCHSupport.cmake REQUIRED)
include(OpenCVModule.cmake REQUIRED)

if(UNIX)
    include(OpenCVFindPkgConfig.cmake OPTIONAL)
    include(CheckFunctionExists)
    include(CheckIncludeFile)
endif()

#if(MSVC)
#   set(DEFAULT_ENABLE_OPENMP ON)
#else()
#   set(DEFAULT_ENABLE_OPENMP OFF)
#endif()
#set(ENABLE_OPENMP ${DEFAULT_ENABLE_OPENMP} CACHE BOOL "")

if(CMAKE_COMPILER_IS_GNUCXX)
    set(ENABLE_PROFILING OFF CACHE BOOL "Enable profiling in the GCC compiler (Add flags: -g -pg)")
    set(USE_OMIT_FRAME_POINTER ON CACHE BOOL "Enable -fomit-frame-pointer for GCC")
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES arm*)
        # We can use only -O2 because the -O3 causes gcc crash
        set(USE_O2 ON CACHE BOOL "Enable -O2 for GCC")
        set(USE_FAST_MATH ON CACHE BOOL "Enable -ffast-math for GCC")
    endif()

    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES amd64*)
        set(X86_64 1)
    endif()
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES x86_64*)
        set(X86_64 1)
    endif()

    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES i686*)
        set(X86 1)
    endif()
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES i386*)
        set(X86 1)
    endif()
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES x86*)
        set(X86 1)
    endif()

    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES powerpc*)
        set(USE_O3 ON CACHE BOOL "Enable -O3 for GCC")
        set(ENABLE_POWERPC ON CACHE BOOL "Enable PowerPC for GCC")
    endif ()

    if(X86 OR X86_64)
        # enable everything, since the available set of instructions is checked at runtime
        set(USE_O3 ON CACHE BOOL "Enable -O3 for GCC")
        set(USE_FAST_MATH ON CACHE BOOL "Enable -ffast-math for GCC")
        set(ENABLE_SSE ON CACHE BOOL "Enable SSE for GCC")
        set(ENABLE_SSE2 ON CACHE BOOL "Enable SSE2 for GCC")
        set(ENABLE_SSE3 OFF CACHE BOOL "Enable SSE3 for GCC")
        set(ENABLE_SSSE3 OFF CACHE BOOL "Enable SSSE3 for GCC")
        #set(ENABLE_SSE4_1 OFF CACHE BOOL "Enable SSE4.1 for GCC")
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

set(WITH_TBB OFF CACHE BOOL "Include TBB support")
set(WITH_EIGEN2 ON CACHE BOOL "Include Eigen2/Eigen3 support")
set(WITH_CUDA OFF CACHE BOOL "Include NVidia Cuda Runtime support")

if(WIN32)
        set(WITH_VIDEOINPUT ON CACHE BOOL "Enable VideoInput support")
endif()

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
        CHECK_MODULE(gstreamer-app-0.10 HAVE_GSTREAMER_APP)
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

        if(NOT OPENCV_BUILD_3RDPARTY_LIBS)
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

        if(${CMAKE_SYSTEM_NAME} MATCHES "FreeBSD")
            set(OPENCV_LINKER_LIBS m pthread)
        else()
            set(OPENCV_LINKER_LIBS dl m pthread rt)
        endif()
    else()
        add_definitions(-DHAVE_ALLOCA -DHAVE_ALLOCA_H -DHAVE_LIBPTHREAD -DHAVE_UNISTD_H)
    endif()
endif()

if(WITH_OPENEXR)
    include(OpenCVFindOpenEXR.cmake)
endif()

set(BUILD_NEW_PYTHON_SUPPORT ON CACHE BOOL "Build with Python support")

if(BUILD_NEW_PYTHON_SUPPORT)
    find_package(PythonInterp)
    find_package(PythonLibs)

    # cmake 2.4 (at least on Ubuntu 8.04 (hardy)) don't define PYTHONLIBS_FOUND
    if(NOT PYTHONLIBS_FOUND AND PYTHON_INCLUDE_PATH)
        set(PYTHONLIBS_FOUND ON)
    endif(NOT PYTHONLIBS_FOUND AND PYTHON_INCLUDE_PATH)

    execute_process(COMMAND ${PYTHON_EXECUTABLE} --version
              ERROR_VARIABLE PYTHON_VERSION_FULL
              OUTPUT_STRIP_TRAILING_WHITESPACE)

    string(REGEX MATCH "[0-9].[0-9]" PYTHON_VERSION_MAJOR_MINOR "${PYTHON_VERSION_FULL}")
    if(UNIX)
        set(PYTHON_PLUGIN_INSTALL_PATH lib/python${PYTHON_VERSION_MAJOR_MINOR}/site-packages/opencv)
        set(PYTHON_PACKAGES_PATH lib/python${PYTHON_VERSION_MAJOR_MINOR}/site-packages)
    endif()
    if(WIN32)
        get_filename_component(PYTHON_PATH "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Python\\PythonCore\\${PYTHON_VERSION_MAJOR_MINOR}\\InstallPath]" ABSOLUTE CACHE)
        set(PYTHON_PLUGIN_INSTALL_PATH "${PYTHON_PATH}/Lib/site-packages/opencv")
        set(PYTHON_PACKAGES_PATH "${PYTHON_PATH}/Lib/site-packages")
    endif()

    # Attempt to discover the NumPy include directory.  If this succeeds, build with NumPy

    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import numpy.distutils; print numpy.distutils.misc_util.get_numpy_include_dirs()[0]"
                    RESULT_VARIABLE PYTHON_NUMPY_PROCESS
                    OUTPUT_VARIABLE PYTHON_NUMPY_INCLUDE_DIRS
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(PYTHON_NUMPY_PROCESS EQUAL 0)
      set(PYTHON_USE_NUMPY 1)
      add_definitions(-D PYTHON_USE_NUMPY=1)
      include_directories(AFTER ${PYTHON_NUMPY_INCLUDE_DIRS})
      message(STATUS "    Use INCLUDE: ${PYTHON_NUMPY_INCLUDE_DIRS}")
    else()
      set(PYTHON_USE_NUMPY 0)
    endif()


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
    if (UNIX AND NOT APPLE)
        PKG_CHECK_MODULES(TBB tbb)
        message(STATUS "TBB detected: ${TBBLIB_FOUND}")

        if (TBB_FOUND)
            set(HAVE_TBB 1)
            if(NOT "${TBB_INCLUDE_DIRS}" STREQUAL "")
                include_directories("${TBB_INCLUDE_DIRS}")
            endif()
            link_directories("${TBB_LIBRARY_DIRS}")
            set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} ${TBB_LIBRARIES})
        else()
            set(TBB_DEFAULT_INCLUDE_DIRS
              "/opt/intel/tbb"
              "/usr/local/include"
              "/usr/include")
        endif()
    endif()
    if (APPLE)
        set(TBB_DEFAULT_INCLUDE_DIRS
          "/usr/local/include"
          "/usr/include")
    endif()
    if (WIN32)
        set(TBB_DEFAULT_INCLUDE_DIRS
          "C:/Program Files/Intel/TBB"
          "C:/Program Files (x86)/Intel/TBB")
    endif()
    if (NOT HAVE_TBB)
        find_path(TBB_INCLUDE_DIR "tbb/tbb.h"
                  PATHS TBB_DEFAULT_INCLUDE_DIRS
                  DOC "The path to TBB headers")
        if (TBB_INCLUDE_DIR)
            if (UNIX)
                set(TBB_LIB_DIR "${TBB_INCLUDE_DIR}/../lib" CACHE PATH "Full path of TBB library directory")
                link_directories("${TBB_LIB_DIR}")
            endif()
            if (APPLE)
                set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} libtbb.dylib)
            elseif (UNIX)
                set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} tbb)
            elseif (WIN32)
               if (CMAKE_COMPILER_IS_GNUCXX)
                   set(TBB_LIB_DIR "${TBB_INCLUDE_DIR}/../lib" CACHE PATH "Full path of TBB library directory")
                   link_directories("${TBB_LIB_DIR}")
                   set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} tbb)
               else()
                   if(${CMAKE_SYSTEM_PROCESSOR} MATCHES amd64*)
                       set(X86_64 1)
                   endif()
                   if(${CMAKE_SYSTEM_PROCESSOR} MATCHES x86_64*)
                       set(X86_64 1)
                   endif()
                   if(${CMAKE_SYSTEM_PROCESSOR} MATCHES i686*)
                       set(X86 1)
                   endif()
                   if(${CMAKE_SYSTEM_PROCESSOR} MATCHES i386*)
                       set(X86 1)
                   endif()
                   if(${CMAKE_SYSTEM_PROCESSOR} MATCHES x86*)
                       set(X86 1)
                   endif()

                   set(_TBB_LIB_PATH "${TBB_INCLUDE_DIR}/../lib")
                   if (X86_64)
                       set(_TBB_LIB_PATH "${_TBB_LIB_PATH}/intel64")
                   elseif(X86)
                       set(_TBB_LIB_PATH "${_TBB_LIB_PATH}/ia32")
                   endif()
                   if (MSVC80)
                       set(_TBB_LIB_PATH "${_TBB_LIB_PATH}/vc8")
                   elseif(MSVC90)
                       set(_TBB_LIB_PATH "${_TBB_LIB_PATH}/vc9")
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
endif()

############################### CUDA ################################

if (WITH_CUDA)
	find_package(CUDA)
	if (CUDA_FOUND)
		message(STATUS "CUDA detected.")
		set(HAVE_CUDA 1)

		set(CUDA_COMPUTE_CAPABILITIES " 1.0 1.1 1.2 1.3 2.0 " CACHE STRING "Add or remove compute capability")
		set(CUDA_NVCC_FLAGS_ARCH ${CUDA_COMPUTE_CAPABILITIES})

		set(CUDA_NVCC_FLAGS_NUM "")

		while(NOT ${CUDA_NVCC_FLAGS_ARCH} STREQUAL "")
		    string(REGEX MATCH "[0-9]+.[0-9]+" RESULT_NUM ${CUDA_NVCC_FLAGS_ARCH})
		    string(REGEX MATCHALL "[0-9]" RESULT_STR ${RESULT_NUM})
		    string(REGEX REPLACE ";" "\ " RESULT ${RESULT_STR})
		    list(APPEND CUDA_NVCC_FLAGS_NUM ${RESULT})
		    string(REGEX REPLACE "${RESULT_NUM}" "\ " CUDA_NVCC_FLAGS_ARCH_STR ${CUDA_NVCC_FLAGS_ARCH})
		    string(STRIP ${CUDA_NVCC_FLAGS_ARCH_STR} CUDA_NVCC_FLAGS_ARCH)
		endwhile()

		set (OpenCV_CUDA_CC "")
		set (loop_var "")
		foreach( loop_var IN LISTS CUDA_NVCC_FLAGS_NUM)
		    set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_${loop_var},code=sm_${loop_var})
		    set (OpenCV_CUDA_CC  ${OpenCV_CUDA_CC}  -gencode arch=compute_${loop_var},code=sm_${loop_var})
		endforeach()

		### set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${OpenCV_COMPUTE_CAPABILITIES})
	endif()
endif()

############################### VideoInput ################################

if (WIN32 AND WITH_VIDEOINPUT)
	if(CMAKE_CXX_COMPILER MATCHES "dw2")
	else()
	if(NOT MINGW64)
		  set(HAVE_VIDEOINPUT 1)
	endif()
	endif()
endif()

############################## Eigen2 ##############################

if(WITH_EIGEN2)
        find_path(EIGEN2_INCLUDE_PATH "Eigen/Core"
            PATHS "/usr/local/include/eigen2" "/opt/include/eigen2" "/usr/include/eigen2"
            "/usr/local/include/eigen3" "/opt/include/eigen3" "/usr/include/eigen3"
            DOC "The path to Eigen2/Eigen3 headers")
    if(EIGEN2_INCLUDE_PATH)
        include_directories(${EIGEN2_INCLUDE_PATH})
        set(HAVE_EIGEN2 1)
    endif()
endif()

############################### IPP ################################
set(IPP_FOUND)
set(OPENCV_LOADER_PATH)

if(UNIX)
if(APPLE)
    set(OPENCV_LOADER_PATH DYLD_LIBRARY_PATH)
else()
    set(OPENCV_LOADER_PATH LD_LIBRARY_PATH)
endif()
endif()

foreach(v "6.1" "6.0" "5.3" "5.2" "5.1")
    if(NOT IPP_FOUND)
        if(WIN32)
            find_path(IPP_PATH "ippi-${v}.dll"
                PATHS ${CMAKE_PROGRAM_PATH} ${CMAKE_SYSTEM_PROGRAM_PATH}
                DOC "The path to IPP dynamic libraries")
            if(NOT IPP_PATH)
                find_path(IPP_PATH "ippiem64t-${v}.dll"
                    PATHS ${CMAKE_PROGRAM_PATH} ${CMAKE_SYSTEM_PROGRAM_PATH}
                    DOC "The path to IPP dynamic libraries")
            endif()
        endif()
        if(UNIX)
            find_path(IPP_PATH "libippi${CMAKE_SHARED_LIBRARY_SUFFIX}.${v}"
                PATHS ${CMAKE_LIBRARY_PATH} ${CMAKE_SYSTEM_LIBRARY_PATH} ENV ${OPENCV_LOADER_PATH}
                DOC "The path to IPP dynamic libraries")
            if(NOT IPP_PATH)
                find_path(IPP_PATH "libippiem64t${CMAKE_SHARED_LIBRARY_SUFFIX}.${v}"
                    PATHS ${CMAKE_LIBRARY_PATH} ${CMAKE_SYSTEM_LIBRARY_PATH} ENV ${OPENCV_LOADER_PATH}
                    DOC "The path to IPP dynamic libraries")
            endif()
        endif()
        if(IPP_PATH)
            file(GLOB IPP_HDRS "${IPP_PATH}/../include")
            if(IPP_HDRS)
                set(IPP_FOUND TRUE)
            endif()
        endif()
    endif()
endforeach()

message(STATUS "IPP detected: ${IPP_FOUND}")

if(WIN32 AND NOT MSVC)
    set(IPP_FOUND)
endif()

set(USE_IPP ${IPP_FOUND} CACHE BOOL "Use IPP when available")

if(IPP_FOUND AND USE_IPP)
    add_definitions(-DHAVE_IPP)
    include_directories("${IPP_PATH}/../include")
    link_directories("${IPP_PATH}/../lib")

    file(GLOB em64t_files "${IPP_PATH}/../lib/*em64t*")
    set(IPP_ARCH)
    if(em64t_files)
        set(IPP_ARCH "em64t")
    endif()

    set(A ${CMAKE_STATIC_LIBRARY_PREFIX})
    set(B ${IPP_ARCH}${CMAKE_STATIC_LIBRARY_SUFFIX})
    if(WIN32)
        set(L l)
    else()
        set(L)
    endif()
    set(IPP_LIBS ${A}ippsmerged${B} ${A}ippsemerged${B}
                 ${A}ippimerged${B} ${A}ippiemerged${B}
                 ${A}ippvmmerged${B} ${A}ippvmemerged${B}
                 ${A}ippccmerged${B} ${A}ippccemerged${B}
                 ${A}ippcvmerged${B} ${A}ippcvemerged${B}
                 ${A}ippcore${IPP_ARCH}${L}${CMAKE_STATIC_LIBRARY_SUFFIX})
endif()


################## LATEX ##################
set(BUILD_LATEX_DOCS OFF CACHE BOOL "Build LaTeX OpenCV Documentation")

################### DOXYGEN ###############

find_package(Doxygen)

if(DOXYGEN_FOUND)
    set(BUILD_DOXYGEN_DOCS ON CACHE BOOL "Generate HTML docs using Doxygen")
endif()

# ----------------------------------------------------------------------------
#                   UPDATE CONFIG FILES & SCRIPTS:
#
#  CONFIGURE_FILE(InputFile OutputFile [COPYONLY] [ESCAPE_QUOTES] [@ONLY])
# If @ONLY is specified, only variables of the form @VAR@ will be
#  replaces and ${VAR} will be ignored.
#
#  A directory will be created for each platform so the "cvconfig.h" file is
#   not overwritten if cmake generates code in the same path.
# ----------------------------------------------------------------------------
add_definitions(-DHAVE_CONFIG_H)

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

    set(EXTRA_EXE_LINKER_FLAGS_RELEASE "${EXTRA_EXE_LINKER_FLAGS_RELEASE} /debug")

    #if(ENABLE_OPENMP)
    #    set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} /openmp")
    #endif()

    # Remove unreferenced functions: function level linking
    set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} /Gy")
    set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} /Zi")
endif()

if(CMAKE_COMPILER_IS_GNUCXX)

    # High level of warnings.
    set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -Wall")

    # The -Wno-long-long is required in 64bit systems when including sytem headers.
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES x86_64*)
    set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -Wno-long-long")
    endif()
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES amd64*)
    set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -Wno-long-long")
    endif()

    # We need pthread's
    if(UNIX)
        set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -pthread")
    endif()

    if(OPENCV_WARNINGS_ARE_ERRORS)
        set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -Werror")
    endif()

    if(X86)
        if(NOT MINGW64)
            if(NOT X86_64)
                if(NOT APPLE)
                    set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -march=i686")
                endif()
            endif()
        endif()
    endif()

    # Other optimizations
    if(USE_OMIT_FRAME_POINTER)
       set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -fomit-frame-pointer")
    endif()
    if(USE_O2)
       set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -O2")
    endif()
    if(USE_O3)
       set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -O3")
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
       if(HAVE_GCC43_OR_NEWER OR APPLE)
          if(ENABLE_SSSE3)
             set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -mssse3")
          endif()
          #if(ENABLE_SSE4_1)
          #   set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -msse4.1")
          #endif()
       endif()
    endif()

    if(X86 OR X86_64)
        if(NOT APPLE)
            if(${CMAKE_SIZEOF_VOID_P} EQUAL 4)
                set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -mfpmath=387")
            endif()
        endif()
    endif()

    # Profiling?
    if(ENABLE_PROFILING)
        set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -pg -g")
    else()
        # Remove unreferenced functions: function level linking
        if(NOT APPLE)
            set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -ffunction-sections")
        endif()
    endif()

    # Parallel mode
    #if(ENABLE_OPENMP)
    #    set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -D_GLIBCXX_PARALLEL -fopenmp")
    #    set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} gomp)
    #endif()

    set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -DNDEBUG")
    set(EXTRA_C_FLAGS_DEBUG "${EXTRA_C_FLAGS_DEBUG} -O0 -ggdb3 -DDEBUG -D_DEBUG")
endif()

# Extra link libs if the user selects building static libs:
IF(NOT BUILD_SHARED_LIBS)
    if(CMAKE_COMPILER_IS_GNUCXX)
        set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} stdc++)
    endif()

    set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} opencv_flann zlib opencv_lapack)
endif()


# Add user supplied extra options (optimization, etc...)
# ==========================================================
set(OPENCV_EXTRA_C_FLAGS "" CACHE STRING "Extra compiler options")
set(OPENCV_EXTRA_C_FLAGS_RELEASE "" CACHE STRING "Extra compiler options for Release build")
set(OPENCV_EXTRA_C_FLAGS_DEBUG "" CACHE STRING "Extra compiler options for Debug build")
set(OPENCV_EXTRA_EXE_LINKER_FLAGS "" CACHE STRING "Extra linker flags" FORCE)
set(OPENCV_EXTRA_EXE_LINKER_FLAGS_RELEASE "" CACHE STRING "Extra linker flags for Release build" FORCE)
set(OPENCV_EXTRA_EXE_LINKER_FLAGS_DEBUG "" CACHE STRING "Extra linker flags for Debug build" FORCE)

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
if (${CMAKE_GENERATOR} MATCHES ".*Makefiles")
    if("${CMAKE_BUILD_TYPE}" STREQUAL "")
        set(CMAKE_BUILD_TYPE Release)
    endif()
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

# ----------------------------------------------------------------------------
#                       PROCESS SUBDIRECTORIES:
# ----------------------------------------------------------------------------
# Save libs and executables in the same place
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib CACHE PATH "Output directory for libraries" )
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin CACHE PATH "Output directory for applications" )

if(WIN32)
set(OPENCV_DOC_INSTALL_PATH doc)
else()
set(OPENCV_DOC_INSTALL_PATH share/opencv/doc)
endif()

# --------------------------------------------------------------------------------------------
#   Installation for CMake Module:  OpenCVConfig.cmake
#  Part 1/3: ${BIN_DIR}/OpenCVConfig.cmake              -> For use *without* "make install"
#  Part 2/3: ${BIN_DIR}/unix-install/OpenCVConfig.cmake -> For use with "make install"
#  Part 3/3: ${BIN_DIR}/win-install/OpenCVConfig.cmake  -> For use within binary installers/packages
# -------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------
#  Part 1/3: ${BIN_DIR}/OpenCVConfig.cmake              -> For use *without* "make install"
# -------------------------------------------------------------------------------------------

# Name of libraries is: libcv.so.1.1.0, etc...
#  OPENCV_DLLVERSION must be set to "110", etc..
# Already done above.

# Set CMAKE_INCLUDE_DIRS_CONFIGCMAKE to the list of include directories:
set(CMAKE_INCLUDE_DIRS_CONFIGCMAKE "\"${OPENCV_CONFIG_FILE_INCLUDE_DIR}\" \"${CMAKE_CURRENT_SOURCE_DIR}/include\" \"${CMAKE_CURRENT_SOURCE_DIR}/include/opencv\"")
set(CMAKE_BASE_INCLUDE_DIRS_CONFIGCMAKE "\"${CMAKE_CURRENT_SOURCE_DIR}\"")

set(CMAKE_LIB_DIRS_CONFIGCMAKE "${LIBRARY_OUTPUT_PATH}")

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/OpenCVConfig.cmake.in" "${CMAKE_BINARY_DIR}/OpenCVConfig.cmake" IMMEDIATE @ONLY)

# --------------------------------------------------------------------------------------------
#  Part 2/3: ${BIN_DIR}/unix-install/OpenCVConfig.cmake -> For use *with* "make install"
# -------------------------------------------------------------------------------------------
# Set CMAKE_INCLUDE_DIRS_CONFIGCMAKE to the list of include directories:

set(CMAKE_INCLUDE_DIRS_CONFIGCMAKE "${CMAKE_INSTALL_PREFIX}/include/opencv" "${CMAKE_INSTALL_PREFIX}/include")
set(CMAKE_BASE_INCLUDE_DIRS_CONFIGCMAKE "\"\"")

set(CMAKE_LIB_DIRS_CONFIGCMAKE "${CMAKE_INSTALL_PREFIX}/lib")

#exec_program(${CMAKE_COMMAND} ARGS "-E make_directory \"${CMAKE_BINARY_DIR}/unix-install/\"")
exec_program(mkdir ARGS "-p \"${CMAKE_BINARY_DIR}/unix-install/\"")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/OpenCVConfig.cmake.in" "${CMAKE_BINARY_DIR}/unix-install/OpenCVConfig.cmake" IMMEDIATE @ONLY)

if(UNIX)
    # For a command "FIND_PACKAGE(FOO)", CMake will look at the directory /usr/share|lib/FOO/FOOConfig.cmake, so:
    install(FILES ${CMAKE_BINARY_DIR}/unix-install/OpenCVConfig.cmake DESTINATION ${CMAKE_INSTALL_PREFIX}/share/opencv/)
endif()

# --------------------------------------------------------------------------------------------
#  Part 3/3: ${BIN_DIR}/win-install/OpenCVConfig.cmake  -> For use within binary installers/packages
# -------------------------------------------------------------------------------------------
if(WIN32)
    # Set CMAKE_INCLUDE_DIRS_CONFIGCMAKE to the list of include directories:

    # This will expand to, for example, <program files>/OpenCV 1.1.0/include
    set(CMAKE_INCLUDE_DIRS_CONFIGCMAKE "\"\${THIS_OPENCV_CONFIG_PATH}/include\" \"\${THIS_OPENCV_CONFIG_PATH}/include/opencv\"")
        set(CMAKE_BASE_INCLUDE_DIRS_CONFIGCMAKE "\"${THIS_OPENCV_CONFIG_PATH}\"")

    # This will expand to, for example, <program files>/OpenCV 1.1.0/lib
    set(CMAKE_LIB_DIRS_CONFIGCMAKE "\"\${THIS_OPENCV_CONFIG_PATH}/lib\"")

    #exec_program(${CMAKE_COMMAND} ARGS "-E make_directory \"${CMAKE_BINARY_DIR}/win-install/\"")
    exec_program(mkdir ARGS "-p \"${CMAKE_BINARY_DIR}/win-install/\"")
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/OpenCVConfig.cmake.in" "${CMAKE_BINARY_DIR}/win-install/OpenCVConfig.cmake" IMMEDIATE @ONLY)
endif()


# --------------------------------------------------------------------------------------------
#  Part 2/2: ${BIN_DIR}/unix-install/opencv.pc -> For use *with* "make install"
# prefix=/usr
# exec_prefix=${prefix}
# libdir=${exec_prefix}/lib
# includedir=${prefix}/include/opencv
# -------------------------------------------------------------------------------------------
set(prefix      ${CMAKE_INSTALL_PREFIX})
set(exec_prefix     "\${prefix}")
set(libdir      "\${exec_prefix}/lib")
set(includedir      "\${prefix}/include")
set(VERSION             ${OPENCV_VERSION})

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/opencv.pc.cmake.in" "${CMAKE_BINARY_DIR}/unix-install/opencv.pc" @ONLY IMMEDIATE)

if(UNIX)
    install(FILES ${CMAKE_BINARY_DIR}/unix-install/opencv.pc DESTINATION lib/pkgconfig)
endif()


# ----------------------------------------------------------------------------
#   Uninstall target, for "make uninstall"
# ----------------------------------------------------------------------------
CONFIGURE_FILE(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
  IMMEDIATE @ONLY)

ADD_CUSTOM_TARGET(uninstall
  "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake")


# ----------------------------------------------------------------------------
#   CPack target
# ----------------------------------------------------------------------------

set(BUILD_PACKAGE OFF CACHE BOOL "Build a installer with the SDK")

if(BUILD_PACKAGE)

    configure_file(
    Package.cmake.in
    ${CMAKE_BINARY_DIR}/.cpack/Package.cmake
    @ONLY
    )

    include(${CMAKE_BINARY_DIR}/.cpack/Package.cmake)

    set(root_files README)

    file(GLOB root_src_files *.in *.cmake CMakeLists.txt)

    if (NOT WIN32)
        install(FILES ${root_files}
                DESTINATION ${OPENCV_DOC_INSTALL_PATH}
                COMPONENT main)
    else()
        install(FILES ${root_files}
                DESTINATION "."
                COMPONENT main)
        install(FILES ${root_src_files}
                DESTINATION "."
                COMPONENT src)
        # Add the "win-install/OpenCVConfig.cmake" file to the "main" install component
        install(FILES
            "${CMAKE_BINARY_DIR}/win-install/OpenCVConfig.cmake"
            DESTINATION "."
            COMPONENT main
            )

        if(CMAKE_INSTALL_DEBUG_LIBRARIES)
            foreach(m cxcore cv cvaux ml highgui opencv_ffmpeg)
                install(FILES ${CMAKE_BINARY_DIR}/bin/Debug/${m}${OPENCV_DLLVERSION}d.dll
            DESTINATION bin COMPONENT main)
                install(FILES ${CMAKE_BINARY_DIR}/lib/Debug/${m}${OPENCV_DLLVERSION}d.lib
            DESTINATION lib COMPONENT main)
            endforeach()
        endif()

        install(DIRECTORY data samples DESTINATION "." COMPONENT main PATTERN ".svn" EXCLUDE)
        install(DIRECTORY 3rdparty apps interfaces src tests utils DESTINATION "." COMPONENT src PATTERN ".svn" EXCLUDE)
        install(DIRECTORY doc/ DESTINATION doc COMPONENT src FILES_MATCHING PATTERN "*.tex")
        install(DIRECTORY doc/pics DESTINATION doc COMPONENT src PATTERN ".svn" EXCLUDE)
        install(DIRECTORY doc/plastex DESTINATION doc COMPONENT src PATTERN ".svn" EXCLUDE)

    endif()
endif()


#-----------------------------------
# Subdirectories:
#-----------------------------------
add_subdirectory(include)
add_subdirectory(modules)

if(BUILD_LATEX_DOCS)
  include(OpenCVFindLATEX.cmake REQUIRED)

  if(PDFLATEX_COMPILER)
    message(STATUS "PDF LaTeX found!")
  endif()
endif()

add_subdirectory(doc)
add_subdirectory(data)

if(BUILD_EXAMPLES OR INSTALL_PYTHON_EXAMPLES)
    add_subdirectory(samples)
endif()

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

add_subdirectory(3rdparty)


# ----------------------------------------------------------------------------
#   Sumary:
# ----------------------------------------------------------------------------
message(STATUS "")
message(STATUS "General configuration for opencv ${OPENCV_VERSION} =====================================")
message(STATUS "")
message(STATUS "    Built as dynamic libs?:    ${BUILD_SHARED_LIBS}")
message(STATUS "    Compiler:                  ${CMAKE_COMPILER}")
message(STATUS "    C++ flags (Release):       ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "    C++ flags (Debug):         ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
if(WIN32)
message(STATUS "    Linker flags (Release):    ${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS_RELEASE}")
message(STATUS "    Linker flags (Debug):      ${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS_DEBUG}")
else()
message(STATUS "    Linker flags (Release):    ${CMAKE_SHARED_LINKER_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS_RELEASE}")
message(STATUS "    Linker flags (Debug):      ${CMAKE_SHARED_LINKER_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS_DEBUG}")
endif()

#YV
message(STATUS "")
message(STATUS "  GUI: ")

if (HAVE_QT)
    message(STATUS "    QT 4.x:                    ${HAVE_QT}")
    message(STATUS "    QT OpenGL support:         ${HAVE_QT_OPENGL}")
else()
    if(WIN32)
        message(STATUS "    Win32 UI:                  1")
    else()
        if(APPLE)
            if(WITH_CARBON)
                message(STATUS "    Carbon:                    1")
            else()
                message(STATUS "    Cocoa:                     1")
            endif()
        else()
            message(STATUS "    GTK+ 2.x:                  ${HAVE_GTK}")
            message(STATUS "    GThread:                   ${HAVE_GTHREAD}")
        endif()
    endif()
endif()

message(STATUS "")
message(STATUS "  Image I/O: ")
if(NOT WITH_JPEG OR JPEG_FOUND)
message(STATUS "    JPEG:                      ${JPEG_FOUND}")
else()
message(STATUS "    JPEG:                      build")
endif()

if(NOT WITH_PNG OR PNG_FOUND)
message(STATUS "    PNG:                       ${PNG_FOUND}")
else()
message(STATUS "    PNG:                       build")
endif()

if(NOT WITH_TIFF OR TIFF_FOUND)
message(STATUS "    TIFF:                      ${TIFF_FOUND}")
else()
message(STATUS "    TIFF:                      build")
endif()

if(NOT WITH_JASPER OR JASPER_FOUND)
message(STATUS "    JPEG 2000:                 ${JASPER_FOUND}")
else()
message(STATUS "    JPEG 2000:                 build")
endif()

if(WITH_OPENEXR AND OPENEXR_FOUND)
message(STATUS "    OpenEXR:                   YES")
else()
message(STATUS "    OpenEXR:                   NO")
endif()

if(UNIX AND NOT APPLE)
message(STATUS "")
message(STATUS "  Video I/O: ")
message(STATUS "    DC1394 1.x:                ${HAVE_DC1394}")
message(STATUS "    DC1394 2.x:                ${HAVE_DC1394_2}")
message(STATUS "    FFMPEG:                    ${HAVE_FFMPEG}")
message(STATUS "      codec:                   ${HAVE_FFMPEG_CODEC}")
message(STATUS "      format:                  ${HAVE_FFMPEG_FORMAT}")
message(STATUS "      util:                    ${HAVE_FFMPEG_UTIL}")
message(STATUS "      swscale:                 ${HAVE_FFMPEG_SWSCALE}")
message(STATUS "      gentoo-style:            ${HAVE_GENTOO_FFMPEG}")
message(STATUS "    GStreamer:                 ${HAVE_GSTREAMER}")
message(STATUS "    UniCap:                    ${HAVE_UNICAP}")
message(STATUS "    PvAPI:                     ${HAVE_PVAPI}")
if(HAVE_LIBV4L)
message(STATUS "    V4L/V4L2:                  Using libv4l")
else()
message(STATUS "    V4L/V4L2:                  ${HAVE_CAMV4L}/${HAVE_CAMV4L2}")
endif()
message(STATUS "    Xine:                      ${HAVE_XINE}")
endif()

if(APPLE)
message(STATUS "")
if(WITH_QUICKTIME)
message(STATUS "  Video I/O:                   QuickTime")
else()
message(STATUS "  Video I/O:                   QTKit")
endif()
endif()

if(WIN32)
message(STATUS "")
message(STATUS "  Video I/O: ")
if(HAVE_VIDEOINPUT)
message(STATUS "    VideoInput:                1")
else()
message(STATUS "    VideoInput:                0")
endif()
endif()

message(STATUS "")
message(STATUS "  Interfaces: ")
message(STATUS "    Python:                    ${BUILD_NEW_PYTHON_SUPPORT}")
message(STATUS "    Python interpreter:        ${PYTHON_EXECUTABLE}")
message(STATUS "    Python numpy:              ${PYTHON_USE_NUMPY}")

if(IPP_FOUND AND USE_IPP)
message(STATUS "    Use IPP:                   ${IPP_PATH}")
else()
message(STATUS "    Use IPP:                   NO")
endif()

if(HAVE_TBB)
message(STATUS "    Use TBB:                   YES")
else()
message(STATUS "    Use TBB:                   NO")
endif()

if (HAVE_CUDA)
message(STATUS "    Use Cuda:                  YES")
else()
message(STATUS "    Use Cuda:                  No")
endif()

if(HAVE_EIGEN2)
message(STATUS "    Use Eigen2:                YES")
else()
message(STATUS "    Use Eigen2:                NO")
endif()

message(STATUS "")
message(STATUS "  Documentation: ")

if(BUILD_LATEX_DOCS AND PDFLATEX_COMPILER)
message(STATUS "    Build PDF                  YES")
else()
message(STATUS "    Build PDF                  NO")
endif()

if(BUILD_DOXYGEN_DOCS AND DOXYGEN_FOUND)
message(STATUS "    Doxygen HTMLs              YES")
else()
message(STATUS "    Doxygen HTMLs              NO")
endif()

message(STATUS "")
message(STATUS "    Install path:              ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
message(STATUS "    cvconfig.h is in:          ${OPENCV_CONFIG_FILE_INCLUDE_DIR}")
message(STATUS "-----------------------------------------------------------------")
message(STATUS "")

#----------------------------------------------------------------------------
#   Generate the OpenCVConfig.cmake file for unix
#      installation in CMAKE_INSTALL_PREFIX
#----------------------------------------------------------------------------
