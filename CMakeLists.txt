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

# --------------------------------------------------------------
# Indicate CMake 2.7 and above that we don't want to mix relative
#  and absolute paths in linker lib lists.
# Run "cmake --help-policy CMP0003" for more information.
# --------------------------------------------------------------
if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif()

# Following block can broke build in case of cross-compilng
# but CMAKE_CROSSCOMPILING variable will be set only on project(OpenCV) command
# so we will try to detect crosscompiling by presense of CMAKE_TOOLCHAIN_FILE
if(NOT CMAKE_TOOLCHAIN_FILE)
  # it _must_ go before project(OpenCV) in order to work
  if(WIN32)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory")
  else()
    set(CMAKE_INSTALL_PREFIX "/usr/local" CACHE PATH "Installation Directory")
  endif()

  if(MSVC)
    set(CMAKE_USE_RELATIVE_PATHS ON CACHE INTERNAL "" FORCE)
  endif()
else(NOT CMAKE_TOOLCHAIN_FILE)
  #Android: set output folder to ${CMAKE_BINARY_DIR}
  set( LIBRARY_OUTPUT_PATH_ROOT ${CMAKE_BINARY_DIR} CACHE PATH "root for library output, set this to change where android libs are compiled to" )
endif(NOT CMAKE_TOOLCHAIN_FILE)

# --------------------------------------------------------------
# Top level OpenCV project
# --------------------------------------------------------------
if(NOT IOS)
  cmake_minimum_required(VERSION 2.6.3)
else()
  cmake_minimum_required(VERSION 2.8)
endif()

set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Configs" FORCE)
if(DEFINED CMAKE_BUILD_TYPE AND CMAKE_VERSION VERSION_GREATER "2.8")
  set_property( CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${CMAKE_CONFIGURATION_TYPES} )
endif()

project(OpenCV)

include(cmake/OpenCVUtils.cmake REQUIRED)

# ----------------------------------------------------------------------------
# Detect compiler and target platform architecture
# ----------------------------------------------------------------------------
include(cmake/OpenCVDetectCXXCompiler.cmake REQUIRED)

# Add these standard paths to the search paths for FIND_LIBRARY
# to find libraries from these locations first
if(UNIX AND NOT ANDROID)
  if(X86_64 OR CMAKE_SIZEOF_VOID_P EQUAL 8)
    if(EXISTS /lib64)
      list(APPEND CMAKE_LIBRARY_PATH /lib64)
    else()
      list(APPEND CMAKE_LIBRARY_PATH /lib)
    endif()
    if(EXISTS /usr/lib64)
      list(APPEND CMAKE_LIBRARY_PATH /usr/lib64)
    else()
      list(APPEND CMAKE_LIBRARY_PATH /usr/lib)
    endif()
  elseif(X86 OR CMAKE_SIZEOF_VOID_P EQUAL 4)
    if(EXISTS /lib32)
      list(APPEND CMAKE_LIBRARY_PATH /lib32)
    else()
      list(APPEND CMAKE_LIBRARY_PATH /lib)
    endif()
    if(EXISTS /usr/lib32)
      list(APPEND CMAKE_LIBRARY_PATH /usr/lib32)
    else()
      list(APPEND CMAKE_LIBRARY_PATH /usr/lib)
    endif()
  endif()
endif()


# ----------------------------------------------------------------------------
# OpenCV cmake options
# ----------------------------------------------------------------------------

# Optional 3rd party components
# ===================================================
OCV_OPTION(WITH_1394           "Include IEEE1394 support"                    ON   IF (UNIX AND NOT ANDROID AND NOT IOS) )
OCV_OPTION(WITH_AVFOUNDATION   "Use AVFoundation for Video I/O"              ON   IF IOS)
OCV_OPTION(WITH_CARBON         "Use Carbon for UI instead of Cocoa"          OFF  IF APPLE )
OCV_OPTION(WITH_CUBLAS         "Include NVidia Cuda Basic Linear Algebra Subprograms (BLAS) library support" OFF IF (CMAKE_VERSION VERSION_GREATER "2.8" AND NOT ANDROID AND NOT IOS) )
OCV_OPTION(WITH_CUDA           "Include NVidia Cuda Runtime support"         ON   IF (CMAKE_VERSION VERSION_GREATER "2.8" AND NOT ANDROID AND NOT IOS) )
OCV_OPTION(WITH_CUFFT          "Include NVidia Cuda Fast Fourier Transform (FFT) library support"            ON  IF (CMAKE_VERSION VERSION_GREATER "2.8" AND NOT ANDROID AND NOT IOS) )
OCV_OPTION(WITH_EIGEN          "Include Eigen2/Eigen3 support"               ON)
OCV_OPTION(WITH_FFMPEG         "Include FFMPEG support"                      ON   IF (UNIX AND NOT ANDROID AND NOT IOS) )
OCV_OPTION(WITH_GSTREAMER      "Include Gstreamer support"                   ON   IF (UNIX AND NOT APPLE AND NOT ANDROID AND NOT IOS) )
OCV_OPTION(WITH_GTK            "Include GTK support"                         ON   IF (UNIX AND NOT APPLE AND NOT ANDROID AND NOT IOS) )
OCV_OPTION(WITH_IPP            "Include Intel IPP support"                   OFF  IF (MSVC OR X86 OR X86_64) )
OCV_OPTION(WITH_JASPER         "Include JPEG2K support"                      ON   IF (NOT IOS) )
OCV_OPTION(WITH_JPEG           "Include JPEG support"                        ON   IF (NOT IOS) )
OCV_OPTION(WITH_OPENEXR        "Include ILM support via OpenEXR"             ON   IF (NOT IOS) )
OCV_OPTION(WITH_OPENGL         "Include OpenGL support"                      OFF  IF (NOT ANDROID AND NOT IOS) )
OCV_OPTION(WITH_OPENNI         "Include OpenNI support"                      OFF  IF (NOT ANDROID AND NOT IOS) )
OCV_OPTION(WITH_PNG            "Include PNG support"                         ON   IF (NOT IOS) )
OCV_OPTION(WITH_PVAPI          "Include Prosilica GigE support"              ON   IF (UNIX AND NOT ANDROID AND NOT IOS) )
OCV_OPTION(WITH_QT             "Build with Qt Backend support"               OFF  IF (NOT ANDROID AND NOT IOS) )
OCV_OPTION(WITH_QUICKTIME      "Use QuickTime for Video I/O insted of QTKit" OFF  IF APPLE )
OCV_OPTION(WITH_TBB            "Include Intel TBB support"                   OFF  IF (NOT IOS) )
OCV_OPTION(WITH_TIFF           "Include TIFF support"                        ON   IF (NOT IOS) )
OCV_OPTION(WITH_UNICAP         "Include Unicap support (GPL)"                OFF  IF (UNIX AND NOT APPLE AND NOT ANDROID AND NOT IOS) )
OCV_OPTION(WITH_V4L            "Include Video 4 Linux support"               ON   IF (UNIX AND NOT APPLE AND NOT ANDROID AND NOT IOS) )
OCV_OPTION(WITH_VIDEOINPUT     "Build HighGUI with DirectShow support"       ON   IF WIN32 )
OCV_OPTION(WITH_XIMEA          "Include XIMEA cameras support"               OFF  IF WIN32 )
OCV_OPTION(WITH_XINE           "Include Xine support (GPL)"                  OFF  IF (UNIX AND NOT APPLE AND NOT ANDROID AND NOT IOS) )

# OpenCV build components
# ===================================================
OCV_OPTION(BUILD_SHARED_LIBS        "Build shared libraries (.dll/.so) instead of static ones (.lib/.a)" NOT (ANDROID OR IOS) )
OCV_OPTION(BUILD_ANDROID_EXAMPLES   "Build examples for Android platform"         ON  IF ANDROID )
OCV_OPTION(BUILD_DOCS               "Create build rules for OpenCV Documentation" ON )
OCV_OPTION(BUILD_EXAMPLES           "Build all examples"                          OFF )
OCV_OPTION(BUILD_PACKAGE            "Enables 'make package_source' command"       ON )
OCV_OPTION(BUILD_PERF_TESTS         "Build performance tests"                     ON  IF (NOT IOS) )
OCV_OPTION(BUILD_TESTS              "Build accuracy & regression tests"           ON  IF (NOT IOS) )
OCV_OPTION(BUILD_WITH_DEBUG_INFO    "Include debug info into debug libs (not MSCV only)" ON )
OCV_OPTION(BUILD_WITH_STATIC_CRT    "Enables use of staticaly linked CRT for staticaly linked OpenCV" ON IF MSVC )
OCV_OPTION(BUILD_FAT_JAVA_LIB       "Create fat java wrapper containing the whole OpenCV library" ON IF ANDROID AND NOT BUILD_SHARED_LIBS AND CMAKE_COMPILER_IS_GNUCXX )

# 3rd party libs
OCV_OPTION(BUILD_ZLIB               "Build zlib from source"         WIN32 OR IOS )
OCV_OPTION(BUILD_TIFF               "Build libtiff from source"      WIN32 OR IOS OR ANDROID )
OCV_OPTION(BUILD_JASPER             "Build libjasper from source"    WIN32 OR IOS OR ANDROID )
OCV_OPTION(BUILD_JPEG               "Build libjpeg from source"      WIN32 OR IOS OR ANDROID )
OCV_OPTION(BUILD_PNG                "Build libpng from source"       WIN32 OR IOS OR ANDROID )

# OpenCV installation options
# ===================================================
OCV_OPTION(INSTALL_C_EXAMPLES       "Install C examples"        OFF )
OCV_OPTION(INSTALL_PYTHON_EXAMPLES  "Install Python examples"   OFF )
OCV_OPTION(INSTALL_ANDROID_EXAMPLES "Install Android examples"  OFF  IF ANDROID )
OCV_OPTION(INSTALL_TO_MANGLED_PATHS "Enables mangled install paths, that help with side by side installs." OFF IF (UNIX AND NOT ANDROID AND NOT IOS AND BUILD_SHARED_LIBS) )

# OpenCV build options
# ===================================================
OCV_OPTION(ENABLE_PRECOMPILED_HEADERS "Use precompiled headers"                                  ON   IF (NOT IOS) )
OCV_OPTION(ENABLE_SOLUTION_FOLDERS    "Solution folder in Visual Studio or in other IDEs"        MSVC_IDE IF (CMAKE_VERSION VERSION_GREATER "2.8.0") )
OCV_OPTION(ENABLE_PROFILING           "Enable profiling in the GCC compiler (Add flags: -g -pg)" OFF  IF CMAKE_COMPILER_IS_GNUCXX )
OCV_OPTION(ENABLE_OMIT_FRAME_POINTER  "Enable -fomit-frame-pointer for GCC"                      ON   IF CMAKE_COMPILER_IS_GNUCXX )
OCV_OPTION(ENABLE_POWERPC             "Enable PowerPC for GCC"                                   ON   IF (CMAKE_COMPILER_IS_GNUCXX AND CMAKE_SYSTEM_PROCESSOR MATCHES powerpc.*) )
OCV_OPTION(ENABLE_FAST_MATH           "Enable -ffast-math (not recommended for GCC 4.6.x)"       OFF  IF (CMAKE_COMPILER_IS_GNUCXX AND (X86 OR X86_64)) )
OCV_OPTION(ENABLE_SSE                 "Enable SSE instructions"                                  ON   IF (MSVC OR CMAKE_COMPILER_IS_GNUCXX AND (X86 OR X86_64)) )
OCV_OPTION(ENABLE_SSE2                "Enable SSE2 instructions"                                 ON   IF (MSVC OR CMAKE_COMPILER_IS_GNUCXX AND (X86 OR X86_64)) )
OCV_OPTION(ENABLE_SSE3                "Enable SSE3 instructions"                                 OFF  IF (CV_ICC OR CMAKE_COMPILER_IS_GNUCXX AND (X86 OR X86_64)) )
OCV_OPTION(ENABLE_SSSE3               "Enable SSSE3 instructions"                                OFF  IF (CMAKE_COMPILER_IS_GNUCXX AND (X86 OR X86_64)) )
OCV_OPTION(ENABLE_SSE41               "Enable SSE4.1 instructions"                               OFF  IF (CV_ICC OR CMAKE_COMPILER_IS_GNUCXX AND (X86 OR X86_64)) )
OCV_OPTION(ENABLE_SSE42               "Enable SSE4.2 instructions"                               OFF  IF (CMAKE_COMPILER_IS_GNUCXX AND (X86 OR X86_64)) )
OCV_OPTION(ENABLE_NOISY_WARNINGS      "Show all warnings even if they are too noisy"             OFF )
OCV_OPTION(OPENCV_WARNINGS_ARE_ERRORS "Treat warnings as errors"                                 OFF )

# uncategorized options
# ===================================================
OCV_OPTION(CMAKE_VERBOSE "Verbose mode" OFF )

# backward compatibility
# ===================================================
include(cmake/OpenCVLegacyOptions.cmake OPTIONAL)


# ----------------------------------------------------------------------------
#  Get actual OpenCV version number from sources
# ----------------------------------------------------------------------------
include(cmake/OpenCVVersion.cmake REQUIRED)


# ----------------------------------------------------------------------------
#  Build & install layouts
# ----------------------------------------------------------------------------

# Save libs and executables in the same place
set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/lib" CACHE PATH "Output directory for libraries" )
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin" CACHE PATH "Output directory for applications" )

if(ANDROID OR WIN32)
    set(OPENCV_DOC_INSTALL_PATH doc)
elseif(INSTALL_TO_MANGLED_PATHS)
    set(OPENCV_DOC_INSTALL_PATH share/OpenCV-${OPENCV_VERSION}/doc)
else()
    set(OPENCV_DOC_INSTALL_PATH share/OpenCV/doc)
endif()

if(ANDROID)
    set(OPENCV_LIB_INSTALL_PATH libs/${ANDROID_NDK_ABI_NAME})
else()
    set(OPENCV_LIB_INSTALL_PATH lib${LIB_SUFFIX})
endif()

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${OPENCV_LIB_INSTALL_PATH}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(OPENCV_INCLUDE_PREFIX include)
if(INSTALL_TO_MANGLED_PATHS)
  set(OPENCV_INCLUDE_PREFIX include/opencv-${OPENCV_VERSION})
endif()

if(WIN32)
  # Postfix of DLLs:
  set(OPENCV_DLLVERSION "${OPENCV_VERSION_MAJOR}${OPENCV_VERSION_MINOR}${OPENCV_VERSION_PATCH}")
  set(OPENCV_DEBUG_POSTFIX d)
else()
  # Postfix of so's:
  set(OPENCV_DLLVERSION "")
  set(OPENCV_DEBUG_POSTFIX "")
endif()

if(DEFINED CMAKE_DEBUG_POSTFIX)
  set(OPENCV_DEBUG_POSTFIX "${CMAKE_DEBUG_POSTFIX}")
endif()

if(CMAKE_VERBOSE)
  set(CMAKE_VERBOSE_MAKEFILE 1)
endif()


# ----------------------------------------------------------------------------
#  Path for build/platform -specific headers
# ----------------------------------------------------------------------------
set(OPENCV_CONFIG_FILE_INCLUDE_DIR "${CMAKE_BINARY_DIR}/" CACHE PATH "Where to create the platform-dependant cvconfig.h")
add_definitions(-DHAVE_CVCONFIG_H)
ocv_include_directories(${OPENCV_CONFIG_FILE_INCLUDE_DIR})


# ----------------------------------------------------------------------------
# Use statically or dynamically linked CRT?
# Default: dynamic
# ----------------------------------------------------------------------------
if(MSVC)
  include(cmake/OpenCVCRTLinkage.cmake REQUIRED)
endif(MSVC)


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
# OpenCV compiler and linker options
# ----------------------------------------------------------------------------
include(cmake/OpenCVCompilerOptions.cmake REQUIRED)

# In case of Makefiles if the user does not setup CMAKE_BUILD_TYPE, assume it's Release:
if(CMAKE_GENERATOR MATCHES "Makefiles" AND "${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE Release)
endif()


# ----------------------------------------------------------------------------
#       CHECK FOR SYSTEM LIBRARIES, OPTIONS, ETC..
# ----------------------------------------------------------------------------
if(UNIX)
  include(cmake/OpenCVFindPkgConfig.cmake OPTIONAL)
  include(CheckFunctionExists)
  include(CheckIncludeFile)
endif()

include(cmake/OpenCVPCHSupport.cmake REQUIRED)
include(cmake/OpenCVModule.cmake REQUIRED)

# ----------------------------------------------------------------------------
#  Detect 3rd-party tools and libraries
# ----------------------------------------------------------------------------

# IO libraries
include(cmake/OpenCVIOLibs.cmake REQUIRED)

#Graphic libraries
set(HAVE_OPENGL 0)
if(UNIX)
    if(NOT APPLE)
      if(WITH_GTK)
        CHECK_MODULE(gtk+-2.0 HAVE_GTK)
        CHECK_MODULE(gthread-2.0 HAVE_GTHREAD)
        if(WITH_OPENGL)
            CHECK_MODULE(gtkglext-1.0 HAVE_GTKGLEXT)
            if(HAVE_GTKGLEXT)
                find_package(OpenGL QUIET)
                if(OPENGL_FOUND)
                    set(HAVE_OPENGL 1)
                    set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} ${OPENGL_LIBRARIES})
                    ocv_include_directories(${OPENGL_INCLUDE_DIR})
                endif()
            endif()
        endif()
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

    if(HAVE_FFMPEG)
    CHECK_MODULE(libavformat>=52.111.0 NEW_FFMPEG)
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

        if(ANDROID)
            set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} dl m log)
        elseif(${CMAKE_SYSTEM_NAME} MATCHES "FreeBSD" OR ${CMAKE_SYSTEM_NAME} MATCHES "NetBSD")
            set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} m pthread)
        else()
            set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} dl m pthread rt)
        endif()
    else()
        add_definitions(-DHAVE_ALLOCA -DHAVE_ALLOCA_H -DHAVE_LIBPTHREAD -DHAVE_UNISTD_H)
    endif()
endif()

if(APPLE AND WITH_FFMPEG)
  set(FFMPEG_DEFAULT_INCLUDE_DIRS "/usr/local/include/" "/usr/include/" "opt/include/")

  find_path(FFMPEG_INCLUDE_DIR "libavformat/avformat.h" PATHS ${FFMPEG_DEFAULT_INCLUDE_DIRS} DOC "The path to FFMPEG headers")
  if(FFMPEG_INCLUDE_DIR)
    set(HAVE_GENTOO_FFMPEG 1)
    set(FFMPEG_LIB_DIR "${FFMPEG_INCLUDE_DIR}/../lib" CACHE PATH "Full path of FFMPEG library directory")
    if(EXISTS "${FFMPEG_LIB_DIR}/libavcodec.a")
    set(HAVE_FFMPEG_CODEC 1)
    if(EXISTS "${FFMPEG_LIB_DIR}/libavformat.a")
    set(HAVE_FFMPEG_FORMAT 1)
    if(EXISTS "${FFMPEG_LIB_DIR}/libavutil.a")
    set(HAVE_FFMPEG_UTIL 1)
    if(EXISTS "${FFMPEG_LIB_DIR}/libswscale.a")
        ocv_include_directories(${FFMPEG_INCLUDE_DIR})
        set(HAVE_FFMPEG_SWSCALE 1)
        set(HIGHGUI_LIBRARIES ${HIGHGUI_LIBRARIES} "${FFMPEG_LIB_DIR}/libavcodec.a"
            "${FFMPEG_LIB_DIR}/libavformat.a" "${FFMPEG_LIB_DIR}/libavutil.a"
            "${FFMPEG_LIB_DIR}/libswscale.a")
        set(HAVE_FFMPEG 1)
        set(NEW_FFMPEG 1)
    endif()
    endif()
    endif()
    endif()
  endif()
endif()

#################### LATEX for dpf documentation ##################
if(BUILD_DOCS)
  include(cmake/OpenCVFindLATEX.cmake REQUIRED)
endif()

########################## Python Support #########################
include(cmake/OpenCVDetectPython.cmake REQUIRED)

########################### Java Support ##########################
if(ANDROID)
  include(cmake/OpenCVDetectApacheAnt.cmake REQUIRED)
  include(cmake/OpenCVDetectAndroidSDK.cmake REQUIRED)
  
  if(NOT ANDROID_TOOLS_Pkg_Revision GREATER 13)
    message(WARNING "OpenCV requires Android SDK tools revision 14 or newer. Otherwise tests and samples will no be compiled.")
  endif()
endif()

if(ANDROID AND ANDROID_EXECUTABLE AND ANT_EXECUTABLE AND (ANT_VERSION VERSION_GREATER 1.7) AND (ANDROID_TOOLS_Pkg_Revision GREATER 13))
  SET(CAN_BUILD_ANDROID_PROJECTS TRUE)
else()
  SET(CAN_BUILD_ANDROID_PROJECTS FALSE)
endif()

############################### QT ################################
set(HAVE_QT 0)
set(HAVE_QT_OPENGL 0)

if(WITH_QT)
  find_package(Qt4)
  if(QT4_FOUND)
    set(HAVE_QT 1)
    add_definitions(-DHAVE_QT) #We need to define te macro this way, using cvconfig.h.cmake does not work

    if(WITH_OPENGL)
      find_package (OpenGL QUIET)
      if(QT_QTOPENGL_FOUND AND OPENGL_FOUND)
        set(HAVE_OPENGL 1)
        set(HAVE_QT_OPENGL 1)
        add_definitions(-DHAVE_QT_OPENGL)
        set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} ${OPENGL_LIBRARIES})
      endif()
    endif()
  endif()
endif()

############################### TBB ################################
if(WITH_TBB)
  include(cmake/OpenCVDetectTBB.cmake REQUIRED)
endif()

############################ Intel IPP #############################
set(IPP_FOUND)

if(WITH_IPP)
  include(cmake/OpenCVFindIPP.cmake)
endif()

if(IPP_FOUND)
  add_definitions(-DHAVE_IPP)
  ocv_include_directories(${IPP_INCLUDE_DIRS})
  link_directories(${IPP_LIBRARY_DIRS})
  set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} ${IPP_LIBRARIES})
endif()

############################### CUDA ################################
if(WITH_CUDA)
  include(cmake/OpenCVDetectCUDA.cmake REQUIRED)
endif()

############################### OpenNI ################################
set(HAVE_OPENNI FALSE)
set(HAVE_OPENNI_PRIME_SENSOR_MODULE FALSE)

if(WITH_OPENNI)
  include(cmake/OpenCVFindOpenNI.cmake)
endif()

############################### XIMEA ################################
set(HAVE_XIMEA FALSE)

if(WITH_XIMEA)
  include(cmake/OpenCVFindXimea.cmake)
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
    ocv_include_directories(${EIGEN_INCLUDE_PATH})
    set(HAVE_EIGEN 1)
  endif()
endif()

################## Extra HighGUI libs on Windows ###################
if(WIN32)
  set(HIGHGUI_LIBRARIES ${HIGHGUI_LIBRARIES} comctl32 gdi32 ole32)

  if(WITH_VIDEOINPUT)
    set(HAVE_VIDEOINPUT 1)
  endif()

  if(MSVC)
    set(HIGHGUI_LIBRARIES ${HIGHGUI_LIBRARIES} vfw32)
  endif()

  if(MINGW)
    if(MINGW64)
      set(HIGHGUI_LIBRARIES ${HIGHGUI_LIBRARIES} msvfw32 avifil32 avicap32 winmm)
    else()
      set(HIGHGUI_LIBRARIES ${HIGHGUI_LIBRARIES} vfw32 winmm)
    endif()
  endif()

  if(WITH_OPENGL AND NOT HAVE_QT_OPENGL)
    find_package(OpenGL QUIET)

    if(OPENGL_FOUND)
      set(HAVE_OPENGL 1)
      set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} ${OPENGL_LIBRARIES})
      ocv_include_directories(${OPENGL_INCLUDE_DIR})
    endif()
  endif()
endif()


# ----------------------------------------------------------------------------
# Solution folders:
# ----------------------------------------------------------------------------
if(ENABLE_SOLUTION_FOLDERS)
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
  set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "CMakeTargets")
endif()

# Extra OpenCV targets: uninstall, package_source, perf, etc.
include(cmake/OpenCVExtraTargets.cmake REQUIRED)


# ----------------------------------------------------------------------------
# Process subdirectories
# ----------------------------------------------------------------------------

# opencv.hpp and legacy headers
add_subdirectory(include)

# OpenCV modules
add_subdirectory(modules)

# Generate targets for documentation
add_subdirectory(doc)

# various data that is used by cv libraries and/or demo applications.
add_subdirectory(data)

# extra applications
add_subdirectory(apps)

# examples
if(BUILD_EXAMPLES OR BUILD_ANDROID_EXAMPLES OR INSTALL_PYTHON_EXAMPLES)
  add_subdirectory(samples)
endif()


# ----------------------------------------------------------------------------
# Finalization: generate configuration-based files
# ----------------------------------------------------------------------------
ocv_track_build_dependencies()

# Generate platform-dependent and configuration-dependent headers
include(cmake/OpenCVGenHeaders.cmake REQUIRED)

# Generate opencv.pc for pkg-config command
include(cmake/OpenCVGenPkgconfig.cmake REQUIRED)

# Generate OpenCV.mk for ndk-build (Android build tool)
include(cmake/OpenCVGenAndroidMK.cmake REQUIRED)

# Generate OpenCVСonfig.cmake and OpenCVConfig-version.cmake for cmake projects
include(cmake/OpenCVGenConfig.cmake REQUIRED)


# ----------------------------------------------------------------------------
# Summary:
# ----------------------------------------------------------------------------
status("")
status("General configuration for OpenCV ${OPENCV_VERSION} =====================================")

#build platform
status("")
status("  Platform:")
status("    Host:"             ${CMAKE_HOST_SYSTEM_NAME} ${CMAKE_HOST_SYSTEM_VERSION} ${CMAKE_HOST_SYSTEM_PROCESSOR})
status("    Target:"           ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_VERSION} ${CMAKE_SYSTEM_PROCESSOR})
status("    CMake:"            ${CMAKE_VERSION})
status("    CMake generator:"  ${CMAKE_GENERATOR})
status("    CMake build tool:" ${CMAKE_BUILD_TOOL})
if(MSVC)
  status("    MSVC:"           ${MSVC_VERSION})
endif()
if(CMAKE_GENERATOR MATCHES Xcode)
  status("    Xcode:"          ${XCODE_VERSION})
endif()

# C/C++ options
status("")
status("  C++:")
status("    Built as dynamic libs?:" BUILD_SHARED_LIBS THEN YES ELSE NO)
status("    C++ Compiler:"           ${CMAKE_CXX_COMPILER})
status("    C++ flags (Release):"    ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE})
status("    C++ flags (Debug):"      ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG})
status("    C Compiler:"             ${CMAKE_C_COMPILER})
status("    C flags (Release):"      ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELEASE})
status("    C flags (Debug):"        ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_DEBUG})
if(WIN32)
  status("    Linker flags (Release):" ${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS_RELEASE})
  status("    Linker flags (Debug):"   ${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS_DEBUG})
else()
  status("    Linker flags (Release):" ${CMAKE_SHARED_LINKER_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS_RELEASE})
  status("    Linker flags (Debug):"   ${CMAKE_SHARED_LINKER_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS_DEBUG})
endif()

# OpenCV modules
status("")
status("  OpenCV modules:")
string(REPLACE "opencv_" "" OPENCV_MODULES_BUILD_ST          "${OPENCV_MODULES_BUILD}")
string(REPLACE "opencv_" "" OPENCV_MODULES_DISABLED_USER_ST  "${OPENCV_MODULES_DISABLED_USER}")
string(REPLACE "opencv_" "" OPENCV_MODULES_DISABLED_FORCE_ST "${OPENCV_MODULES_DISABLED_FORCE}")
set(OPENCV_MODULES_DISABLED_AUTO_ST "")
foreach(m ${OPENCV_MODULES_DISABLED_AUTO})
  set(__mdeps "")
  foreach(d ${OPENCV_MODULE_${m}_DEPS})
    if(d MATCHES "^opencv_" AND NOT HAVE_${d})
      list(APPEND __mdeps ${d})
    endif()
  endforeach()
  list(APPEND OPENCV_MODULES_DISABLED_AUTO_ST "${m}(deps: ${__mdeps})")
endforeach()
string(REPLACE "opencv_" "" OPENCV_MODULES_DISABLED_AUTO_ST  "${OPENCV_MODULES_DISABLED_AUTO_ST}")

status("    To be built:"            OPENCV_MODULES_BUILD          THEN ${OPENCV_MODULES_BUILD_ST}          ELSE "-")
status("    Disabled by user:"       OPENCV_MODULES_DISABLED_USER  THEN ${OPENCV_MODULES_DISABLED_USER_ST}  ELSE "-")
status("    Disabled by dependency:" OPENCV_MODULES_DISABLED_AUTO  THEN ${OPENCV_MODULES_DISABLED_AUTO_ST}  ELSE "-")
status("    Unavailable:"            OPENCV_MODULES_DISABLED_FORCE THEN ${OPENCV_MODULES_DISABLED_FORCE_ST} ELSE "-")

# Android extra
if(ANDROID)
  status("")
  status("  Android: ")
  status("    Android ABI:" ${ANDROID_ABI})
  status("    Native API level:" android-${ANDROID_NATIVE_API_LEVEL})
  status("    SDK target:" "${ANDROID_SDK_TARGET}")
  if(BUILD_WITH_ANDROID_NDK)
    status("    Android NDK:" "${ANDROID_NDK} (toolchain: ${ANDROID_TOOLCHAIN_NAME})")
  elseif(BUILD_WITH_STANDALONE_TOOLCHAIN)
    status("    Android toolchain:" "${ANDROID_STANDALONE_TOOLCHAIN}")
  endif()
  status("    android tool:"  ANDROID_EXECUTABLE  THEN "${ANDROID_EXECUTABLE} (${ANDROID_TOOLS_Pkg_Desc})" ELSE NO)
  status("    ant:"           ANT_EXECUTABLE      THEN "${ANT_EXECUTABLE} (ver ${ANT_VERSION})"            ELSE NO)
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
      status("    GTK+ 2.x:" HAVE_GTK      THEN YES ELSE NO)
      status("    GThread :" HAVE_GTHREAD  THEN YES ELSE NO)
      status("    GtkGlExt:" HAVE_GTKGLEXT THEN YES ELSE NO)
    endif()
  endif()
endif()

status("    OpenGL support:" HAVE_OPENGL THEN YES ELSE NO)

# media
status("")
status("  Media I/O: ")
status("    ZLib:"         ZLIB_FOUND    THEN "${ZLIB_LIBRARY}"    ELSE build)

if(WITH_JPEG)
  status("    JPEG:"       JPEG_FOUND    THEN "${JPEG_LIBRARY}"    ELSE build)
else()
  status("    JPEG:"       "NO")
endif()
if(WITH_PNG)
  status("    PNG:"        PNG_FOUND     THEN "${PNG_LIBRARY}"     ELSE build)
else()
  status("    PNG:"        "NO")
endif()
if(WITH_TIFF)
  status("    TIFF:"       TIFF_FOUND    THEN "${TIFF_LIBRARY}"    ELSE build)
else()
  status("    TIFF:"       "NO")
endif()
if(WITH_JASPER)
  status("    JPEG 2000:"  JASPER_FOUND  THEN "${JASPER_LIBRARY}"  ELSE build)
else()
  status("    JPEG 2000:"  "NO")
endif()

status("    OpenEXR:"   WITH_OPENEXR AND OPENEXR_FOUND     THEN ${OPENEXR_LIBRARIES} ELSE NO)
status("    OpenNI:"    HAVE_OPENNI                        THEN YES             ELSE NO)
status("    OpenNI PrimeSensor Modules:"
                        HAVE_OPENNI_PRIME_SENSOR_MODULE    THEN YES             ELSE NO)
if(WIN32)
  status("    XIMEA:"     HAVE_XIMEA  THEN YES ELSE NO)
endif()

# video
status("")
if(UNIX AND NOT APPLE)
  status("  Video I/O:")
  status("    DC1394 1.x:"     HAVE_DC1394         THEN YES ELSE NO)
  status("    DC1394 2.x:"     HAVE_DC1394_2       THEN YES ELSE NO)
endif()
if(UNIX)
  status("    FFMPEG:"         HAVE_FFMPEG         THEN YES ELSE NO)
  status("      codec:"        HAVE_FFMPEG_CODEC   THEN YES ELSE NO)
  status("      format:"       HAVE_FFMPEG_FORMAT  THEN YES ELSE NO)
  status("      util:"         HAVE_FFMPEG_UTIL    THEN YES ELSE NO)
  status("      swscale:"      HAVE_FFMPEG_SWSCALE THEN YES ELSE NO)
  status("      gentoo-style:" HAVE_GENTOO_FFMPEG  THEN YES ELSE NO)
endif()
if(UNIX AND NOT APPLE)
  status("    GStreamer:"      HAVE_GSTREAMER      THEN YES ELSE NO)
  status("    UniCap:"         HAVE_UNICAP         THEN YES ELSE NO)
  status("    PvAPI:"          HAVE_PVAPI          THEN YES ELSE NO)
  status("    V4L/V4L2:"       HAVE_LIBV4L         THEN "Using libv4l" ELSE ${HAVE_CAMV4L}/${HAVE_CAMV4L2})
  status("    Xine:"           HAVE_XINE           THEN YES ELSE NO)

  if(ANDROID)
    if(HAVE_opencv_androidcamera)
      status("    AndroidNativeCamera:" BUILD_ANDROID_CAMERA_WRAPPER THEN "YES, build for Android ${ANDROID_VERSION}" ELSE "YES, use prebuilt libraries")
    else()
      status("    AndroidNativeCamera:" "NO (native camera requires Android API level 8 or higher)")
    endif()
  endif()
elseif(APPLE)
  if(NOT IOS)
    status("  Video I/O:"    WITH_QUICKTIME      THEN QuickTime ELSE QTKit)
  else()
    status("  Video I/O: AVFoundation")
  endif()
elseif(WIN32)
  status("  Video I/O:"      HAVE_VIDEOINPUT     THEN DirectShow ELSE NO)
endif()

# Other third-party libraries
status("")
status("  Other third-party libraries:")

if(WITH_IPP AND IPP_FOUND)
  status("    Use IPP:" "${IPP_LATEST_VERSION_STR} [${IPP_LATEST_VERSION_MAJOR}.${IPP_LATEST_VERSION_MINOR}.${IPP_LATEST_VERSION_BUILD}]")
  status("         at:" "${IPP_ROOT_DIR}")
else()
  status("    Use IPP:"   WITH_IPP AND NOT IPP_FOUND THEN "IPP not found" ELSE NO)
endif()

status("    Use TBB:"   HAVE_TBB   THEN YES ELSE NO)
status("    Use Cuda:"  HAVE_CUDA  THEN YES ELSE NO)
status("    Use Eigen:" HAVE_EIGEN THEN YES ELSE NO)

# interfaces to other languages
status("")
status("  Python:")
status("    Interpreter:"   PYTHON_EXECUTABLE   THEN "${PYTHON_EXECUTABLE} (ver ${PYTHON_VERSION_FULL})"         ELSE NO)
status("    Libraries:"     HAVE_opencv_python  THEN ${PYTHON_LIBRARIES}                                         ELSE NO)
status("    numpy:"         PYTHON_USE_NUMPY    THEN "${PYTHON_NUMPY_INCLUDE_DIR} (ver ${PYTHON_NUMPY_VERSION})" ELSE "NO (Python wrappers can not be generated)")
status("    packages path:" PYTHON_EXECUTABLE   THEN "${PYTHON_PACKAGES_PATH}"                                   ELSE "-")

if(ANDROID)
  status("")
  status("  Java:" HAVE_opencv_java THEN YES ELSE NO)
endif()

# documentation
if(BUILD_DOCS)
  status("")
  status("  Documentation:")
  status("    Sphinx:"              HAVE_SPHINX              THEN "${SPHINX_BUILD} (ver ${SPHINX_VERSION})" ELSE NO)
  status("    PdfLaTeX compiler:"   PDFLATEX_COMPILER        THEN "${PDFLATEX_COMPILER}" ELSE NO)
  if(HAVE_SPHINX)
    status("    Build Documentation:" PDFLATEX_COMPILER      THEN YES ELSE "YES (only HTML without math expressions)")
  else()
    status("    Build Documentation:" NO)
  endif()
endif()

# samples and tests
status("")
status("  Tests and samples:")
status("    Tests:"             BUILD_TESTS AND HAVE_opencv_ts       THEN YES ELSE NO)
status("    Performance tests:" BUILD_PERF_TESTS AND HAVE_opencv_ts  THEN YES ELSE NO)
status("    Examples:"          BUILD_EXAMPLES                       THEN YES ELSE NO)

if(ANDROID)
  status("    Android tests:"    BUILD_TESTS AND CAN_BUILD_ANDROID_PROJECTS THEN YES ELSE NO)
  status("    Android examples:" BUILD_ANDROID_EXAMPLES                     THEN YES ELSE NO)
endif()

#auxiliary
status("")
status("  Install path:" "${CMAKE_INSTALL_PREFIX}")
status("")
status("  cvconfig.h is in:" "${OPENCV_CONFIG_FILE_INCLUDE_DIR}")
status("-----------------------------------------------------------------")
status("")

# ----------------------------------------------------------------------------
# Warn in the case of in-source build
# ----------------------------------------------------------------------------
if("${CMAKE_CURRENT_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_BINARY_DIR}")
  message(WARNING "The source directory is the same as binary directory. \"make clean\" may damage the source tree")
endif()

