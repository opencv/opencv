# ----------------------------------------------------------------------------
#                                  OPENCV
#                        Open Computer Vision Library
#
# From the off-tree build directory, invoke:
#   $ cmake /path/to/opencv/root
#
# ----------------------------------------------------------------------------
include(cmake/OpenCVMinDepVersions.cmake)
if(CMAKE_GENERATOR MATCHES Xcode AND XCODE_VERSION VERSION_GREATER 4.3)
  cmake_minimum_required(VERSION 2.8.8 FATAL_ERROR)
else()
  cmake_minimum_required(VERSION "${MIN_VER_CMAKE}" FATAL_ERROR)
endif()
include(cmake/OpenCVPreamble.cmake)
project(OpenCV CXX C)


# ----------------------------------------------------------------------------
# USER-CONFIGURABLE OPTIONS
# ----------------------------------------------------------------------------
# There are 7 types of user-configurable components:
#   1. 3rd party source plugins     plugins to build from source
#   2. 3rd party plugins            existing plugins to use/link
#   3. Build components             documentation, tests, performance, etc
#   4. Build options                precompiled headers, solution folders, etc
#   5. Optimization options         SSE, AVX, fast math, debug, etc
#   6. Language bindings            python, java, matlab bindings
#   7. Installation options         mangle paths, RPATHS, etc
# ----------------------------------------------------------------------------
include(cmake/OpenCVUtils.cmake)
include(cmake/OpenCVDetectCXXCompiler.cmake)
include(cmake/OpenCVVersion.cmake)

# 1. 3rd party plugins/components to build from source
# ========================================================
OCV_OPTION(BUILD_JASPER               "Build libjasper from source"        WIN32 OR ANDROID OR APPLE )
OCV_OPTION(BUILD_JPEG                 "Build libjpeg from source"          WIN32 OR ANDROID OR APPLE )
OCV_OPTION(BUILD_OPENEXR              "Build openexr from source"          WIN32 OR ANDROID OR APPLE )
OCV_OPTION(BUILD_PNG                  "Build libpng from source"           WIN32 OR ANDROID OR APPLE )
OCV_OPTION(BUILD_TBB                  "Download and build TBB from source" ANDROID )
OCV_OPTION(BUILD_TIFF                 "Build libtiff from source"          WIN32 OR ANDROID OR APPLE )
OCV_OPTION(BUILD_ZLIB                 "Build zlib from source"             WIN32 OR APPLE )

# 2. 3rd party plugins/components
# ========================================================
# GUI
OCV_OPTION(WITH_CARBON                "Use Carbon for UI instead of Cocoa"          OFF  IF APPLE )
OCV_OPTION(WITH_DSHOW                 "Build HighGUI with DirectShow support"       ON   IF (WIN32 AND NOT ARM) )
OCV_OPTION(WITH_GTK                   "Include GTK support"                         ON   IF (UNIX AND NOT APPLE AND NOT ANDROID) )
OCV_OPTION(WITH_MSMF                  "Build HighGUI with Media Foundation support" OFF  IF WIN32 )
OCV_OPTION(WITH_OPENGL                "Include OpenGL support"                      OFF  IF (NOT ANDROID AND NOT APPLE) )
OCV_OPTION(WITH_QT                    "Build with Qt Backend support"               OFF  IF (NOT ANDROID AND NOT IOS) )
OCV_OPTION(WITH_WIN32UI               "Build with Win32 UI Backend support"         ON   IF WIN32 )
# GPU/concurrency
OCV_OPTION(WITH_CSTRIPES              "Include C= support"                          OFF  IF WIN32 )
OCV_OPTION(WITH_CUDA                  "Include NVidia Cuda Runtime support"                                         ON  IF (NOT ANDROID AND NOT IOS) )
OCV_OPTION(WITH_CUBLAS                "Include NVidia Cuda Basic Linear Algebra Subprograms (BLAS) library support" OFF IF (NOT ANDROID AND NOT IOS) )
OCV_OPTION(WITH_CUFFT                 "Include NVidia Cuda Fast Fourier Transform (FFT) library support"            ON  IF (NOT ANDROID AND NOT IOS) )
OCV_OPTION(WITH_NVCUVID               "Include NVidia Video Decoding library support"                               OFF IF (NOT ANDROID AND NOT IOS AND NOT APPLE) )
OCV_OPTION(WITH_OPENCL                "Include OpenCL Runtime support"              ON  IF (NOT ANDROID AND NOT IOS) )
OCV_OPTION(WITH_OPENCLAMDBLAS         "Include AMD OpenCL BLAS library support"     ON  IF (NOT ANDROID AND NOT IOS) )
OCV_OPTION(WITH_OPENCLAMDFFT          "Include AMD OpenCL FFT library support"      ON  IF (NOT ANDROID AND NOT IOS) )
OCV_OPTION(WITH_TBB                   "Include Intel TBB support"                   OFF  IF (NOT IOS) )
# Video
OCV_OPTION(WITH_AVFOUNDATION          "Use AVFoundation for Video I/O"              ON   IF IOS)
OCV_OPTION(WITH_FFMPEG                "Include FFMPEG support"                      ON   IF (NOT ANDROID AND NOT IOS))
OCV_OPTION(WITH_GSTREAMER             "Include Gstreamer support"                   ON   IF (UNIX AND NOT APPLE AND NOT ANDROID) )
OCV_OPTION(WITH_GSTREAMER_1_X         "Include Gstreamer 1.x support"               OFF)
OCV_OPTION(WITH_QUICKTIME             "Use QuickTime for Video I/O insted of QTKit" OFF  IF APPLE )
OCV_OPTION(WITH_UNICAP                "Include Unicap support (GPL)"                OFF  IF (UNIX AND NOT APPLE AND NOT ANDROID) )
OCV_OPTION(WITH_V4L                   "Include Video 4 Linux support"               ON   IF (UNIX AND NOT ANDROID) )
OCV_OPTION(WITH_VFW                   "Include Video for Windows support"           ON   IF WIN32 )
OCV_OPTION(WITH_XINE                  "Include Xine support (GPL)"                  OFF  IF (UNIX AND NOT APPLE AND NOT ANDROID) )
# Hardware
OCV_OPTION(WITH_1394                  "Include IEEE1394 support"                    ON   IF (NOT ANDROID AND NOT IOS) )
OCV_OPTION(WITH_GIGEAPI               "Include Smartek GigE support"                ON   IF (NOT ANDROID AND NOT IOS) )
OCV_OPTION(WITH_OPENNI                "Include OpenNI support"                      OFF  IF (NOT ANDROID AND NOT IOS) )
OCV_OPTION(WITH_PVAPI                 "Include Prosilica GigE support"              ON   IF (NOT ANDROID AND NOT IOS) )
OCV_OPTION(WITH_XIMEA                 "Include XIMEA cameras support"               OFF  IF (NOT ANDROID AND NOT APPLE) )
# Codecs
OCV_OPTION(WITH_JASPER                "Include JPEG2K support"                      ON   IF (NOT IOS) )
OCV_OPTION(WITH_JPEG                  "Include JPEG support"                        ON)
OCV_OPTION(WITH_OPENEXR               "Include ILM support via OpenEXR"             ON   IF (NOT IOS) )
OCV_OPTION(WITH_PNG                   "Include PNG support"                         ON)
OCV_OPTION(WITH_TIFF                  "Include TIFF support"                        ON   IF (NOT IOS) )
OCV_OPTION(WITH_WEBP                  "Include WebP support"                        ON   IF (NOT IOS) )
# Core extensions
OCV_OPTION(WITH_CLP                   "Include Clp support (EPL)"                   OFF)
OCV_OPTION(WITH_EIGEN                 "Include Eigen2/Eigen3 support"               ON)
OCV_OPTION(WITH_IPP                   "Include Intel IPP support"                   OFF  IF (MSVC OR X86 OR X86_64) )

# 3. Build components
# ========================================================
OCV_OPTION(BUILD_ANDROID_EXAMPLES     "Build examples for Android platform"         ON  IF ANDROID )
OCV_OPTION(BUILD_ANDROID_PACKAGE      "Build platform-specific package for Google Play" OFF IF ANDROID )
OCV_OPTION(BUILD_ANDROID_SERVICE      "Build OpenCV Manager for Google Play" OFF IF ANDROID AND ANDROID_SOURCE_TREE )
OCV_OPTION(BUILD_DOCS                 "Create build rules for OpenCV Documentation" OFF )
OCV_OPTION(BUILD_EXAMPLES             "Build all examples"                          OFF )
OCV_OPTION(BUILD_PACKAGE              "Enables 'make package_source' command"       ON  )
OCV_OPTION(BUILD_PERF_TESTS           "Build performance tests"                     OFF )
OCV_OPTION(BUILD_SHARED_LIBS          "Build shared libraries (.dll/.so/.dylib) instead of static ones (.lib/.a)" NOT (ANDROID OR IOS) )
OCV_OPTION(BUILD_TESTS                "Build accuracy & regression tests"           OFF )
OCV_OPTION(BUILD_WITH_DEBUG_INFO      "Include debug info into debug libs (not MSCV only)" ON )
OCV_OPTION(BUILD_WITH_STATIC_CRT      "Enables use of staticaly linked CRT for staticaly linked OpenCV" ON IF MSVC )
OCV_OPTION(BUILD_opencv_apps          "Build utility applications (used for example to train classifiers)" (NOT ANDROID) IF (NOT IOS) )

# 4. Build options
# ========================================================
OCV_OPTION(ENABLE_NOISY_WARNINGS      "Show all warnings even if they are too noisy"             OFF )
OCV_OPTION(ENABLE_OMIT_FRAME_POINTER  "Enable -fomit-frame-pointer for GCC"                      ON   IF CMAKE_COMPILER_IS_GNUCXX AND NOT (APPLE AND CMAKE_COMPILER_IS_CLANGCXX) )
OCV_OPTION(ENABLE_POWERPC             "Enable PowerPC for GCC"                                   ON   IF (CMAKE_COMPILER_IS_GNUCXX AND CMAKE_SYSTEM_PROCESSOR MATCHES powerpc.*) )
OCV_OPTION(ENABLE_PRECOMPILED_HEADERS "Use precompiled headers"                                  ON   IF (NOT IOS) )
OCV_OPTION(ENABLE_PROFILING           "Enable profiling in the GCC compiler (Add flags: -g -pg)" OFF  IF CMAKE_COMPILER_IS_GNUCXX )
OCV_OPTION(ENABLE_SOLUTION_FOLDERS    "Solution folder in Visual Studio or in other IDEs"        (MSVC_IDE OR CMAKE_GENERATOR MATCHES Xcode) )
OCV_OPTION(ENABLE_WINRT_MODE          "Build with Windows Runtime support"                       OFF  IF WIN32 )
OCV_OPTION(OPENCV_WARNINGS_ARE_ERRORS "Treat warnings as errors"                                 OFF )

# 5. Optimization options
# ========================================================
OCV_OPTION(ENABLE_FAST_MATH           "Enable -ffast-math (not recommended for GCC 4.6.x)"       OFF  IF (CMAKE_COMPILER_IS_GNUCXX AND (X86 OR X86_64)) )
OCV_OPTION(ENABLE_SSE                 "Enable SSE instructions"                                  ON   IF ((MSVC OR CMAKE_COMPILER_IS_GNUCXX) AND (X86 OR X86_64)) )
OCV_OPTION(ENABLE_SSE2                "Enable SSE2 instructions"                                 ON   IF ((MSVC OR CMAKE_COMPILER_IS_GNUCXX) AND (X86 OR X86_64)) )
OCV_OPTION(ENABLE_SSE3                "Enable SSE3 instructions"                                 ON   IF ((CV_ICC OR CMAKE_COMPILER_IS_GNUCXX) AND (X86 OR X86_64)) )
OCV_OPTION(ENABLE_SSE41               "Enable SSE4.1 instructions"                               OFF  IF ((CV_ICC OR CMAKE_COMPILER_IS_GNUCXX) AND (X86 OR X86_64)) )
OCV_OPTION(ENABLE_SSE42               "Enable SSE4.2 instructions"                               OFF  IF (CMAKE_COMPILER_IS_GNUCXX AND (X86 OR X86_64)) )
OCV_OPTION(ENABLE_SSSE3               "Enable SSSE3 instructions"                                OFF  IF (CMAKE_COMPILER_IS_GNUCXX AND (X86 OR X86_64)) )
OCV_OPTION(ENABLE_AVX                 "Enable AVX instructions"                                  OFF  IF ((MSVC OR CMAKE_COMPILER_IS_GNUCXX) AND (X86 OR X86_64)) )

# 6. Language bindings
# ========================================================
OCV_OPTION(BUILD_JAVA_BINDINGS        "Build java language bindings"    OFF)
OCV_OPTION(BUILD_JAVA_FAT_LIB         "Create fat java wrapper containing the whole OpenCV library" OFF)
OCV_OPTION(BUILD_MATLAB_BINDINGS      "Build matlab language bindings"  OFF)
OCV_OPTION(BUILD_PYTHON_BINDINGS      "Build python language bindings"  OFF)

# 7. Installation options
# ========================================================
OCV_OPTION(INSTALL_ANDROID_EXAMPLES "Install Android examples"  OFF IF ANDROID )
OCV_OPTION(INSTALL_C_EXAMPLES       "Install C examples"        OFF )
OCV_OPTION(INSTALL_PYTHON_EXAMPLES  "Install Python examples"   OFF )
OCV_OPTION(INSTALL_TO_MANGLED_PATHS "Enables mangled install paths, that help with side by side installs." OFF IF (UNIX AND NOT ANDROID AND NOT IOS AND BUILD_SHARED_LIBS) )


# ----------------------------------------------------------------------------
#  END USER-CONFIGURABLE OPTIONS
# ----------------------------------------------------------------------------
include(cmake/OpenCVConfigure.cmake)


# ----------------------------------------------------------------------------
#  Summary:
# ----------------------------------------------------------------------------
status("")
status("==============================================================================")
status("                                 OpenCV ${OPENCV_VERSION}")
status("                        General configuration summary")
status("==============================================================================")
status("")
if(OPENCV_VCSVERSION)
  status("  Version Control:")
  status("  ----------------")
  status("    git:" ${OPENCV_VCSVERSION})
endif()

# ============================== Build Platform ==============================
status("")
status("  Platform:")
status("  ---------")

status("    Host:"             ${CMAKE_HOST_SYSTEM_NAME} ${CMAKE_HOST_SYSTEM_VERSION} ${CMAKE_HOST_SYSTEM_PROCESSOR})
if(CMAKE_CROSSCOMPILING)
  status("    Target:"         ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_VERSION} ${CMAKE_SYSTEM_PROCESSOR})
endif()
status("    CMake:"            ${CMAKE_VERSION})
status("    CMake generator:"  ${CMAKE_GENERATOR})
status("    CMake build tool:" ${CMAKE_BUILD_TOOL})
if(MSVC)
  status("    MSVC:"           ${MSVC_VERSION})
endif()
if(CMAKE_GENERATOR MATCHES Xcode)
  status("    Xcode:"          ${XCODE_VERSION})
endif()
if(NOT CMAKE_GENERATOR MATCHES "Xcode|Visual Studio")
  status("    Configuration:"  ${CMAKE_BUILD_TYPE})
endif()

# ============================== C/C++ Options ===============================
status("")
status("  C/C++:")
status("  ------")

if(CMAKE_CXX_COMPILER_VERSION)
  set(OPENCV_COMPILER_STR "${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_ARG1} (ver ${CMAKE_CXX_COMPILER_VERSION})")
elseif(CMAKE_COMPILER_IS_CLANGCXX)
  set(OPENCV_COMPILER_STR "${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_ARG1} (ver ${CMAKE_CLANG_REGEX_VERSION})")
elseif(CMAKE_COMPILER_IS_GNUCXX)
  set(OPENCV_COMPILER_STR "${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_ARG1} (ver ${CMAKE_GCC_REGEX_VERSION})")
else()
  set(OPENCV_COMPILER_STR "${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_ARG1}")
endif()
string(STRIP "${OPENCV_COMPILER_STR}" OPENCV_COMPILER_STR)

status("    Built as dynamic libs?:" BUILD_SHARED_LIBS THEN YES ELSE NO)
status("    C++ Compiler:"           ${OPENCV_COMPILER_STR})
status("    C++ flags (Release):"    ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE})
status("    C++ flags (Debug):"      ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG})
status("    C Compiler:"             ${CMAKE_C_COMPILER} ${CMAKE_C_COMPILER_ARG1})
status("    C flags (Release):"      ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELEASE})
status("    C flags (Debug):"        ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_DEBUG})
if(WIN32)
  status("    Linker flags (Release):" ${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS_RELEASE})
  status("    Linker flags (Debug):"   ${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS_DEBUG})
else()
  status("    Linker flags (Release):" ${CMAKE_SHARED_LINKER_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS_RELEASE})
  status("    Linker flags (Debug):"   ${CMAKE_SHARED_LINKER_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS_DEBUG})
endif()
status("    Precompiled headers:"     PCHSupport_FOUND AND ENABLE_PRECOMPILED_HEADERS THEN YES ELSE NO)

# ============================== OpenCV Modules ==============================
status("")
status("  OpenCV Modules:")
status("  ---------------")

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
  if(__mdeps)
    list(APPEND OPENCV_MODULES_DISABLED_AUTO_ST "${m}(deps: ${__mdeps})")
  else()
    list(APPEND OPENCV_MODULES_DISABLED_AUTO_ST "${m}")
  endif()
endforeach()
string(REPLACE "opencv_" "" OPENCV_MODULES_DISABLED_AUTO_ST  "${OPENCV_MODULES_DISABLED_AUTO_ST}")

status("    To be built:"            OPENCV_MODULES_BUILD          THEN ${OPENCV_MODULES_BUILD_ST}          ELSE "-")
status("    Disabled:"               OPENCV_MODULES_DISABLED_USER  THEN ${OPENCV_MODULES_DISABLED_USER_ST}  ELSE "-")
status("    Disabled by dependency:" OPENCV_MODULES_DISABLED_AUTO  THEN ${OPENCV_MODULES_DISABLED_AUTO_ST}  ELSE "-")
status("    Unavailable:"            OPENCV_MODULES_DISABLED_FORCE THEN ${OPENCV_MODULES_DISABLED_FORCE_ST} ELSE "-")

# ============================= Android Details ==============================
if(ANDROID)
  status("")
  status("  Android:")
  status("  --------")

  status("    Android ABI:" ${ANDROID_ABI})
  status("    STL type:" ${ANDROID_STL})
  status("    Native API level:" android-${ANDROID_NATIVE_API_LEVEL})
  android_get_compatible_target(android_sdk_target_status ${ANDROID_NATIVE_API_LEVEL} ${ANDROID_SDK_TARGET} 11)
  status("    SDK target:" "${android_sdk_target_status}")
  if(BUILD_WITH_ANDROID_NDK)
    status("    Android NDK:" "${ANDROID_NDK} (toolchain: ${ANDROID_TOOLCHAIN_NAME})")
  elseif(BUILD_WITH_STANDALONE_TOOLCHAIN)
    status("    Android toolchain:" "${ANDROID_STANDALONE_TOOLCHAIN}")
  endif()
  status("    android tool:"  ANDROID_EXECUTABLE  THEN "${ANDROID_EXECUTABLE} (${ANDROID_TOOLS_Pkg_Desc})" ELSE NO)
  status("    Google Play package:" BUILD_ANDROID_PACKAGE                                         THEN YES ELSE NO)
  status("    Android examples:"    BUILD_ANDROID_EXAMPLES AND CAN_BUILD_ANDROID_PROJECTS         THEN YES ELSE NO)
endif()

# ============================ Windows RT features ===========================
if(WIN32)
status("")
    status("  Windows RT Support:" HAVE_WINRT THEN YES ELSE NO)
    status("  -------------------")

    if (ENABLE_WINRT_MODE)
      status("    Windows SDK v8.0:" ${WINDOWS_SDK_PATH})
      status("    Visual Studio 2012:" ${VISUAL_STUDIO_PATH})
    endif()
endif(WIN32)

# =================================== GUI ====================================
status("")
status("  GUI:")
status("  ----")

if(HAVE_QT5)
  status("    QT 5.x:"            HAVE_QT        THEN "YES (ver ${Qt5Core_VERSION_STRING})" ELSE NO)
  status("    QT OpenGL support:" HAVE_QT_OPENGL THEN "YES (${Qt5OpenGL_LIBRARIES} ${Qt5OpenGL_VERSION_STRING})" ELSE NO)
elseif(HAVE_QT)
  status("    QT 4.x:"            HAVE_QT        THEN "YES (ver ${QT_VERSION_MAJOR}.${QT_VERSION_MINOR}.${QT_VERSION_PATCH} ${QT_EDITION})" ELSE NO)
  status("    QT OpenGL support:" HAVE_QT_OPENGL THEN "YES (${QT_QTOPENGL_LIBRARY})" ELSE NO)
else()
  if(DEFINED WITH_QT)
    status("    QT:" NO)
  endif()
  if(DEFINED WITH_WIN32UI)
    status("    Win32 UI:" HAVE_WIN32UI THEN YES ELSE NO)
  else()
    if(APPLE)
      if(WITH_CARBON)
        status("    Carbon:" YES)
      else()
        status("    Cocoa:"  YES)
      endif()
    else()
      status("    GTK+ 2.x:" HAVE_GTK      THEN "YES (ver ${ALIASOF_gtk+-2.0_VERSION})"     ELSE NO)
      status("    GThread :" HAVE_GTHREAD  THEN "YES (ver ${ALIASOF_gthread-2.0_VERSION})"  ELSE NO)
      status("    GtkGlExt:" HAVE_GTKGLEXT THEN "YES (ver ${ALIASOF_gtkglext-1.0_VERSION})" ELSE NO)
    endif()
  endif()
endif()

status("    OpenGL support:" HAVE_OPENGL THEN "YES (${OPENGL_LIBRARIES})" ELSE NO)

# ================================== Codecs ==================================
status("")
status("  Codecs:")
status("  -------")

status("    ZLib:" BUILD_ZLIB THEN "build (ver ${ZLIB_VERSION_STRING})" ELSE "${ZLIB_LIBRARIES} (ver ${ZLIB_VERSION_STRING})")

if(WITH_JPEG)
  status("    JPEG:" JPEG_FOUND THEN "${JPEG_LIBRARY} (ver ${JPEG_LIB_VERSION})" ELSE "build (ver ${JPEG_LIB_VERSION})")
else()
  status("    JPEG:" "NO")
endif()

if(WITH_WEBP)
  status("    WEBP:" WEBP_FOUND THEN "${WEBP_LIBRARY} (ver ${WEBP_VERSION})" ELSE "build (ver ${WEBP_VERSION})")
else()
  status("    WEBP:" "NO")
endif()

if(WITH_PNG)
  status("    PNG:" PNG_FOUND THEN "${PNG_LIBRARY} (ver ${PNG_VERSION})" ELSE "build (ver ${PNG_VERSION})")
else()
  status("    PNG:" "NO")
endif()

if(WITH_TIFF)
  if(TIFF_VERSION_STRING AND TIFF_FOUND)
    status("    TIFF:" "${TIFF_LIBRARY} (ver ${TIFF_VERSION} - ${TIFF_VERSION_STRING})")
  else()
    status("    TIFF:"  TIFF_FOUND THEN "${TIFF_LIBRARY} (ver ${TIFF_VERSION})" ELSE "build (ver ${TIFF_VERSION} - ${TIFF_VERSION_STRING})")
  endif()
else()
  status("    TIFF:" "NO")
endif()

if(WITH_JASPER)
  status("    JPEG 2000:" JASPER_FOUND THEN "${JASPER_LIBRARY} (ver ${JASPER_VERSION_STRING})" ELSE "build (ver ${JASPER_VERSION_STRING})")
else()
  status("    JPEG 2000:" "NO")
endif()

if(WITH_OPENEXR)
  status("    OpenEXR:" OPENEXR_FOUND THEN "${OPENEXR_LIBRARIES} (ver ${OPENEXR_VERSION})" ELSE "build (ver ${OPENEXR_VERSION})")
else()
  status("    OpenEXR:"  "NO")
endif()

# ============================= Video Frameworks =============================
status("")
status("  Video Frameworks:")
status("  -----------------")

if (DEFINED WITH_VFW)
  status("    Video for Windows:" HAVE_VFW         THEN YES                                        ELSE NO)
endif(DEFINED WITH_VFW)

if(DEFINED WITH_AVFOUNDATION)
  status("    AVFoundation:"   WITH_AVFOUNDATION   THEN YES                                        ELSE NO)
endif(DEFINED WITH_AVFOUNDATION)

if(DEFINED WITH_FFMPEG)
  if(WIN32)
    status("    FFMPEG:"       WITH_FFMPEG         THEN "YES (prebuilt binaries)"                  ELSE NO)
  else()
    status("    FFMPEG:"       HAVE_FFMPEG         THEN YES ELSE NO)
  endif()
  status("      codec:"        HAVE_FFMPEG_CODEC   THEN "YES (ver ${ALIASOF_libavcodec_VERSION})"  ELSE NO)
  status("      format:"       HAVE_FFMPEG_FORMAT  THEN "YES (ver ${ALIASOF_libavformat_VERSION})" ELSE NO)
  status("      util:"         HAVE_FFMPEG_UTIL    THEN "YES (ver ${ALIASOF_libavutil_VERSION})"   ELSE NO)
  status("      swscale:"      HAVE_FFMPEG_SWSCALE THEN "YES (ver ${ALIASOF_libswscale_VERSION})"  ELSE NO)
  status("      gentoo-style:" HAVE_GENTOO_FFMPEG  THEN YES                                        ELSE NO)
endif(DEFINED WITH_FFMPEG)

if(DEFINED WITH_GSTREAMER)
  status("    GStreamer:"      HAVE_GSTREAMER      THEN ""                                         ELSE NO)
  if(HAVE_GSTREAMER)
    status("      base:"       "YES (ver ${GSTREAMER_BASE_VERSION})")
    status("      video:"      "YES (ver ${GSTREAMER_VIDEO_VERSION})")
    status("      app:"        "YES (ver ${GSTREAMER_APP_VERSION})")
    status("      riff:"       "YES (ver ${GSTREAMER_RIFF_VERSION})")
    status("      pbutils:"    "YES (ver ${GSTREAMER_PBUTILS_VERSION})")
  endif(HAVE_GSTREAMER)
endif(DEFINED WITH_GSTREAMER)

if(DEFINED WITH_QUICKTIME)
  status("    QuickTime:"      HAVE_QUICKTIME   THEN YES    ELSE NO)
  status("    QTKit:"          HAVE_QTKIT       THEN YES    ELSE NO)
endif(DEFINED WITH_QUICKTIME)

if(DEFINED WITH_DSHOW)
  status("    DirectShow:"     HAVE_DSHOW       THEN YES    ELSE NO)
endif(DEFINED WITH_DSHOW)

if(DEFINED WITH_MSMF)
  status("    Media Foundation:" HAVE_MSMF      THEN YES    ELSE NO)
endif(DEFINED WITH_MSMF)

if(DEFINED WITH_XINE)
  status("    Xine:"           HAVE_XINE         THEN "YES (ver ${ALIASOF_libxine_VERSION})" ELSE NO)
endif(DEFINED WITH_XINE)

# ============================= Hardware Support =============================
status("")
status("  Hardware/device Support:")
status("  ------------------------")

if(ANDROID)
  if(HAVE_opencv_androidcamera)
    status("    AndroidNativeCamera:" BUILD_ANDROID_CAMERA_WRAPPER
                                                   THEN "YES, build for Android${ANDROID_VERSION}" ELSE "YES, use prebuilt libraries")
  else()
    status("    AndroidNativeCamera:" "NO (native camera requires Android API level 8 or higher)")
  endif()
endif()

if(DEFINED WITH_1394)
  status("    DC1394 1.x:"     HAVE_DC1394         THEN "YES (ver ${ALIASOF_libdc1394_VERSION})"   ELSE NO)
  status("    DC1394 2.x:"     HAVE_DC1394_2       THEN "YES (ver ${ALIASOF_libdc1394-2_VERSION})" ELSE NO)
endif(DEFINED WITH_1394)

if(DEFINED WITH_OPENNI)
  status("    OpenNI:"         HAVE_OPENNI         THEN "YES (ver ${OPENNI_VERSION_STRING}, build ${OPENNI_VERSION_BUILD})"
                                                                                                   ELSE NO)
  status("    OpenNI PrimeSensor Modules:" HAVE_OPENNI_PRIME_SENSOR_MODULE
                                                   THEN "YES (${OPENNI_PRIME_SENSOR_MODULE})"      ELSE NO)
endif(DEFINED WITH_OPENNI)

if(DEFINED WITH_UNICAP)
  status("    UniCap:"         HAVE_UNICAP         THEN "YES (ver ${ALIASOF_libunicap_VERSION})"   ELSE NO)
  status("    UniCap ucil:"    HAVE_UNICAP_UCIL    THEN "YES (ver ${ALIASOF_libucil_VERSION})"     ELSE NO)
endif(DEFINED WITH_UNICAP)

if(DEFINED WITH_PVAPI)
  status("    PvAPI:"          HAVE_PVAPI          THEN YES                                        ELSE NO)
endif(DEFINED WITH_PVAPI)

if(DEFINED WITH_GIGEAPI)
  status("    GigEVisionSDK:"  HAVE_GIGE_API       THEN YES                                        ELSE NO)
endif(DEFINED WITH_GIGEAPI)

if(DEFINED WITH_V4L)
  if(HAVE_CAMV4L)
    set(HAVE_CAMV4L_STR "YES")
  else()
    set(HAVE_CAMV4L_STR "NO")
  endif()
  if(HAVE_CAMV4L2)
    set(HAVE_CAMV4L2_STR "YES")
  elseif(HAVE_VIDEOIO)
    set(HAVE_CAMV4L2_STR "YES(videoio)")
  else()
    set(HAVE_CAMV4L2_STR "NO")
  endif()
  status("    V4L/V4L2:"       HAVE_LIBV4L         THEN "Using libv4l (ver ${ALIASOF_libv4l1_VERSION})"
                                                   ELSE "${HAVE_CAMV4L_STR}/${HAVE_CAMV4L2_STR}")
endif(DEFINED WITH_V4L)

if(DEFINED WITH_XIMEA)
  status("    XIMEA:"          HAVE_XIMEA          THEN YES                                        ELSE NO)
endif(DEFINED WITH_XIMEA)

# ============================= GPU/Concurrency ==============================
status("")
status("  GPU/Concurrency:")
status("  ----------------")

status("    Use TBB:"         HAVE_TBB         THEN "YES (ver ${TBB_VERSION_MAJOR}.${TBB_VERSION_MINOR} interface ${TBB_INTERFACE_VERSION})" ELSE NO)
status("    Use OpenMP:"      HAVE_OPENMP      THEN YES ELSE NO)
status("    Use GCD"          HAVE_GCD         THEN YES ELSE NO)
status("    Use Concurrency"  HAVE_CONCURRENCY THEN YES ELSE NO)
status("    Use C=:"          HAVE_CSTRIPES    THEN YES ELSE NO)
# NVIDIA CUDA
status("    Use NVIDIA Cuda:" HAVE_CUDA        THEN "YES (ver ${CUDA_VERSION_STRING})" ELSE NO)
if(HAVE_CUDA)
  status("      Use CUFFT:"            HAVE_CUFFT   THEN YES ELSE NO)
  status("      Use CUBLAS:"           HAVE_CUBLAS  THEN YES ELSE NO)
  status("      USE NVCUVID:"          HAVE_NVCUVID THEN YES ELSE NO)
  status("      NVIDIA GPU arch:"      ${OPENCV_CUDA_ARCH_BIN})
  status("      NVIDIA PTX archs:"     ${OPENCV_CUDA_ARCH_PTX})
  status("      Use fast math:"        CUDA_FAST_MATH THEN YES ELSE NO)
endif()
# OpenCL
status("    Use OpenCL:"      HAVE_OPENCL      THEN YES ELSE NO)
if(HAVE_OPENCL)
  if(OPENCL_INCLUDE_DIR)
    status("      Include path:"       ${OPENCL_INCLUDE_DIRS})
  endif()
  if(OPENCL_LIBRARIES)
    status("      libraries:"          ${OPENCL_LIBRARIES})
  endif()
  status("      Use AMDFFT:"           HAVE_CLAMDFFT  THEN YES ELSE NO)
  status("      Use AMDBLAS:"          HAVE_CLAMDBLAS THEN YES ELSE NO)
endif()

# ============================= Core Extensions ==============================
status("")
status("  Core Extensions:")
status("  ----------------")

status("    Use Eigen:" HAVE_EIGEN THEN "YES (ver ${EIGEN_WORLD_VERSION}.${EIGEN_MAJOR_VERSION}.${EIGEN_MINOR_VERSION})" ELSE NO)
if(WITH_IPP AND IPP_FOUND)
  status("    Use IPP:" "${IPP_LATEST_VERSION_STR} [${IPP_LATEST_VERSION_MAJOR}.${IPP_LATEST_VERSION_MINOR}.${IPP_LATEST_VERSION_BUILD}]")
  status("         at:" "${IPP_ROOT_DIR}")
else()
  status("    Use IPP:" WITH_IPP AND NOT IPP_FOUND  THEN "IPP not found" ELSE NO)
endif()

# ============================ Language Bindings =============================
status("")
status("  Language Bindings:")
status("  ------------------")

# Python
if(BUILD_PYTHON_BINDINGS)
  status("    Python:")
  status("      Interpreter:"   PYTHONINTERP_FOUND  THEN "${PYTHON_EXECUTABLE} (ver ${PYTHON_VERSION_STRING})"       ELSE NO)
  if(PYTHONLIBS_VERSION_STRING)
    status("      Libraries:"   HAVE_opencv_python  THEN  "${PYTHON_LIBRARIES} (ver ${PYTHONLIBS_VERSION_STRING})"   ELSE NO)
  else()
    status("      Libraries:"   HAVE_opencv_python  THEN  "${PYTHON_LIBRARIES}"                                      ELSE NO)
  endif()
  status("      packages path:" PYTHON_EXECUTABLE         THEN "${PYTHON_PACKAGES_PATH}"                                    ELSE "-")
  status("      numpy:"         PYTHON_NUMPY_INCLUDE_DIRS THEN "${PYTHON_NUMPY_INCLUDE_DIRS} (ver ${PYTHON_NUMPY_VERSION})" ELSE "NO (bindings will not be generated)")
else()
  status("    Python:" "NO")
endif()

# Java/Scala
if (BUILD_JAVA_BINDINGS)
  status("    Java:")
  status("      ant:"           ANT_EXECUTABLE      THEN "${ANT_EXECUTABLE} (ver ${ANT_VERSION})"                    ELSE NO)
  if(NOT ANDROID)
    status("      JNI:"         JNI_INCLUDE_DIRS    THEN "${JNI_INCLUDE_DIRS}"                                       ELSE NO)
  endif()
  status("      Java tests:"    BUILD_TESTS AND (CAN_BUILD_ANDROID_PROJECTS OR HAVE_opencv_java)            THEN YES ELSE NO)
else()
  status("    Java:" "NO")
endif()

# Matlab
if (BUILD_MATLAB_BINDINGS)
  status("    Matlab:")
  status("      mex:"         MATLAB_MEX_SCRIPT     THEN  "${MATLAB_MEX_SCRIPT}"   ELSE NO)
  if (MATLAB_FOUND)
    status("      Compiler/generator:" MEX_WORKS    THEN  "Working"                ELSE "Not working (bindings will not be generated)")
  endif()
else()
  status("    Matlab:" "NO")
endif()

# ============================== Documentation ===============================
status("")
status("  Documentation:")
status("  --------------")

if(BUILD_DOCS AND HAVE_SPHINX)
  status("    Build Documentation:" PDFLATEX_COMPILER  THEN YES ELSE "YES (only HTML and without math expressions)")
  status("    Sphinx:"              HAVE_SPHINX        THEN "${SPHINX_BUILD} (ver ${SPHINX_VERSION})" ELSE NO)
  status("    PdfLaTeX compiler:"   PDFLATEX_COMPILER  THEN "${PDFLATEX_COMPILER}" ELSE NO)
  status("    PlantUML:"            PLANTUML           THEN "${PLANTUML}" ELSE NO)
else()
  status("    Build Documentation:" NO)
endif()

# ============================= Tests and Samples ============================
status("")
status("  Tests and Samples:")
status("  ------------------")

status("    Tests:"             BUILD_TESTS AND HAVE_opencv_ts       THEN YES ELSE NO)
status("    Performance tests:" BUILD_PERF_TESTS AND HAVE_opencv_ts  THEN YES ELSE NO)
status("    C/C++ Examples:"    BUILD_EXAMPLES                       THEN YES ELSE NO)

# ================================ Auxiliary =================================
status("")
status("  Install path:" "${CMAKE_INSTALL_PREFIX}")
status("  Config file: " "${OPENCV_CONFIG_FILE_INCLUDE_DIR}/cvconfig.h")
status("==============================================================================")
status("")


# ----------------------------------------------------------------------------
# FINALIZE AND PRINT
# ----------------------------------------------------------------------------
ocv_finalize_status()
