# ----------------------------------------------------------------------------
#  Root CMake file for OpenCV
#
#    From the off-tree build directory, invoke:
#      $ cmake <PATH_TO_OPENCV_ROOT>
#
# ----------------------------------------------------------------------------



include(cmake/OpenCVMinDepVersions.cmake)

if(CMAKE_GENERATOR MATCHES Xcode AND XCODE_VERSION VERSION_GREATER 4.3)
  cmake_minimum_required(VERSION 2.8.8 FATAL_ERROR)
elseif(CMAKE_SYSTEM_NAME MATCHES WindowsPhone OR CMAKE_SYSTEM_NAME MATCHES WindowsStore)
  cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
  #Required to resolve linker error issues due to incompatibility with CMake v3.0+ policies.
  #CMake fails to find _fseeko() which leads to subsequent linker error.
  #See details here: http://www.cmake.org/Wiki/CMake/Policies
  cmake_policy(VERSION 2.8)
else()
  cmake_minimum_required(VERSION "${MIN_VER_CMAKE}" FATAL_ERROR)
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
else(NOT CMAKE_TOOLCHAIN_FILE)
  #Android: set output folder to ${CMAKE_BINARY_DIR}
  set( LIBRARY_OUTPUT_PATH_ROOT ${CMAKE_BINARY_DIR} CACHE PATH "root for library output, set this to change where android libs are compiled to" )
  # any crosscompiling
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory")
endif(NOT CMAKE_TOOLCHAIN_FILE)

if(CMAKE_SYSTEM_NAME MATCHES WindowsPhone OR CMAKE_SYSTEM_NAME MATCHES WindowsStore)
  set(WINRT TRUE)
endif(CMAKE_SYSTEM_NAME MATCHES WindowsPhone OR CMAKE_SYSTEM_NAME MATCHES WindowsStore)

if(WINRT)
  add_definitions(-DWINRT -DNO_GETENV)

  # Making definitions available to other configurations and
  # to filter dependency restrictions at compile time.
  if(CMAKE_SYSTEM_NAME MATCHES WindowsPhone)
    set(WINRT_PHONE TRUE)
    add_definitions(-DWINRT_PHONE)
  elseif(CMAKE_SYSTEM_NAME MATCHES WindowsStore)
    set(WINRT_STORE TRUE)
    add_definitions(-DWINRT_STORE)
  endif()

  if(CMAKE_SYSTEM_VERSION MATCHES 10)
    set(WINRT_10 TRUE)
    add_definitions(-DWINRT_10)
  elseif(CMAKE_SYSTEM_VERSION MATCHES 8.1)
    set(WINRT_8_1 TRUE)
    add_definitions(-DWINRT_8_1)
  elseif(CMAKE_SYSTEM_VERSION MATCHES 8.0)
    set(WINRT_8_0 TRUE)
    add_definitions(-DWINRT_8_0)
  endif()
endif()

if(POLICY CMP0020)
  cmake_policy(SET CMP0020 OLD)
endif()

if(POLICY CMP0022)
  cmake_policy(SET CMP0022 OLD)
endif()

if(POLICY CMP0026)
  # silence cmake 3.0+ warnings about reading LOCATION attribute
  cmake_policy(SET CMP0026 OLD)
endif()

if(POLICY CMP0042)
  cmake_policy(SET CMP0042 NEW)
endif()

include(cmake/OpenCVUtils.cmake)

# must go before the project command
ocv_update(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Configs" FORCE)
if(DEFINED CMAKE_BUILD_TYPE)
  set_property( CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${CMAKE_CONFIGURATION_TYPES} )
endif()

enable_testing()

project(OpenCV CXX C)

if(MSVC)
  set(CMAKE_USE_RELATIVE_PATHS ON CACHE INTERNAL "" FORCE)
endif()

ocv_cmake_eval(DEBUG_PRE ONCE)

ocv_clear_vars(OpenCVModules_TARGETS)

# ----------------------------------------------------------------------------
# Break in case of popular CMake configuration mistakes
# ----------------------------------------------------------------------------
if(NOT CMAKE_SIZEOF_VOID_P GREATER 0)
  message(FATAL_ERROR "CMake fails to deterimine the bitness of target platform.
  Please check your CMake and compiler installation. If you are crosscompiling then ensure that your CMake toolchain file correctly sets the compiler details.")
endif()

# ----------------------------------------------------------------------------
# Detect compiler and target platform architecture
# ----------------------------------------------------------------------------
include(cmake/OpenCVDetectCXXCompiler.cmake)

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

# Add these standard paths to the search paths for FIND_PATH
# to find include files from these locations first
if(MINGW)
  if(EXISTS /mingw)
      list(APPEND CMAKE_INCLUDE_PATH /mingw)
  endif()
  if(EXISTS /mingw32)
      list(APPEND CMAKE_INCLUDE_PATH /mingw32)
  endif()
  if(EXISTS /mingw64)
      list(APPEND CMAKE_INCLUDE_PATH /mingw64)
  endif()
endif()

# ----------------------------------------------------------------------------
# OpenCV cmake options
# ----------------------------------------------------------------------------

OCV_OPTION(OPENCV_ENABLE_NONFREE "Enable non-free algorithms" OFF)

# Optional 3rd party components
# ===================================================
OCV_OPTION(WITH_1394           "Include IEEE1394 support"                    ON   IF (NOT ANDROID AND NOT IOS AND NOT WINRT) )
OCV_OPTION(WITH_AVFOUNDATION   "Use AVFoundation for Video I/O (iOS/Mac)"    ON   IF APPLE)
OCV_OPTION(WITH_CARBON         "Use Carbon for UI instead of Cocoa"          OFF  IF APPLE )
OCV_OPTION(WITH_CAROTENE       "Use NVidia carotene acceleration library for ARM platform"                   ON  IF (ARM OR AARCH64) AND NOT IOS AND NOT (CMAKE_VERSION VERSION_LESS "2.8.11"))
OCV_OPTION(WITH_VTK            "Include VTK library support (and build opencv_viz module eiher)"             ON  IF (NOT ANDROID AND NOT IOS AND NOT WINRT AND NOT CMAKE_CROSSCOMPILING) )
OCV_OPTION(WITH_CUDA           "Include NVidia Cuda Runtime support"                                         ON  IF (NOT IOS AND NOT WINRT) )
OCV_OPTION(WITH_CUFFT          "Include NVidia Cuda Fast Fourier Transform (FFT) library support"            ON  IF (NOT IOS AND NOT WINRT) )
OCV_OPTION(WITH_CUBLAS         "Include NVidia Cuda Basic Linear Algebra Subprograms (BLAS) library support" OFF IF (NOT IOS AND NOT WINRT) )
OCV_OPTION(WITH_NVCUVID        "Include NVidia Video Decoding library support"                               OFF IF (NOT IOS AND NOT APPLE) )
OCV_OPTION(WITH_EIGEN          "Include Eigen2/Eigen3 support"               ON   IF (NOT WINRT) )
OCV_OPTION(WITH_VFW            "Include Video for Windows support"           ON   IF WIN32 )
OCV_OPTION(WITH_FFMPEG         "Include FFMPEG support"                      ON   IF (NOT ANDROID AND NOT IOS AND NOT WINRT) )
OCV_OPTION(WITH_GSTREAMER      "Include Gstreamer support"                   ON   IF (NOT ANDROID) )
OCV_OPTION(WITH_GSTREAMER_0_10 "Enable Gstreamer 0.10 support (instead of 1.x)"                              OFF )
OCV_OPTION(WITH_GTK            "Include GTK support"                         ON   IF (UNIX AND NOT APPLE AND NOT ANDROID) )
OCV_OPTION(WITH_GTK_2_X        "Use GTK version 2"                           OFF  IF (UNIX AND NOT APPLE AND NOT ANDROID) )
OCV_OPTION(WITH_IPP            "Include Intel IPP support"                   NOT MINGW IF (X86_64 OR X86) AND NOT WINRT )
OCV_OPTION(WITH_JASPER         "Include JPEG2K support"                      ON   IF (NOT IOS) )
OCV_OPTION(WITH_JPEG           "Include JPEG support"                        ON)
OCV_OPTION(WITH_WEBP           "Include WebP support"                        ON   IF (NOT IOS AND NOT WINRT) )
OCV_OPTION(WITH_OPENEXR        "Include ILM support via OpenEXR"             ON   IF (NOT IOS AND NOT WINRT) )
OCV_OPTION(WITH_OPENGL         "Include OpenGL support"                      OFF  IF (NOT ANDROID AND NOT WINRT) )
OCV_OPTION(WITH_OPENVX         "Include OpenVX support"                      OFF)
OCV_OPTION(WITH_OPENNI         "Include OpenNI support"                      OFF  IF (NOT ANDROID AND NOT IOS AND NOT WINRT) )
OCV_OPTION(WITH_OPENNI2        "Include OpenNI2 support"                     OFF  IF (NOT ANDROID AND NOT IOS AND NOT WINRT) )
OCV_OPTION(WITH_PNG            "Include PNG support"                         ON)
OCV_OPTION(WITH_GDCM           "Include DICOM support"                       OFF)
OCV_OPTION(WITH_PVAPI          "Include Prosilica GigE support"              OFF   IF (NOT ANDROID AND NOT IOS AND NOT WINRT) )
OCV_OPTION(WITH_GIGEAPI        "Include Smartek GigE support"                OFF   IF (NOT ANDROID AND NOT IOS AND NOT WINRT) )
OCV_OPTION(WITH_ARAVIS         "Include Aravis GigE support"                 OFF   IF (NOT ANDROID AND NOT IOS AND NOT WINRT AND NOT WIN32) )
OCV_OPTION(WITH_QT             "Build with Qt Backend support"               OFF  IF (NOT ANDROID AND NOT IOS AND NOT WINRT) )
OCV_OPTION(WITH_WIN32UI        "Build with Win32 UI Backend support"         ON   IF WIN32 AND NOT WINRT)
OCV_OPTION(WITH_QUICKTIME      "Use QuickTime for Video I/O"                 OFF  IF APPLE )
OCV_OPTION(WITH_QTKIT          "Use QTKit Video I/O backend"                 OFF  IF APPLE )
OCV_OPTION(WITH_TBB            "Include Intel TBB support"                   OFF  IF (NOT IOS AND NOT WINRT) )
OCV_OPTION(WITH_OPENMP         "Include OpenMP support"                      OFF)
OCV_OPTION(WITH_CSTRIPES       "Include C= support"                          OFF  IF (WIN32 AND NOT WINRT)  )
OCV_OPTION(WITH_PTHREADS_PF    "Use pthreads-based parallel_for"             ON   IF (NOT WIN32 OR MINGW) )
OCV_OPTION(WITH_TIFF           "Include TIFF support"                        ON   IF (NOT IOS) )
OCV_OPTION(WITH_UNICAP         "Include Unicap support (GPL)"                OFF  IF (UNIX AND NOT APPLE AND NOT ANDROID) )
OCV_OPTION(WITH_V4L            "Include Video 4 Linux support"               ON   IF (UNIX AND NOT ANDROID) )
OCV_OPTION(WITH_LIBV4L         "Use libv4l for Video 4 Linux support"        OFF  IF (UNIX AND NOT ANDROID) )
OCV_OPTION(WITH_DSHOW          "Build VideoIO with DirectShow support"       ON   IF (WIN32 AND NOT ARM AND NOT WINRT) )
OCV_OPTION(WITH_MSMF           "Build VideoIO with Media Foundation support" OFF  IF WIN32 )
OCV_OPTION(WITH_XIMEA          "Include XIMEA cameras support"               OFF  IF (NOT ANDROID AND NOT WINRT) )
OCV_OPTION(WITH_XINE           "Include Xine support (GPL)"                  OFF  IF (UNIX AND NOT APPLE AND NOT ANDROID) )
OCV_OPTION(WITH_CLP            "Include Clp support (EPL)"                   OFF)
OCV_OPTION(WITH_OPENCL         "Include OpenCL Runtime support"              NOT ANDROID IF (NOT IOS AND NOT WINRT) )
OCV_OPTION(WITH_OPENCL_SVM     "Include OpenCL Shared Virtual Memory support" OFF ) # experimental
OCV_OPTION(WITH_OPENCLAMDFFT   "Include AMD OpenCL FFT library support"      ON   IF (NOT ANDROID AND NOT IOS AND NOT WINRT) )
OCV_OPTION(WITH_OPENCLAMDBLAS  "Include AMD OpenCL BLAS library support"     ON   IF (NOT ANDROID AND NOT IOS AND NOT WINRT) )
OCV_OPTION(WITH_DIRECTX        "Include DirectX support"                     ON   IF (WIN32 AND NOT WINRT) )
OCV_OPTION(WITH_INTELPERC      "Include Intel Perceptual Computing support"  OFF  IF (WIN32 AND NOT WINRT) )
OCV_OPTION(WITH_IPP_A          "Include Intel IPP_A support"                 OFF  IF (MSVC OR X86 OR X86_64) )
OCV_OPTION(WITH_MATLAB         "Include Matlab support"                      ON   IF (NOT ANDROID AND NOT IOS AND NOT WINRT))
OCV_OPTION(WITH_VA             "Include VA support"                          OFF  IF (UNIX AND NOT ANDROID) )
OCV_OPTION(WITH_VA_INTEL       "Include Intel VA-API/OpenCL support"         OFF  IF (UNIX AND NOT ANDROID) )
OCV_OPTION(WITH_GDAL           "Include GDAL Support"                        OFF  IF (NOT ANDROID AND NOT IOS AND NOT WINRT) )
OCV_OPTION(WITH_GPHOTO2        "Include gPhoto2 library support"             ON   IF (UNIX AND NOT ANDROID) )
OCV_OPTION(WITH_LAPACK         "Include Lapack library support"              ON   IF (NOT ANDROID AND NOT IOS) )

# OpenCV build components
# ===================================================
OCV_OPTION(BUILD_SHARED_LIBS        "Build shared libraries (.dll/.so) instead of static ones (.lib/.a)" NOT (ANDROID OR APPLE_FRAMEWORK) )
OCV_OPTION(BUILD_opencv_apps        "Build utility applications (used for example to train classifiers)" (NOT ANDROID AND NOT WINRT) IF (NOT APPLE_FRAMEWORK) )
OCV_OPTION(BUILD_ANDROID_EXAMPLES   "Build examples for Android platform"         ON  IF ANDROID )
OCV_OPTION(BUILD_DOCS               "Create build rules for OpenCV Documentation" ON  IF (NOT WINRT OR APPLE_FRAMEWORK))
OCV_OPTION(BUILD_EXAMPLES           "Build all examples"                          OFF )
OCV_OPTION(BUILD_PACKAGE            "Enables 'make package_source' command"       ON  IF NOT WINRT)
OCV_OPTION(BUILD_PERF_TESTS         "Build performance tests"                     ON  IF (NOT APPLE_FRAMEWORK) )
OCV_OPTION(BUILD_TESTS              "Build accuracy & regression tests"           ON  IF (NOT APPLE_FRAMEWORK) )
OCV_OPTION(BUILD_WITH_DEBUG_INFO    "Include debug info into debug libs (not MSCV only)" ON )
OCV_OPTION(BUILD_WITH_STATIC_CRT    "Enables use of staticaly linked CRT for staticaly linked OpenCV" ON IF MSVC )
OCV_OPTION(BUILD_WITH_DYNAMIC_IPP   "Enables dynamic linking of IPP (only for standalone IPP)" OFF )
OCV_OPTION(BUILD_FAT_JAVA_LIB       "Create fat java wrapper containing the whole OpenCV library" ON IF NOT BUILD_SHARED_LIBS AND CMAKE_COMPILER_IS_GNUCXX )
OCV_OPTION(BUILD_ANDROID_SERVICE    "Build OpenCV Manager for Google Play" OFF IF ANDROID )
OCV_OPTION(BUILD_CUDA_STUBS         "Build CUDA modules stubs when no CUDA SDK" OFF  IF (NOT APPLE_FRAMEWORK) )

# 3rd party libs
OCV_OPTION(BUILD_ZLIB               "Build zlib from source"             WIN32 OR APPLE )
OCV_OPTION(BUILD_TIFF               "Build libtiff from source"          WIN32 OR ANDROID OR APPLE )
OCV_OPTION(BUILD_JASPER             "Build libjasper from source"        WIN32 OR ANDROID OR APPLE )
OCV_OPTION(BUILD_JPEG               "Build libjpeg from source"          WIN32 OR ANDROID OR APPLE )
OCV_OPTION(BUILD_PNG                "Build libpng from source"           WIN32 OR ANDROID OR APPLE )
OCV_OPTION(BUILD_OPENEXR            "Build openexr from source"          (WIN32 OR ANDROID OR APPLE) AND NOT WINRT)
OCV_OPTION(BUILD_TBB                "Download and build TBB from source" ANDROID )

# OpenCV installation options
# ===================================================
OCV_OPTION(INSTALL_CREATE_DISTRIB   "Change install rules to build the distribution package" OFF )
OCV_OPTION(INSTALL_C_EXAMPLES       "Install C examples"        OFF )
OCV_OPTION(INSTALL_PYTHON_EXAMPLES  "Install Python examples"   OFF )
OCV_OPTION(INSTALL_ANDROID_EXAMPLES "Install Android examples"  OFF IF ANDROID )
OCV_OPTION(INSTALL_TO_MANGLED_PATHS "Enables mangled install paths, that help with side by side installs." OFF IF (UNIX AND NOT ANDROID AND NOT APPLE_FRAMEWORK AND BUILD_SHARED_LIBS) )
OCV_OPTION(INSTALL_TESTS            "Install accuracy and performance test binaries and test data" OFF)

# OpenCV build options
# ===================================================
OCV_OPTION(ENABLE_CCACHE              "Use ccache"                                               (UNIX AND NOT IOS AND (CMAKE_GENERATOR MATCHES "Makefile" OR CMAKE_GENERATOR MATCHES "Ninja")) )
OCV_OPTION(ENABLE_PRECOMPILED_HEADERS "Use precompiled headers"                                  ON IF (NOT IOS AND NOT CMAKE_CROSSCOMPILING) )
OCV_OPTION(ENABLE_SOLUTION_FOLDERS    "Solution folder in Visual Studio or in other IDEs"        (MSVC_IDE OR CMAKE_GENERATOR MATCHES Xcode) )
OCV_OPTION(ENABLE_PROFILING           "Enable profiling in the GCC compiler (Add flags: -g -pg)" OFF  IF CMAKE_COMPILER_IS_GNUCXX )
OCV_OPTION(ENABLE_COVERAGE            "Enable coverage collection with  GCov"                    OFF  IF CMAKE_COMPILER_IS_GNUCXX )
OCV_OPTION(ENABLE_OMIT_FRAME_POINTER  "Enable -fomit-frame-pointer for GCC"                      ON   IF CMAKE_COMPILER_IS_GNUCXX AND NOT (APPLE AND CMAKE_COMPILER_IS_CLANGCXX) )
OCV_OPTION(ENABLE_POWERPC             "Enable PowerPC for GCC"                                   ON   IF (CMAKE_COMPILER_IS_GNUCXX AND CMAKE_SYSTEM_PROCESSOR MATCHES powerpc.*) )
OCV_OPTION(ENABLE_FAST_MATH           "Enable -ffast-math (not recommended for GCC 4.6.x)"       OFF  IF (CMAKE_COMPILER_IS_GNUCXX AND (X86 OR X86_64)) )
OCV_OPTION(ENABLE_SSE                 "Enable SSE instructions"                                  ON   IF ((MSVC OR CMAKE_COMPILER_IS_GNUCXX) AND (X86 OR X86_64)) )
OCV_OPTION(ENABLE_SSE2                "Enable SSE2 instructions"                                 ON   IF ((MSVC OR CMAKE_COMPILER_IS_GNUCXX) AND (X86 OR X86_64)) )
OCV_OPTION(ENABLE_SSE3                "Enable SSE3 instructions"                                 ON   IF ((MSVC OR CMAKE_COMPILER_IS_GNUCXX OR CV_ICC) AND (X86 OR X86_64)) )
OCV_OPTION(ENABLE_SSSE3               "Enable SSSE3 instructions"                                OFF  IF ((MSVC OR CMAKE_COMPILER_IS_GNUCXX) AND (X86 OR X86_64)) )
OCV_OPTION(ENABLE_SSE41               "Enable SSE4.1 instructions"                               OFF  IF ((MSVC OR CMAKE_COMPILER_IS_GNUCXX OR CV_ICC) AND (X86 OR X86_64)) )
OCV_OPTION(ENABLE_SSE42               "Enable SSE4.2 instructions"                               OFF  IF ((MSVC OR CMAKE_COMPILER_IS_GNUCXX) AND (X86 OR X86_64)) )
OCV_OPTION(ENABLE_POPCNT              "Enable POPCNT instructions"                               OFF  IF ((MSVC OR CMAKE_COMPILER_IS_GNUCXX) AND (X86 OR X86_64)) )
OCV_OPTION(ENABLE_AVX                 "Enable AVX instructions"                                  OFF  IF ((MSVC OR CMAKE_COMPILER_IS_GNUCXX) AND (X86 OR X86_64)) )
OCV_OPTION(ENABLE_AVX2                "Enable AVX2 instructions"                                 OFF  IF ((MSVC OR CMAKE_COMPILER_IS_GNUCXX) AND (X86 OR X86_64)) )
OCV_OPTION(ENABLE_FMA3                "Enable FMA3 instructions"                                 OFF  IF ((MSVC OR CMAKE_COMPILER_IS_GNUCXX) AND (X86 OR X86_64)) )
OCV_OPTION(ENABLE_NEON                "Enable NEON instructions"                                 "${NEON}" IF CMAKE_COMPILER_IS_GNUCXX AND (ARM OR AARCH64 OR IOS) )
OCV_OPTION(ENABLE_VFPV3               "Enable VFPv3-D32 instructions"                            OFF  IF CMAKE_COMPILER_IS_GNUCXX AND (ARM OR AARCH64 OR IOS) )
OCV_OPTION(ENABLE_NOISY_WARNINGS      "Show all warnings even if they are too noisy"             OFF )
OCV_OPTION(OPENCV_WARNINGS_ARE_ERRORS "Treat warnings as errors"                                 OFF )
OCV_OPTION(ANDROID_EXAMPLES_WITH_LIBS "Build binaries of Android examples with native libraries" OFF  IF ANDROID )
OCV_OPTION(ENABLE_IMPL_COLLECTION     "Collect implementation data on function call"             OFF )
OCV_OPTION(ENABLE_INSTRUMENTATION     "Instrument functions to collect calls trace and performance" OFF )
OCV_OPTION(GENERATE_ABI_DESCRIPTOR    "Generate XML file for abi_compliance_checker tool" OFF IF UNIX)

OCV_OPTION(DOWNLOAD_EXTERNAL_TEST_DATA "Download external test data (Python executable and OPENCV_TEST_DATA_PATH environment variable may be required)" OFF )


if(ENABLE_IMPL_COLLECTION)
  add_definitions(-DCV_COLLECT_IMPL_DATA)
endif()


# ----------------------------------------------------------------------------
#  Get actual OpenCV version number from sources
# ----------------------------------------------------------------------------
include(cmake/OpenCVVersion.cmake)


# ----------------------------------------------------------------------------
#  Build & install layouts
# ----------------------------------------------------------------------------

# Save libs and executables in the same place
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin" CACHE PATH "Output directory for applications")

if(ANDROID)
  if(ANDROID_ABI MATCHES "NEON")
    set(ENABLE_NEON ON)
  endif()
  if(ANDROID_ABI MATCHES "VFPV3")
    set(ENABLE_VFPV3 ON)
  endif()
endif()

if(ANDROID OR WIN32)
  ocv_update(OPENCV_DOC_INSTALL_PATH doc)
else()
  ocv_update(OPENCV_DOC_INSTALL_PATH share/OpenCV/doc)
endif()

if(WIN32 AND CMAKE_HOST_SYSTEM_NAME MATCHES Windows)
  if(DEFINED OpenCV_RUNTIME AND DEFINED OpenCV_ARCH)
    ocv_update(OpenCV_INSTALL_BINARIES_PREFIX "${OpenCV_ARCH}/${OpenCV_RUNTIME}/")
  else()
    message(STATUS "Can't detect runtime and/or arch")
    ocv_update(OpenCV_INSTALL_BINARIES_PREFIX "")
  endif()
elseif(ANDROID)
  ocv_update(OpenCV_INSTALL_BINARIES_PREFIX "sdk/native/")
else()
  ocv_update(OpenCV_INSTALL_BINARIES_PREFIX "")
endif()

if(ANDROID)
  ocv_update(OPENCV_SAMPLES_BIN_INSTALL_PATH "${OpenCV_INSTALL_BINARIES_PREFIX}samples/${ANDROID_NDK_ABI_NAME}")
else()
  ocv_update(OPENCV_SAMPLES_BIN_INSTALL_PATH "${OpenCV_INSTALL_BINARIES_PREFIX}samples")
endif()

if(ANDROID)
  ocv_update(OPENCV_BIN_INSTALL_PATH "${OpenCV_INSTALL_BINARIES_PREFIX}bin/${ANDROID_NDK_ABI_NAME}")
else()
  ocv_update(OPENCV_BIN_INSTALL_PATH "${OpenCV_INSTALL_BINARIES_PREFIX}bin")
endif()

if(NOT OPENCV_TEST_INSTALL_PATH)
  ocv_update(OPENCV_TEST_INSTALL_PATH "${OPENCV_BIN_INSTALL_PATH}")
endif()

if (OPENCV_TEST_DATA_PATH)
  get_filename_component(OPENCV_TEST_DATA_PATH ${OPENCV_TEST_DATA_PATH} ABSOLUTE)
endif()

if(ANDROID)
  ocv_update(OPENCV_TEST_DATA_INSTALL_PATH "sdk/etc/testdata")
elseif(WIN32)
  ocv_update(OPENCV_TEST_DATA_INSTALL_PATH "testdata")
else()
  ocv_update(OPENCV_TEST_DATA_INSTALL_PATH "share/OpenCV/testdata")
endif()

if(ANDROID)
  set(LIBRARY_OUTPUT_PATH                "${OpenCV_BINARY_DIR}/lib/${ANDROID_NDK_ABI_NAME}")
  ocv_update(3P_LIBRARY_OUTPUT_PATH      "${OpenCV_BINARY_DIR}/3rdparty/lib/${ANDROID_NDK_ABI_NAME}")
  ocv_update(OPENCV_LIB_INSTALL_PATH     sdk/native/libs/${ANDROID_NDK_ABI_NAME})
  ocv_update(OPENCV_3P_LIB_INSTALL_PATH  sdk/native/3rdparty/libs/${ANDROID_NDK_ABI_NAME})
  ocv_update(OPENCV_CONFIG_INSTALL_PATH  sdk/native/jni)
  ocv_update(OPENCV_INCLUDE_INSTALL_PATH sdk/native/jni/include)
  ocv_update(OPENCV_SAMPLES_SRC_INSTALL_PATH samples/native)
  ocv_update(OPENCV_OTHER_INSTALL_PATH   sdk/etc)
else()
  set(LIBRARY_OUTPUT_PATH                "${OpenCV_BINARY_DIR}/lib")
  ocv_update(3P_LIBRARY_OUTPUT_PATH      "${OpenCV_BINARY_DIR}/3rdparty/lib${LIB_SUFFIX}")

  if(WIN32 AND CMAKE_HOST_SYSTEM_NAME MATCHES Windows)
    if(OpenCV_STATIC)
      ocv_update(OPENCV_LIB_INSTALL_PATH   "${OpenCV_INSTALL_BINARIES_PREFIX}staticlib${LIB_SUFFIX}")
    else()
      ocv_update(OPENCV_LIB_INSTALL_PATH   "${OpenCV_INSTALL_BINARIES_PREFIX}lib${LIB_SUFFIX}")
    endif()
    ocv_update(OPENCV_3P_LIB_INSTALL_PATH  "${OpenCV_INSTALL_BINARIES_PREFIX}staticlib${LIB_SUFFIX}")
    ocv_update(OPENCV_SAMPLES_SRC_INSTALL_PATH    samples/native)
    ocv_update(OPENCV_JAR_INSTALL_PATH java)
    ocv_update(OPENCV_OTHER_INSTALL_PATH   etc)
    ocv_update(OPENCV_CONFIG_INSTALL_PATH  ".")
  else()
    ocv_update(OPENCV_LIB_INSTALL_PATH     lib${LIB_SUFFIX})
    ocv_update(OPENCV_3P_LIB_INSTALL_PATH  share/OpenCV/3rdparty/${OPENCV_LIB_INSTALL_PATH})
    ocv_update(OPENCV_SAMPLES_SRC_INSTALL_PATH    share/OpenCV/samples)
    ocv_update(OPENCV_JAR_INSTALL_PATH share/OpenCV/java)
    ocv_update(OPENCV_OTHER_INSTALL_PATH   share/OpenCV)

    if(NOT DEFINED OPENCV_CONFIG_INSTALL_PATH)
      math(EXPR SIZEOF_VOID_P_BITS "8 * ${CMAKE_SIZEOF_VOID_P}")
      if(LIB_SUFFIX AND NOT SIZEOF_VOID_P_BITS EQUAL LIB_SUFFIX)
        ocv_update(OPENCV_CONFIG_INSTALL_PATH lib${LIB_SUFFIX}/cmake/opencv)
      else()
        ocv_update(OPENCV_CONFIG_INSTALL_PATH share/OpenCV)
      endif()
    endif()
  endif()
  ocv_update(OPENCV_INCLUDE_INSTALL_PATH "include")
endif()

ocv_update(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${OPENCV_LIB_INSTALL_PATH}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

if(INSTALL_TO_MANGLED_PATHS)
  set(OPENCV_INCLUDE_INSTALL_PATH ${OPENCV_INCLUDE_INSTALL_PATH}/opencv-${OPENCV_VERSION})
  foreach(v
      OPENCV_3P_LIB_INSTALL_PATH
      OPENCV_SAMPLES_SRC_INSTALL_PATH
      OPENCV_CONFIG_INSTALL_PATH
      OPENCV_DOC_INSTALL_PATH
      OPENCV_JAR_INSTALL_PATH
      OPENCV_TEST_DATA_INSTALL_PATH
      OPENCV_OTHER_INSTALL_PATH
    )
    string(REPLACE "OpenCV" "OpenCV-${OPENCV_VERSION}" ${v} "${${v}}")
    string(REPLACE "opencv" "opencv-${OPENCV_VERSION}" ${v} "${${v}}")
  endforeach()
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

if((INSTALL_CREATE_DISTRIB AND BUILD_SHARED_LIBS AND NOT DEFINED BUILD_opencv_world) OR APPLE_FRAMEWORK)
  set(BUILD_opencv_world ON CACHE INTERNAL "")
endif()

# ----------------------------------------------------------------------------
#  Path for build/platform -specific headers
# ----------------------------------------------------------------------------
ocv_update(OPENCV_CONFIG_FILE_INCLUDE_DIR "${CMAKE_BINARY_DIR}/" CACHE PATH "Where to create the platform-dependant cvconfig.h")
ocv_include_directories(${OPENCV_CONFIG_FILE_INCLUDE_DIR})

# ----------------------------------------------------------------------------
#  Path for additional modules
# ----------------------------------------------------------------------------
set(OPENCV_EXTRA_MODULES_PATH "" CACHE PATH "Where to look for additional OpenCV modules")

# ----------------------------------------------------------------------------
#  Autodetect if we are in a GIT repository
# ----------------------------------------------------------------------------
find_host_package(Git QUIET)

if(NOT DEFINED OPENCV_VCSVERSION AND GIT_FOUND)
  execute_process(COMMAND "${GIT_EXECUTABLE}" describe --tags --always --dirty --match "[0-9].[0-9].[0-9]*"
    WORKING_DIRECTORY "${OpenCV_SOURCE_DIR}"
    OUTPUT_VARIABLE OPENCV_VCSVERSION
    RESULT_VARIABLE GIT_RESULT
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(NOT GIT_RESULT EQUAL 0)
    set(OPENCV_VCSVERSION "unknown")
  endif()
elseif(NOT DEFINED OPENCV_VCSVERSION)
  # We don't have git:
  set(OPENCV_VCSVERSION "unknown")
endif()


# ----------------------------------------------------------------------------
# OpenCV compiler and linker options
# ----------------------------------------------------------------------------
# In case of Makefiles if the user does not setup CMAKE_BUILD_TYPE, assume it's Release:
if(CMAKE_GENERATOR MATCHES "Makefiles|Ninja" AND "${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE Release)
endif()

include(cmake/OpenCVCompilerOptions.cmake)


# ----------------------------------------------------------------------------
# Use statically or dynamically linked CRT?
# Default: dynamic
# ----------------------------------------------------------------------------
if(MSVC)
  include(cmake/OpenCVCRTLinkage.cmake)
endif(MSVC)

if(WIN32 AND NOT MINGW)
  add_definitions(-D_VARIADIC_MAX=10)
endif(WIN32 AND NOT MINGW)


# ----------------------------------------------------------------------------
#       CHECK FOR SYSTEM LIBRARIES, OPTIONS, ETC..
# ----------------------------------------------------------------------------
if(UNIX)
  find_package(PkgConfig QUIET)
  include(CheckFunctionExists)
  include(CheckIncludeFile)

  if(NOT APPLE)
    CHECK_INCLUDE_FILE(pthread.h HAVE_LIBPTHREAD)
    if(ANDROID)
      set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} dl m log)
    elseif(${CMAKE_SYSTEM_NAME} MATCHES "FreeBSD|NetBSD|DragonFly|OpenBSD")
      set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} m pthread)
    elseif(EMSCRIPTEN)
      # no need to link to system libs with emscripten
    else()
      set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} dl m pthread rt)
    endif()
  else()
    set(HAVE_LIBPTHREAD YES)
  endif()
endif()

include(cmake/OpenCVPCHSupport.cmake)
include(cmake/OpenCVModule.cmake)

# ----------------------------------------------------------------------------
#  Detect endianness of build platform
# ----------------------------------------------------------------------------

if(IOS)
  # test_big_endian needs try_compile, which doesn't work for iOS
  # http://public.kitware.com/Bug/view.php?id=12288
  set(WORDS_BIGENDIAN 0)
else()
  include(TestBigEndian)
  test_big_endian(WORDS_BIGENDIAN)
endif()

# ----------------------------------------------------------------------------
#  Detect 3rd-party libraries
# ----------------------------------------------------------------------------

include(cmake/OpenCVFindLibsGrfmt.cmake)
include(cmake/OpenCVFindLibsGUI.cmake)
include(cmake/OpenCVFindLibsVideo.cmake)
include(cmake/OpenCVFindLibsPerf.cmake)
include(cmake/OpenCVFindLAPACK.cmake)

# ----------------------------------------------------------------------------
#  Detect other 3rd-party libraries/tools
# ----------------------------------------------------------------------------

# --- Doxygen for documentation ---
if(BUILD_DOCS)
  find_package(Doxygen)
else()
  unset(DOXYGEN_FOUND CACHE)
endif()

# --- Python Support ---
include(cmake/OpenCVDetectPython.cmake)

# --- Java Support ---
include(cmake/OpenCVDetectApacheAnt.cmake)
if(ANDROID)
  include(cmake/OpenCVDetectAndroidSDK.cmake)

  if(NOT ANDROID_TOOLS_Pkg_Revision GREATER 13)
    message(WARNING "OpenCV requires Android SDK tools revision 14 or newer. Otherwise tests and samples will no be compiled.")
  endif()
else()
  find_package(JNI)
endif()

if(ANDROID AND ANDROID_EXECUTABLE AND ANT_EXECUTABLE AND (ANT_VERSION VERSION_GREATER 1.7) AND (ANDROID_TOOLS_Pkg_Revision GREATER 13))
  SET(CAN_BUILD_ANDROID_PROJECTS TRUE)
else()
  SET(CAN_BUILD_ANDROID_PROJECTS FALSE)
endif()

# --- OpenCL ---
if(WITH_OPENCL)
  include(cmake/OpenCVDetectOpenCL.cmake)
endif()

# --- DirectX ---
if(WITH_DIRECTX)
  include(cmake/OpenCVDetectDirectX.cmake)
endif()

# --- Matlab/Octave ---
if(WITH_MATLAB)
  include(cmake/OpenCVFindMatlab.cmake)
endif()

include(cmake/OpenCVDetectVTK.cmake)

if(WITH_OPENVX)
  include(cmake/FindOpenVX.cmake)
endif()

# ----------------------------------------------------------------------------
# OpenCV HAL
# ----------------------------------------------------------------------------
set(_hal_includes "")
macro(ocv_hal_register HAL_LIBRARIES_VAR HAL_HEADERS_VAR HAL_INCLUDE_DIRS_VAR)
  # 1. libraries
  foreach (l ${${HAL_LIBRARIES_VAR}})
    if(NOT TARGET ${l})
      get_filename_component(l "${l}" ABSOLUTE)
    endif()
    list(APPEND OPENCV_HAL_LINKER_LIBS ${l})
  endforeach()
  # 2. headers
  foreach (h ${${HAL_HEADERS_VAR}})
    set(_hal_includes "${_hal_includes}\n#include \"${h}\"")
  endforeach()
  # 3. include paths
  ocv_include_directories(${${HAL_INCLUDE_DIRS_VAR}})
endmacro()

if(NOT DEFINED OpenCV_HAL)
  set(OpenCV_HAL "OpenCV_HAL")
endif()

if(HAVE_OPENVX)
  if(NOT ";${OpenCV_HAL};" MATCHES ";openvx;")
    set(OpenCV_HAL "openvx;${OpenCV_HAL}")
  endif()
endif()

if(WITH_CAROTENE)
  ocv_debug_message(STATUS "Enable carotene acceleration")
  if(NOT ";${OpenCV_HAL};" MATCHES ";carotene;")
    set(OpenCV_HAL "carotene;${OpenCV_HAL}")
  endif()
endif()

foreach(hal ${OpenCV_HAL})
  if(hal STREQUAL "carotene")
    add_subdirectory(3rdparty/carotene/hal)
    ocv_hal_register(CAROTENE_HAL_LIBRARIES CAROTENE_HAL_HEADERS CAROTENE_HAL_INCLUDE_DIRS)
    list(APPEND OpenCV_USED_HAL "carotene (ver ${CAROTENE_HAL_VERSION})")
  elseif(hal STREQUAL "openvx")
    add_subdirectory(3rdparty/openvx)
    ocv_hal_register(OPENVX_HAL_LIBRARIES OPENVX_HAL_HEADERS OPENVX_HAL_INCLUDE_DIRS)
    list(APPEND OpenCV_USED_HAL "openvx (ver ${OPENVX_HAL_VERSION})")
  else()
    ocv_debug_message(STATUS "OpenCV HAL: ${hal} ...")
    ocv_clear_vars(OpenCV_HAL_LIBRARIES OpenCV_HAL_HEADERS OpenCV_HAL_INCLUDE_DIRS)
    find_package(${hal} NO_MODULE QUIET)
    if(${hal}_FOUND)
      ocv_hal_register(OpenCV_HAL_LIBRARIES OpenCV_HAL_HEADERS OpenCV_HAL_INCLUDE_DIRS)
      list(APPEND OpenCV_USED_HAL "${hal} (ver ${${hal}_VERSION})")
    endif()
  endif()
endforeach()
configure_file("${OpenCV_SOURCE_DIR}/cmake/templates/custom_hal.hpp.in" "${CMAKE_BINARY_DIR}/custom_hal.hpp" @ONLY)
unset(_hal_includes)


# ----------------------------------------------------------------------------
# Add CUDA libraries (needed for apps/tools, samples)
# ----------------------------------------------------------------------------
if(HAVE_CUDA)
  set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} ${CUDA_LIBRARIES} ${CUDA_npp_LIBRARY})
  if(HAVE_CUBLAS)
    set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} ${CUDA_cublas_LIBRARY})
  endif()
  if(HAVE_CUFFT)
    set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} ${CUDA_cufft_LIBRARY})
  endif()
  foreach(p ${CUDA_LIBS_PATH})
    set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} ${CMAKE_LIBRARY_PATH_FLAG}${p})
  endforeach()
endif()
# ----------------------------------------------------------------------------
# Solution folders:
# ----------------------------------------------------------------------------
if(ENABLE_SOLUTION_FOLDERS)
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
  set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "CMakeTargets")
endif()

# Extra OpenCV targets: uninstall, package_source, perf, etc.
include(cmake/OpenCVExtraTargets.cmake)


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
if(BUILD_opencv_apps)
  add_subdirectory(apps)
endif()

# examples
if(BUILD_EXAMPLES OR BUILD_ANDROID_EXAMPLES OR INSTALL_PYTHON_EXAMPLES)
  add_subdirectory(samples)
endif()

if(ANDROID)
  add_subdirectory(platforms/android/service)
endif()

# ----------------------------------------------------------------------------
# Finalization: generate configuration-based files
# ----------------------------------------------------------------------------

# Generate platform-dependent and configuration-dependent headers
include(cmake/OpenCVGenHeaders.cmake)

# Generate opencv.pc for pkg-config command
include(cmake/OpenCVGenPkgconfig.cmake)

# Generate OpenCV.mk for ndk-build (Android build tool)
include(cmake/OpenCVGenAndroidMK.cmake)

# Generate OpenCVConfig.cmake and OpenCVConfig-version.cmake for cmake projects
include(cmake/OpenCVGenConfig.cmake)

# Generate Info.plist for the IOS framework
if(APPLE_FRAMEWORK)
  include(cmake/OpenCVGenInfoPlist.cmake)
endif()

# Generate ABI descriptor
include(cmake/OpenCVGenABI.cmake)

# Generate environment setup file
if(INSTALL_TESTS AND OPENCV_TEST_DATA_PATH)
  if(ANDROID)
    get_filename_component(TEST_PATH ${OPENCV_TEST_INSTALL_PATH} DIRECTORY)
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/templates/opencv_run_all_tests_android.sh.in"
                   "${CMAKE_BINARY_DIR}/unix-install/opencv_run_all_tests.sh" @ONLY)
    install(PROGRAMS "${CMAKE_BINARY_DIR}/unix-install/opencv_run_all_tests.sh"
            DESTINATION ./ COMPONENT tests)
  elseif(WIN32)
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/templates/opencv_run_all_tests_windows.cmd.in"
                   "${CMAKE_BINARY_DIR}/win-install/opencv_run_all_tests.cmd" @ONLY)
    install(PROGRAMS "${CMAKE_BINARY_DIR}/win-install/opencv_run_all_tests.cmd"
            DESTINATION ${OPENCV_TEST_INSTALL_PATH} COMPONENT tests)
  elseif(UNIX)
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/templates/opencv_run_all_tests_unix.sh.in"
                   "${CMAKE_BINARY_DIR}/unix-install/opencv_run_all_tests.sh" @ONLY)
    install(PROGRAMS "${CMAKE_BINARY_DIR}/unix-install/opencv_run_all_tests.sh"
            DESTINATION ${OPENCV_TEST_INSTALL_PATH} COMPONENT tests)
  endif()
endif()

if(NOT OPENCV_README_FILE)
  if(ANDROID)
    set(OPENCV_README_FILE ${CMAKE_CURRENT_SOURCE_DIR}/platforms/android/README.android)
  endif()
endif()

if(NOT OPENCV_LICENSE_FILE)
  set(OPENCV_LICENSE_FILE ${CMAKE_CURRENT_SOURCE_DIR}/LICENSE)
endif()

# for UNIX it does not make sense as LICENSE and readme will be part of the package automatically
if(ANDROID OR NOT UNIX)
  install(FILES ${OPENCV_LICENSE_FILE}
        PERMISSIONS OWNER_READ GROUP_READ WORLD_READ
        DESTINATION ./ COMPONENT libs)
  if(OPENCV_README_FILE)
    install(FILES ${OPENCV_README_FILE}
            PERMISSIONS OWNER_READ GROUP_READ WORLD_READ
            DESTINATION ./ COMPONENT libs)
  endif()
endif()

# ----------------------------------------------------------------------------
# Summary:
# ----------------------------------------------------------------------------
status("")
status("General configuration for OpenCV ${OPENCV_VERSION} =====================================")
if(OPENCV_VCSVERSION)
  status("  Version control:" ${OPENCV_VCSVERSION})
endif()
if(OPENCV_EXTRA_MODULES_PATH AND NOT BUILD_INFO_SKIP_EXTRA_MODULES)
  set(__dump_extra_header OFF)
  foreach(p ${OPENCV_EXTRA_MODULES_PATH})
    if(EXISTS ${p})
      if(NOT __dump_extra_header)
        set(__dump_extra_header ON)
        status("")
        status("  Extra modules:")
      else()
        status("")
      endif()
      set(EXTRA_MODULES_VCSVERSION "unknown")
      if(GIT_FOUND)
        execute_process(COMMAND "${GIT_EXECUTABLE}" describe --tags --always --dirty --match "[0-9].[0-9].[0-9]*"
          WORKING_DIRECTORY "${p}"
          OUTPUT_VARIABLE EXTRA_MODULES_VCSVERSION
          RESULT_VARIABLE GIT_RESULT
          ERROR_QUIET
          OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(NOT GIT_RESULT EQUAL 0)
          set(EXTRA_MODULES_VCSVERSION "unknown")
        endif()
      endif()
      status("    Location (extra):" ${p})
      status("    Version control (extra):" ${EXTRA_MODULES_VCSVERSION})
    endif()
  endforeach()
  unset(__dump_extra_header)
endif()

# ========================== build platform ==========================
status("")
status("  Platform:")
if(NOT CMAKE_VERSION VERSION_LESS 2.8.11 AND NOT BUILD_INFO_SKIP_TIMESTAMP)
  string(TIMESTAMP TIMESTAMP "" UTC)
  if(TIMESTAMP)
    status("    Timestamp:"    ${TIMESTAMP})
  endif()
endif()
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

# ========================== C/C++ options ==========================
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

status("")
status("  C/C++:")
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
status("    ccache:"                  CMAKE_COMPILER_IS_CCACHE THEN YES ELSE NO)
status("    Precompiled headers:"     PCHSupport_FOUND AND ENABLE_PRECOMPILED_HEADERS THEN YES ELSE NO)

# ========================== Dependencies ============================
ocv_get_all_libs(deps_modules deps_extra deps_3rdparty)
status("    Extra dependencies:" ${deps_extra})
status("    3rdparty dependencies:" ${deps_3rdparty})

# ========================== OpenCV modules ==========================
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

if(OPENCV_ENABLE_NONFREE)
  status("")
  status("  Non-free algorithms are enabled")
endif()

# ========================== Android details ==========================
if(ANDROID)
  status("")
  status("  Android: ")
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
  status("    Google Play manager:" BUILD_ANDROID_SERVICE                                         THEN YES ELSE NO)
  status("    Android examples:"    BUILD_ANDROID_EXAMPLES AND CAN_BUILD_ANDROID_PROJECTS         THEN YES ELSE NO)
endif()

# ================== Windows RT features ==================
if(WIN32)
status("")
status("  Windows RT support:" WINRT THEN YES ELSE NO)
  if(WINRT)
    status("    Building for Microsoft platform: " ${CMAKE_SYSTEM_NAME})
    status("    Building for architectures: " ${CMAKE_VS_EFFECTIVE_PLATFORMS})
    status("    Building for version: " ${CMAKE_SYSTEM_VERSION})
    if (DEFINED ENABLE_WINRT_MODE_NATIVE)
      status("    Building for C++ without CX extensions")
    endif()
  endif()
endif(WIN32)

# ========================== GUI ==========================
status("")
status("  GUI: ")

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
      if(HAVE_GTK3)
        status("    GTK+ 3.x:" HAVE_GTK THEN "YES (ver ${ALIASOF_gtk+-3.0_VERSION})" ELSE NO)
      elseif(HAVE_GTK)
        status("    GTK+ 2.x:" HAVE_GTK THEN "YES (ver ${ALIASOF_gtk+-2.0_VERSION})" ELSE NO)
      else()
        status("    GTK+:" NO)
      endif()
      status("    GThread :" HAVE_GTHREAD THEN "YES (ver ${ALIASOF_gthread-2.0_VERSION})" ELSE NO)
      status("    GtkGlExt:" HAVE_GTKGLEXT THEN "YES (ver ${ALIASOF_gtkglext-1.0_VERSION})" ELSE NO)
    endif()
  endif()
endif()

status("    OpenGL support:" HAVE_OPENGL THEN "YES (${OPENGL_LIBRARIES})" ELSE NO)
status("    VTK support:" HAVE_VTK THEN "YES (ver ${VTK_VERSION})" ELSE NO)

# ========================== MEDIA IO ==========================
status("")
status("  Media I/O: ")
status("    ZLib:"         BUILD_ZLIB    THEN "build (ver ${ZLIB_VERSION_STRING})"               ELSE "${ZLIB_LIBRARIES} (ver ${ZLIB_VERSION_STRING})")

if(WITH_JPEG)
  status("    JPEG:"       JPEG_FOUND    THEN "${JPEG_LIBRARY} (ver ${JPEG_LIB_VERSION})"        ELSE "build (ver ${JPEG_LIB_VERSION})")
else()
  status("    JPEG:"       "NO")
endif()

if(WITH_WEBP)
  status("    WEBP:"       WEBP_FOUND    THEN "${WEBP_LIBRARY} (ver ${WEBP_VERSION})"        ELSE "build (ver ${WEBP_VERSION})")
else()
  status("    WEBP:"       "NO")
endif()

if(WITH_PNG)
  status("    PNG:"        PNG_FOUND     THEN "${PNG_LIBRARY} (ver ${PNG_VERSION})"              ELSE "build (ver ${PNG_VERSION})")
else()
  status("    PNG:"        "NO")
endif()

if(WITH_TIFF)
  if(TIFF_VERSION_STRING AND TIFF_FOUND)
    status("    TIFF:"     "${TIFF_LIBRARY} (ver ${TIFF_VERSION} - ${TIFF_VERSION_STRING})")
  else()
    status("    TIFF:"     TIFF_FOUND    THEN "${TIFF_LIBRARY} (ver ${TIFF_VERSION})"            ELSE "build (ver ${TIFF_VERSION} - ${TIFF_VERSION_STRING})")
  endif()
else()
  status("    TIFF:"       "NO")
endif()
if(WITH_JASPER)
  status("    JPEG 2000:"  JASPER_FOUND  THEN "${JASPER_LIBRARY} (ver ${JASPER_VERSION_STRING})" ELSE "build (ver ${JASPER_VERSION_STRING})")
else()
  status("    JPEG 2000:"  "NO")
endif()
if(WITH_OPENEXR)
  status("    OpenEXR:"  OPENEXR_FOUND  THEN "${OPENEXR_LIBRARIES} (ver ${OPENEXR_VERSION})" ELSE "build (ver ${OPENEXR_VERSION})")
else()
  status("    OpenEXR:"  "NO")
endif()

if( WITH_GDAL )
  status("    GDAL:"   GDAL_FOUND THEN "${GDAL_LIBRARY}" ELSE "NO")
else()
  status("    GDAL:"     "NO")
endif()

if(WITH_GDCM)
  status("    GDCM:"   GDCM_FOUND THEN "YES (ver ${GDCM_VERSION})" ELSE "NO")
else()
  status("    GDCM:"     "NO")
endif()

# ========================== VIDEO IO ==========================
status("")
status("  Video I/O:")

if (DEFINED WITH_VFW)
  status("    Video for Windows:" HAVE_VFW         THEN YES                                        ELSE NO)
endif(DEFINED WITH_VFW)

if(DEFINED WITH_1394)
  status("    DC1394 1.x:"     HAVE_DC1394         THEN "YES (ver ${ALIASOF_libdc1394_VERSION})"   ELSE NO)
  status("    DC1394 2.x:"     HAVE_DC1394_2       THEN "YES (ver ${ALIASOF_libdc1394-2_VERSION})" ELSE NO)
endif(DEFINED WITH_1394)

if(DEFINED WITH_FFMPEG OR HAVE_FFMPEG)
  if(WIN32)
    status("    FFMPEG:"       WITH_FFMPEG         THEN "YES (prebuilt binaries)"                  ELSE NO)
  else()
    status("    FFMPEG:"       HAVE_FFMPEG         THEN YES ELSE NO)
  endif()
  status("      avcodec:"      FFMPEG_libavcodec_FOUND    THEN "YES (ver ${FFMPEG_libavcodec_VERSION})"    ELSE NO)
  status("      avformat:"     FFMPEG_libavformat_FOUND   THEN "YES (ver ${FFMPEG_libavformat_VERSION})"   ELSE NO)
  status("      avutil:"       FFMPEG_libavutil_FOUND     THEN "YES (ver ${FFMPEG_libavutil_VERSION})"     ELSE NO)
  status("      swscale:"      FFMPEG_libswscale_FOUND    THEN "YES (ver ${FFMPEG_libswscale_VERSION})"    ELSE NO)
  status("      avresample:"   FFMPEG_libavresample_FOUND THEN "YES (ver ${FFMPEG_libavresample_VERSION})" ELSE NO)
endif()

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

if(DEFINED WITH_OPENNI)
  status("    OpenNI:"         HAVE_OPENNI         THEN "YES (ver ${OPENNI_VERSION_STRING}, build ${OPENNI_VERSION_BUILD})"
                                                                                                   ELSE NO)
  status("    OpenNI PrimeSensor Modules:" HAVE_OPENNI_PRIME_SENSOR_MODULE
                                                   THEN "YES (${OPENNI_PRIME_SENSOR_MODULE})"      ELSE NO)
endif(DEFINED WITH_OPENNI)

if(DEFINED WITH_OPENNI2)
  status("    OpenNI2:"        HAVE_OPENNI2    THEN "YES (ver ${OPENNI2_VERSION_STRING}, build ${OPENNI2_VERSION_BUILD})"
                                                                                                   ELSE NO)
endif(DEFINED WITH_OPENNI2)

if(DEFINED WITH_PVAPI)
  status("    PvAPI:"          HAVE_PVAPI          THEN YES                                        ELSE NO)
endif(DEFINED WITH_PVAPI)

if(DEFINED WITH_GIGEAPI)
  status("    GigEVisionSDK:"  HAVE_GIGE_API       THEN YES                                        ELSE NO)
endif(DEFINED WITH_GIGEAPI)

if(DEFINED WITH_ARAVIS)
  status("    Aravis SDK:"     HAVE_ARAVIS_API     THEN "YES (${ARAVIS_LIBRARIES})"                ELSE NO)
endif(DEFINED WITH_ARAVIS)

if(DEFINED APPLE)
  status("    AVFoundation:"   HAVE_AVFOUNDATION   THEN YES                                        ELSE NO)
  if(WITH_QUICKTIME OR HAVE_QUICKTIME)
  status("    QuickTime:"      HAVE_QUICKTIME      THEN YES                                        ELSE NO)
  endif()
  if(WITH_QTKIT OR HAVE_QTKIT)
  status("    QTKit:"          HAVE_QTKIT          THEN "YES (deprecated)"                         ELSE NO)
  endif()
endif(DEFINED APPLE)

if(DEFINED WITH_UNICAP)
  status("    UniCap:"         HAVE_UNICAP         THEN "YES (ver ${ALIASOF_libunicap_VERSION})"   ELSE NO)
  status("    UniCap ucil:"    HAVE_UNICAP_UCIL    THEN "YES (ver ${ALIASOF_libucil_VERSION})"     ELSE NO)
endif(DEFINED WITH_UNICAP)

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
  status("    V4L/V4L2:"       HAVE_LIBV4L
             THEN "Using libv4l1 (ver ${ALIASOF_libv4l1_VERSION}) / libv4l2 (ver ${ALIASOF_libv4l2_VERSION})"
             ELSE "${HAVE_CAMV4L_STR}/${HAVE_CAMV4L2_STR}")
endif(DEFINED WITH_V4L)

if(DEFINED WITH_DSHOW)
  status("    DirectShow:"     HAVE_DSHOW     THEN YES                                        ELSE NO)
endif(DEFINED WITH_DSHOW)

if(DEFINED WITH_MSMF)
  status("    Media Foundation:" HAVE_MSMF    THEN YES                                        ELSE NO)
endif(DEFINED WITH_MSMF)

if(DEFINED WITH_XIMEA)
  status("    XIMEA:"          HAVE_XIMEA          THEN YES                                        ELSE NO)
endif(DEFINED WITH_XIMEA)

if(DEFINED WITH_XINE)
  status("    Xine:"           HAVE_XINE           THEN "YES (ver ${ALIASOF_libxine_VERSION})"     ELSE NO)
endif(DEFINED WITH_XINE)

if(DEFINED WITH_INTELPERC)
  status("    Intel PerC:"     HAVE_INTELPERC      THEN "YES"                                 ELSE NO)
endif(DEFINED WITH_INTELPERC)

if(DEFINED WITH_GPHOTO2)
  status("    gPhoto2:"        HAVE_GPHOTO2        THEN "YES"                                 ELSE NO)
endif(DEFINED WITH_GPHOTO2)


# Order is similar to CV_PARALLEL_FRAMEWORK in core/src/parallel.cpp
ocv_clear_vars(CV_PARALLEL_FRAMEWORK)
if(HAVE_TBB)
  set(CV_PARALLEL_FRAMEWORK "TBB (ver ${TBB_VERSION_MAJOR}.${TBB_VERSION_MINOR} interface ${TBB_INTERFACE_VERSION})")
elseif(HAVE_CSTRIPES)
  set(CV_PARALLEL_FRAMEWORK "C=")
elseif(HAVE_OPENMP)
  set(CV_PARALLEL_FRAMEWORK "OpenMP")
elseif(HAVE_GCD)
  set(CV_PARALLEL_FRAMEWORK "GCD")
elseif(WINRT OR HAVE_CONCURRENCY)
  set(CV_PARALLEL_FRAMEWORK "Concurrency")
elseif(HAVE_PTHREADS_PF)
  set(CV_PARALLEL_FRAMEWORK "pthreads")
else()
  set(CV_PARALLEL_FRAMEWORK "none")
endif()
status("")
status("  Parallel framework:" TRUE THEN "${CV_PARALLEL_FRAMEWORK}" ELSE NO)


# ========================== Other third-party libraries ==========================
status("")
status("  Other third-party libraries:")

if(WITH_IPP AND HAVE_IPP)
  status("    Use IPP:" "${IPP_VERSION_STR} [${IPP_VERSION_MAJOR}.${IPP_VERSION_MINOR}.${IPP_VERSION_BUILD}]")
  status("         at:" "${IPP_ROOT_DIR}")
  if(NOT HAVE_IPP_ICV_ONLY)
    status("     linked:" BUILD_WITH_DYNAMIC_IPP THEN "dynamic" ELSE "static")
  endif()
else()
  status("    Use IPP:"   WITH_IPP AND NOT HAVE_IPP THEN "IPP not found or implicitly disabled" ELSE NO)
endif()

if(DEFINED WITH_IPP_A)
status("    Use IPP Async:"  HAVE_IPP_A       THEN "YES" ELSE NO)
endif(DEFINED WITH_IPP_A)

if(DEFINED WITH_VA)
status("    Use VA:"            HAVE_VA          THEN "YES" ELSE NO)
endif(DEFINED WITH_VA)

if(DEFINED WITH_VA_INTEL)
status("    Use Intel VA-API/OpenCL:"  HAVE_VA_INTEL       THEN "YES (MSDK: ${VA_INTEL_MSDK_ROOT}  OpenCL: ${VA_INTEL_IOCL_ROOT})" ELSE NO)
endif(DEFINED WITH_VA_INTEL)

if(DEFINED WITH_LAPACK)
status("    Use Lapack:"      HAVE_LAPACK     THEN "YES (${LAPACK_LIBRARIES})" ELSE NO)
endif(DEFINED WITH_LAPACK)

status("    Use Eigen:"      HAVE_EIGEN       THEN "YES (ver ${EIGEN_WORLD_VERSION}.${EIGEN_MAJOR_VERSION}.${EIGEN_MINOR_VERSION})" ELSE NO)
status("    Use Cuda:"       HAVE_CUDA        THEN "YES (ver ${CUDA_VERSION_STRING})" ELSE NO)
status("    Use OpenCL:"     HAVE_OPENCL      THEN YES ELSE NO)
status("    Use OpenVX:"     HAVE_OPENVX      THEN "YES (${OPENVX_LIBRARIES})" ELSE "NO")
status("    Use custom HAL:" OpenCV_USED_HAL  THEN "YES (${OpenCV_USED_HAL})" ELSE "NO")

if(HAVE_CUDA)
  status("")
  status("  NVIDIA CUDA")

  status("    Use CUFFT:"            HAVE_CUFFT   THEN YES ELSE NO)
  status("    Use CUBLAS:"           HAVE_CUBLAS  THEN YES ELSE NO)
  status("    USE NVCUVID:"          HAVE_NVCUVID THEN YES ELSE NO)
  status("    NVIDIA GPU arch:"      ${OPENCV_CUDA_ARCH_BIN})
  status("    NVIDIA PTX archs:"     ${OPENCV_CUDA_ARCH_PTX})
  status("    Use fast math:"        CUDA_FAST_MATH THEN YES ELSE NO)
endif()

if(HAVE_OPENCL)
  status("")
  if(HAVE_OPENCL_STATIC)
    set(__opencl_type "<Link with OpenCL library>")
  else()
    set(__opencl_type "<Dynamic loading of OpenCL library>")
  endif()
  status("  OpenCL:"                 ${__opencl_type})
  if(OPENCL_INCLUDE_DIRS)
    status("    Include path:"       ${OPENCL_INCLUDE_DIRS})
  endif()
  if(OPENCL_LIBRARIES)
    set(__libs "")
    foreach(l ${OPENCL_LIBRARIES})
      if(TARGET ${l})
        get_target_property(p ${l} LOCATION)
        if(p MATCHES NOTFOUND)
          list(APPEND __libs "${l}")
        else()
          list(APPEND __libs "${p}")
        endif()
      else()
        list(APPEND __libs "${l}")
      endif()
    endforeach()
    status("    Link libraries:"          ${__libs})
  endif()
  status("    Use AMDFFT:"           HAVE_CLAMDFFT  THEN YES ELSE NO)
  status("    Use AMDBLAS:"          HAVE_CLAMDBLAS THEN YES ELSE NO)
endif()

# ========================== python ==========================
status("")
status("  Python 2:")
status("    Interpreter:"     PYTHON2INTERP_FOUND  THEN "${PYTHON2_EXECUTABLE} (ver ${PYTHON2_VERSION_STRING})"       ELSE NO)
if(BUILD_opencv_python2)
  if(PYTHON2LIBS_VERSION_STRING)
    status("    Libraries:"   HAVE_opencv_python2  THEN  "${PYTHON2_LIBRARIES} (ver ${PYTHON2LIBS_VERSION_STRING})"   ELSE NO)
  else()
    status("    Libraries:"   HAVE_opencv_python2  THEN  "${PYTHON2_LIBRARIES}"                                      ELSE NO)
  endif()
  status("    numpy:"         PYTHON2_NUMPY_INCLUDE_DIRS THEN "${PYTHON2_NUMPY_INCLUDE_DIRS} (ver ${PYTHON2_NUMPY_VERSION})" ELSE "NO (Python wrappers can not be generated)")
  status("    packages path:" PYTHON2_EXECUTABLE         THEN "${PYTHON2_PACKAGES_PATH}"                                    ELSE "-")
endif()

status("")
status("  Python 3:")
status("    Interpreter:"     PYTHON3INTERP_FOUND  THEN "${PYTHON3_EXECUTABLE} (ver ${PYTHON3_VERSION_STRING})"       ELSE NO)
if(BUILD_opencv_python3)
  if(PYTHON3LIBS_VERSION_STRING)
    status("    Libraries:"   HAVE_opencv_python3  THEN  "${PYTHON3_LIBRARIES} (ver ${PYTHON3LIBS_VERSION_STRING})"   ELSE NO)
  else()
    status("    Libraries:"   HAVE_opencv_python3  THEN  "${PYTHON3_LIBRARIES}"                                      ELSE NO)
  endif()
  status("    numpy:"         PYTHON3_NUMPY_INCLUDE_DIRS THEN "${PYTHON3_NUMPY_INCLUDE_DIRS} (ver ${PYTHON3_NUMPY_VERSION})" ELSE "NO (Python3 wrappers can not be generated)")
  status("    packages path:" PYTHON3_EXECUTABLE         THEN "${PYTHON3_PACKAGES_PATH}"                                    ELSE "-")
endif()

status("")
status("  Python (for build):"  PYTHON_DEFAULT_AVAILABLE THEN "${PYTHON_DEFAULT_EXECUTABLE}" ELSE NO)

# ========================== java ==========================
status("")
status("  Java:")
status("    ant:"           ANT_EXECUTABLE      THEN "${ANT_EXECUTABLE} (ver ${ANT_VERSION})"                    ELSE NO)
if(NOT ANDROID)
  status("    JNI:"         JNI_INCLUDE_DIRS    THEN "${JNI_INCLUDE_DIRS}"                                       ELSE NO)
endif()
status("    Java wrappers:" HAVE_opencv_java                                                            THEN YES ELSE NO)
status("    Java tests:"    BUILD_TESTS AND opencv_test_java_BINARY_DIR                                 THEN YES ELSE NO)

# ========================= matlab =========================
status("")
if(MATLAB_FOUND)
  status("  Matlab:")
  status("    mex:"         MATLAB_MEX_SCRIPT  THEN  "${MATLAB_MEX_SCRIPT}"   ELSE NO)
  status("    Compiler/generator:" MEX_WORKS   THEN  "Working"                ELSE "Not working (bindings will not be generated)")
else()
  status("  Matlab:"        WITH_MATLAB AND NOT MATLAB_FOUND THEN "Matlab not found or implicitly disabled" ELSE NO)
endif()

# ========================== documentation ==========================
if(BUILD_DOCS)
  status("")
  status("  Documentation:")
  status("    Doxygen:"             DOXYGEN_FOUND             THEN "${DOXYGEN_EXECUTABLE} (ver ${DOXYGEN_VERSION})" ELSE NO)
endif()

# ========================== samples and tests ==========================
status("")
status("  Tests and samples:")
status("    Tests:"             BUILD_TESTS AND HAVE_opencv_ts       THEN YES ELSE NO)
status("    Performance tests:" BUILD_PERF_TESTS AND HAVE_opencv_ts  THEN YES ELSE NO)
status("    C/C++ Examples:"    BUILD_EXAMPLES                       THEN YES ELSE NO)

# ========================== auxiliary ==========================
status("")
status("  Install path:" "${CMAKE_INSTALL_PREFIX}")
status("")
status("  cvconfig.h is in:" "${OPENCV_CONFIG_FILE_INCLUDE_DIR}")
status("-----------------------------------------------------------------")
status("")

ocv_finalize_status()

# ----------------------------------------------------------------------------
# Warn in the case of in-source build
# ----------------------------------------------------------------------------
if("${CMAKE_CURRENT_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_BINARY_DIR}")
  message(WARNING "The source directory is the same as binary directory. \"make clean\" may damage the source tree")
endif()

# ----------------------------------------------------------------------------
# CPack stuff
# ----------------------------------------------------------------------------

include(cmake/OpenCVPackaging.cmake)

# This should be the last command
ocv_cmake_dump_vars("" TOFILE "CMakeVars.txt")
ocv_cmake_eval(DEBUG_POST ONCE)
