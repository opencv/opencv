# ----------------------------------------------------------------------------
#  OpenCVConfigure.cmake
#  The main OpenCV cmake configuration helper
#  The following options can be parsed once the "project(OpenCV ...)"
#  OpenCV options and CXXCompiler have been specified.
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
#  Standard paths
# ----------------------------------------------------------------------------

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
#  Build & install layouts
# ----------------------------------------------------------------------------

# Save libs and executables in the same place
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin" CACHE PATH "Output directory for applications" )

if(ANDROID OR WIN32)
  set(OPENCV_DOC_INSTALL_PATH doc)
elseif(INSTALL_TO_MANGLED_PATHS)
  set(OPENCV_DOC_INSTALL_PATH share/OpenCV-${OPENCV_VERSION}/doc)
else()
  set(OPENCV_DOC_INSTALL_PATH share/OpenCV/doc)
endif()

if(ANDROID)
  set(LIBRARY_OUTPUT_PATH         "${OpenCV_BINARY_DIR}/lib/${ANDROID_NDK_ABI_NAME}")
  set(3P_LIBRARY_OUTPUT_PATH      "${OpenCV_BINARY_DIR}/3rdparty/lib/${ANDROID_NDK_ABI_NAME}")
  set(OPENCV_LIB_INSTALL_PATH     sdk/native/libs/${ANDROID_NDK_ABI_NAME})
  set(OPENCV_3P_LIB_INSTALL_PATH  sdk/native/3rdparty/libs/${ANDROID_NDK_ABI_NAME})
  set(OPENCV_CONFIG_INSTALL_PATH  sdk/native/jni)
  set(OPENCV_INCLUDE_INSTALL_PATH sdk/native/jni/include)
else()
  set(LIBRARY_OUTPUT_PATH         "${OpenCV_BINARY_DIR}/lib")
  set(3P_LIBRARY_OUTPUT_PATH      "${OpenCV_BINARY_DIR}/3rdparty/lib${LIB_SUFFIX}")
  set(OPENCV_LIB_INSTALL_PATH     lib${LIB_SUFFIX})
  set(OPENCV_3P_LIB_INSTALL_PATH  share/OpenCV/3rdparty/${OPENCV_LIB_INSTALL_PATH})
  set(OPENCV_INCLUDE_INSTALL_PATH include)

  math(EXPR SIZEOF_VOID_P_BITS "8 * ${CMAKE_SIZEOF_VOID_P}")
  if(LIB_SUFFIX AND NOT SIZEOF_VOID_P_BITS EQUAL LIB_SUFFIX)
    set(OPENCV_CONFIG_INSTALL_PATH lib${LIB_SUFFIX}/cmake/opencv)
  else()
    set(OPENCV_CONFIG_INSTALL_PATH share/OpenCV)
  endif()
endif()

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${OPENCV_LIB_INSTALL_PATH}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

if(INSTALL_TO_MANGLED_PATHS)
  set(OPENCV_INCLUDE_INSTALL_PATH ${OPENCV_INCLUDE_INSTALL_PATH}/opencv-${OPENCV_VERSION})
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


# ----------------------------------------------------------------------------
#  Path for build/platform -specific headers
# ----------------------------------------------------------------------------
set(OPENCV_CONFIG_FILE_INCLUDE_DIR "${CMAKE_BINARY_DIR}/" CACHE PATH "Where to create the platform-dependant cvconfig.h")
ocv_include_directories(${OPENCV_CONFIG_FILE_INCLUDE_DIR})


# ----------------------------------------------------------------------------
#  Path for additional modules
# ----------------------------------------------------------------------------
set(OPENCV_EXTRA_MODULES_PATH "" CACHE PATH "Where to look for additional OpenCV modules")


# ----------------------------------------------------------------------------
#  Autodetect if we are in a GIT repository
# ----------------------------------------------------------------------------
find_host_package(Git QUIET)

if(GIT_FOUND)
  execute_process(COMMAND "${GIT_EXECUTABLE}" describe --tags --always --dirty --match "2.[0-9].[0-9]*"
    WORKING_DIRECTORY "${OpenCV_SOURCE_DIR}"
    OUTPUT_VARIABLE OPENCV_VCSVERSION
    RESULT_VARIABLE GIT_RESULT
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(NOT GIT_RESULT EQUAL 0)
    set(OPENCV_VCSVERSION "unknown")
  endif()
else()
  # We don't have git:
  set(OPENCV_VCSVERSION "unknown")
endif()


# ----------------------------------------------------------------------------
#  OpenCV compiler and linker options
# ----------------------------------------------------------------------------
# In case of Makefiles if the user does not setup CMAKE_BUILD_TYPE, assume it's Release:
if(CMAKE_GENERATOR MATCHES "Makefiles|Ninja" AND "${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE Release)
endif()

include(cmake/OpenCVCompilerOptions.cmake)


# ----------------------------------------------------------------------------
#  Use statically or dynamically linked CRT?
#  Default: dynamic
# ----------------------------------------------------------------------------
if(MSVC)
  include(cmake/OpenCVCRTLinkage.cmake)
endif(MSVC)

if(WIN32 AND NOT MINGW)
  add_definitions(-D_VARIADIC_MAX=10)
endif(WIN32 AND NOT MINGW)


# ----------------------------------------------------------------------------
#  CHECK FOR SYSTEM LIBRARIES, OPTIONS, ETC..
# ----------------------------------------------------------------------------
if(UNIX)
  find_package(PkgConfig QUIET)
  include(CheckFunctionExists)
  include(CheckIncludeFile)

  if(NOT APPLE)
    CHECK_INCLUDE_FILE(pthread.h HAVE_LIBPTHREAD)
    if(ANDROID)
      set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} dl m log)
    elseif(${CMAKE_SYSTEM_NAME} MATCHES "FreeBSD|NetBSD|DragonFly")
      set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} m pthread)
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
#  Detect 3rd-party libraries
# ----------------------------------------------------------------------------
include(cmake/OpenCVFindLibsGrfmt.cmake)
include(cmake/OpenCVFindLibsGUI.cmake)
include(cmake/OpenCVFindLibsVideo.cmake)
include(cmake/OpenCVFindLibsPerf.cmake)


# ----------------------------------------------------------------------------
#  Detect other 3rd-party libraries/tools
# ----------------------------------------------------------------------------

# --- LATEX for pdf documentation ---
if(BUILD_DOCS)
  include(cmake/OpenCVFindLATEX.cmake)
endif(BUILD_DOCS)

# --- OpenCL ---
if(WITH_OPENCL)
  include(cmake/OpenCVDetectOpenCL.cmake)
endif()


# ----------------------------------------------------------------------------
#  Language bindings
# ----------------------------------------------------------------------------

# --- Python Support, required for all language bindings ---
if (BUILD_PYTHON_BINDINGS OR BUILD_JAVA_BINDINGS OR BUILD_MATLAB_BINDINGS OR ANDROID OR BUILD_DOCS)
  include(cmake/OpenCVDetectPython.cmake)
endif()

# --- Java Support ---
if(ANDROID OR BUILD_JAVA_BINDINGS)
  include(cmake/OpenCVDetectApacheAnt.cmake)
  if(ANDROID)
    include(cmake/OpenCVDetectAndroidSDK.cmake)
    if(NOT ANDROID_TOOLS_Pkg_Revision GREATER 13)
      message(WARNING "OpenCV requires Android SDK tools revision 14 or newer. Otherwise tests and samples will no be compiled.")
    endif()
  else()
    find_package(JNI)
  endif()
endif()

if(ANDROID AND ANDROID_EXECUTABLE AND ANT_EXECUTABLE AND (ANT_VERSION VERSION_GREATER 1.7) AND (ANDROID_TOOLS_Pkg_Revision GREATER 13))
  SET(CAN_BUILD_ANDROID_PROJECTS TRUE)
else()
  SET(CAN_BUILD_ANDROID_PROJECTS FALSE)
endif()

# --- Matlab/Octave ---
if(BUILD_MATLAB_BINDINGS)
  include(cmake/OpenCVFindMatlab.cmake)
endif()


# ----------------------------------------------------------------------------
#  Solution folders
# ----------------------------------------------------------------------------
if(ENABLE_SOLUTION_FOLDERS)
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
  set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "CMakeTargets")
endif()

# Extra OpenCV targets: uninstall, package_source, perf, etc.
include(cmake/OpenCVExtraTargets.cmake)


# ----------------------------------------------------------------------------
#  Process subdirectories
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

if(BUILD_ANDROID_PACKAGE)
  add_subdirectory(platforms/android/package)
endif()

if (ANDROID)
  add_subdirectory(platforms/android/libinfo)
endif()


# ----------------------------------------------------------------------------
#  Finalization: generate configuration-based files
# ----------------------------------------------------------------------------
ocv_track_build_dependencies()

# Generate platform-dependent and configuration-dependent headers
include(cmake/OpenCVGenHeaders.cmake)

# Generate opencv.pc for pkg-config command
include(cmake/OpenCVGenPkgconfig.cmake)

# Generate OpenCV.mk for ndk-build (Android build tool)
include(cmake/OpenCVGenAndroidMK.cmake)

# Generate OpenCVСonfig.cmake and OpenCVConfig-version.cmake for cmake projects
include(cmake/OpenCVGenConfig.cmake)

# Generate Info.plist for the IOS framework
include(cmake/OpenCVGenInfoPlist.cmake)
