#
# The script to detect Intel(R) Integrated Performance Primitives (IPP)
# installation/package
#
# Windows host:
# Run script like this before cmake:
#   call "<IPP_INSTALL_DIR>\bin\ippvars.bat" intel64
# for example:
#   call "C:\Program Files (x86)\Intel\Composer XE\ipp\bin\ippvars.bat" intel64
#
# Linux host:
# Run script like this before cmake:
#   source /opt/intel/ipp/bin/ippvars.sh [ia32|intel64]
#
# On return this will define:
#
# HAVE_IPP          - True if Intel IPP found
# HAVE_IPP_ICV_ONLY - True if Intel IPP ICV version is available
# IPP_ROOT_DIR      - root of IPP installation
# IPP_INCLUDE_DIRS  - IPP include folder
# IPP_LIBRARIES     - IPP libraries that are used by OpenCV
# IPP_VERSION_STR   - string with the newest detected IPP version
# IPP_VERSION_MAJOR - numbers of IPP version (MAJOR.MINOR.BUILD)
# IPP_VERSION_MINOR
# IPP_VERSION_BUILD
#
# Created: 30 Dec 2010 by Vladimir Dudnik (vladimir.dudnik@intel.com)
#

unset(HAVE_IPP CACHE)
unset(HAVE_IPP_ICV_ONLY)
unset(IPP_ROOT_DIR)
unset(IPP_INCLUDE_DIRS)
unset(IPP_LIBRARIES)
unset(IPP_VERSION_STR)
unset(IPP_VERSION_MAJOR)
unset(IPP_VERSION_MINOR)
unset(IPP_VERSION_BUILD)

set(IPP_LIB_PREFIX ${CMAKE_STATIC_LIBRARY_PREFIX})
set(IPP_LIB_SUFFIX  ${CMAKE_STATIC_LIBRARY_SUFFIX})
set(IPP_PREFIX "ipp")
set(IPP_SUFFIX "_l")
set(IPPCORE    "core") # core functionality
set(IPPS       "s")    # signal processing
set(IPPI       "i")    # image processing
set(IPPCC      "cc")   # color conversion
set(IPPCV      "cv")   # computer vision
set(IPPVM      "vm")   # vector math

set(IPP_X64 0)
if(CMAKE_CXX_SIZEOF_DATA_PTR EQUAL 8)
    set(IPP_X64 1)
endif()
if(CMAKE_CL_64)
    set(IPP_X64 1)
endif()

# This function detects IPP version by analyzing ippversion.h file
macro(ipp_get_version _ROOT_DIR)
  unset(_VERSION_STR)
  unset(_MAJOR)
  unset(_MINOR)
  unset(_BUILD)

  # read IPP version info from file
  file(STRINGS ${_ROOT_DIR}/include/ippversion.h STR1 REGEX "IPP_VERSION_MAJOR")
  file(STRINGS ${_ROOT_DIR}/include/ippversion.h STR2 REGEX "IPP_VERSION_MINOR")
  file(STRINGS ${_ROOT_DIR}/include/ippversion.h STR3 REGEX "IPP_VERSION_BUILD")
  if("${STR3}" STREQUAL "")
    file(STRINGS ${_ROOT_DIR}/include/ippversion.h STR3 REGEX "IPP_VERSION_UPDATE")
  endif()
  file(STRINGS ${_ROOT_DIR}/include/ippversion.h STR4 REGEX "IPP_VERSION_STR")

  # extract info and assign to variables
  string(REGEX MATCHALL "[0-9]+" _MAJOR ${STR1})
  string(REGEX MATCHALL "[0-9]+" _MINOR ${STR2})
  string(REGEX MATCHALL "[0-9]+" _BUILD ${STR3})
  string(REGEX MATCHALL "[0-9]+[.]+[0-9]+[^\"]+|[0-9]+[.]+[0-9]+" _VERSION_STR ${STR4})

  # export info to parent scope
  set(IPP_VERSION_STR   ${_VERSION_STR})
  set(IPP_VERSION_MAJOR ${_MAJOR})
  set(IPP_VERSION_MINOR ${_MINOR})
  set(IPP_VERSION_BUILD ${_BUILD})

  set(__msg)
  if(EXISTS ${_ROOT_DIR}/include/ippicv.h)
    ocv_assert(WITH_ICV AND NOT WITH_IPP)
    set(__msg " ICV version")
    set(HAVE_IPP_ICV_ONLY 1)
  endif()

  message(STATUS "found IPP: ${_MAJOR}.${_MINOR}.${_BUILD} [${_VERSION_STR}]${__msg}")
  message(STATUS "at: ${_ROOT_DIR}")
endmacro()


# This function sets IPP_INCLUDE_DIRS and IPP_LIBRARIES variables
macro(ipp_set_variables _LATEST_VERSION)
  if(${_LATEST_VERSION} VERSION_LESS "7.0")
    message(SEND_ERROR "IPP ${_LATEST_VERSION} is not supported")
    unset(HAVE_IPP)
    return()
  endif()

  # set INCLUDE and LIB folders
  set(IPP_INCLUDE_DIRS ${IPP_ROOT_DIR}/include)

  if(NOT HAVE_IPP_ICV_ONLY)
    if(APPLE)
      set(IPP_LIBRARY_DIR ${IPP_ROOT_DIR}/lib)
    elseif(IPP_X64)
      if(NOT EXISTS ${IPP_ROOT_DIR}/lib/intel64)
        message(SEND_ERROR "IPP EM64T libraries not found")
      endif()
      set(IPP_LIBRARY_DIR ${IPP_ROOT_DIR}/lib/intel64)
    else()
      if(NOT EXISTS ${IPP_ROOT_DIR}/lib/ia32)
        message(SEND_ERROR "IPP IA32 libraries not found")
      endif()
      set(IPP_LIBRARY_DIR ${IPP_ROOT_DIR}/lib/ia32)
    endif()
  else()
    if(APPLE)
      set(IPP_LIBRARY_DIR ${IPP_ROOT_DIR}/libs/macosx)
    elseif(WIN32 AND NOT ARM)
      set(IPP_LIBRARY_DIR ${IPP_ROOT_DIR}/libs/windows)
    elseif(UNIX)
      set(IPP_LIBRARY_DIR ${IPP_ROOT_DIR}/libs/linux)
    else()
      message(MESSAGE "IPP ${_LATEST_VERSION} at ${IPP_ROOT_DIR} is not supported")
      unset(HAVE_IPP)
      return()
    endif()
    if(X86_64)
      set(IPP_LIBRARY_DIR ${IPP_LIBRARY_DIR}/intel64)
    else()
      set(IPP_LIBRARY_DIR ${IPP_LIBRARY_DIR}/ia32)
    endif()
  endif()

  set(IPP_PREFIX "ipp")
  if(${_LATEST_VERSION} VERSION_LESS "8.0")
    set(IPP_SUFFIX "_l")        # static not threaded libs suffix IPP 7.x
  else()
    if(WIN32)
      set(IPP_SUFFIX "mt")    # static not threaded libs suffix IPP 8.x for Windows
    else()
      set(IPP_SUFFIX "")      # static not threaded libs suffix IPP 8.x for Linux/OS X
    endif()
  endif()
  set(IPPCORE "core")     # core functionality
  set(IPPSP   "s")        # signal processing
  set(IPPIP   "i")        # image processing
  set(IPPCC   "cc")       # color conversion
  set(IPPCV   "cv")       # computer vision
  set(IPPVM   "vm")       # vector math

  list(APPEND IPP_LIBRARIES ${IPP_LIBRARY_DIR}/${IPP_LIB_PREFIX}${IPP_PREFIX}${IPPVM}${IPP_SUFFIX}${IPP_LIB_SUFFIX})
  list(APPEND IPP_LIBRARIES ${IPP_LIBRARY_DIR}/${IPP_LIB_PREFIX}${IPP_PREFIX}${IPPCC}${IPP_SUFFIX}${IPP_LIB_SUFFIX})
  list(APPEND IPP_LIBRARIES ${IPP_LIBRARY_DIR}/${IPP_LIB_PREFIX}${IPP_PREFIX}${IPPCV}${IPP_SUFFIX}${IPP_LIB_SUFFIX})
  list(APPEND IPP_LIBRARIES ${IPP_LIBRARY_DIR}/${IPP_LIB_PREFIX}${IPP_PREFIX}${IPPI}${IPP_SUFFIX}${IPP_LIB_SUFFIX})
  list(APPEND IPP_LIBRARIES ${IPP_LIBRARY_DIR}/${IPP_LIB_PREFIX}${IPP_PREFIX}${IPPS}${IPP_SUFFIX}${IPP_LIB_SUFFIX})
  list(APPEND IPP_LIBRARIES ${IPP_LIBRARY_DIR}/${IPP_LIB_PREFIX}${IPP_PREFIX}${IPPCORE}${IPP_SUFFIX}${IPP_LIB_SUFFIX})

# FIXIT
#  if(UNIX AND NOT HAVE_IPP_ICV_ONLY)
#    get_filename_component(INTEL_COMPILER_LIBRARY_DIR ${IPP_ROOT_DIR}/../lib REALPATH)
  if(UNIX)
    if(NOT HAVE_IPP_ICV_ONLY)
      get_filename_component(INTEL_COMPILER_LIBRARY_DIR ${IPP_ROOT_DIR}/../lib REALPATH)
    else()
      set(INTEL_COMPILER_LIBRARY_DIR "/opt/intel/lib")
    endif()
    if(IPP_X64)
      if(NOT EXISTS ${INTEL_COMPILER_LIBRARY_DIR}/intel64)
        message(SEND_ERROR "Intel compiler EM64T libraries not found")
      endif()
      set(INTEL_COMPILER_LIBRARY_DIR ${INTEL_COMPILER_LIBRARY_DIR}/intel64)
    else()
      if(NOT EXISTS ${INTEL_COMPILER_LIBRARY_DIR}/ia32)
        message(SEND_ERROR "Intel compiler IA32 libraries not found")
      endif()
      set(INTEL_COMPILER_LIBRARY_DIR ${INTEL_COMPILER_LIBRARY_DIR}/ia32)
    endif()
    list(APPEND IPP_LIBRARIES ${INTEL_COMPILER_LIBRARY_DIR}/${IPP_LIB_PREFIX}irc${CMAKE_SHARED_LIBRARY_SUFFIX})
    list(APPEND IPP_LIBRARIES ${INTEL_COMPILER_LIBRARY_DIR}/${IPP_LIB_PREFIX}imf${CMAKE_SHARED_LIBRARY_SUFFIX})
    list(APPEND IPP_LIBRARIES ${INTEL_COMPILER_LIBRARY_DIR}/${IPP_LIB_PREFIX}svml${CMAKE_SHARED_LIBRARY_SUFFIX})
  endif()

  #message(STATUS "IPP libs: ${IPP_LIBRARIES}")
endmacro()

if(WITH_IPP)
  set(IPPPATH $ENV{IPPROOT})
  if(UNIX)
    list(APPEND IPPPATH /opt/intel/ipp)
  endif()
elseif(WITH_ICV)
  if(DEFINED ENV{IPPICVROOT})
    set(IPPPATH $ENV{IPPICVROOT})
  else()
    set(IPPPATH ${OpenCV_SOURCE_DIR}/3rdparty/ippicv)
  endif()
endif()


find_path(
    IPP_H_PATH
    NAMES ippversion.h
    PATHS ${IPPPATH}
    PATH_SUFFIXES include
    DOC "The path to Intel(R) IPP header files"
    NO_DEFAULT_PATH
    NO_CMAKE_PATH)

if(IPP_H_PATH)
    set(HAVE_IPP 1)

    get_filename_component(IPP_ROOT_DIR ${IPP_H_PATH} PATH)

    ipp_get_version(${IPP_ROOT_DIR})
    ipp_set_variables(${IPP_VERSION_STR})
endif()


if(WIN32 AND MINGW AND NOT IPP_VERSION_MAJOR LESS 7)
    # Since IPP built with Microsoft compiler and /GS option
    # ======================================================
    # From Windows SDK 7.1
    #   (usually in "C:\Program Files\Microsoft Visual Studio 10.0\VC\lib"),
    # to avoid undefined reference to __security_cookie and _chkstk:
    set(MSV_RUNTMCHK "RunTmChk")
    set(IPP_LIBRARIES ${IPP_LIBRARIES} ${MSV_RUNTMCHK}${IPP_LIB_SUFFIX})

    # To avoid undefined reference to _alldiv and _chkstk
    # ===================================================
    # NB: it may require a recompilation of w32api (after having modified
    #     the file ntdll.def) to export the required functions
    #     See http://code.opencv.org/issues/1906 for additional details
    set(MSV_NTDLL    "ntdll")
    set(IPP_LIBRARIES ${IPP_LIBRARIES} ${MSV_NTDLL}${IPP_LIB_SUFFIX})
endif()
