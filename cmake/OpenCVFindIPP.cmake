#
# The script to detect Intel(R) Integrated Performance Primitives (IPP)
# installation/package
#
# By default, ICV version will be used.
# To use standalone IPP update cmake command line:
# cmake ... -DIPPROOT=<path> ...
#
# Note: Backward compatibility is broken, IPPROOT environment path is ignored
#
#
# On return this will define:
#
# HAVE_IPP            - True if Intel IPP found
# HAVE_IPP_ICV        - True if Intel IPP ICV version is available
# IPP_ROOT_DIR        - root of IPP installation
# IPP_INCLUDE_DIRS    - IPP include folder
# IPP_LIBRARIES       - IPP libraries that are used by OpenCV
# IPP_VERSION_STR     - string with the newest detected IPP version
# IPP_VERSION_MAJOR   - numbers of IPP version (MAJOR.MINOR.BUILD)
# IPP_VERSION_MINOR
# IPP_VERSION_BUILD
#
# Created: 30 Dec 2010 by Vladimir Dudnik (vladimir.dudnik@intel.com)
#

unset(HAVE_IPP CACHE)
unset(HAVE_IPP_ICV)
unset(IPP_ROOT_DIR)
unset(IPP_INCLUDE_DIRS)
unset(IPP_LIBRARIES)
unset(IPP_VERSION_STR)
unset(IPP_VERSION_MAJOR)
unset(IPP_VERSION_MINOR)
unset(IPP_VERSION_BUILD)

if (X86 AND UNIX AND NOT APPLE AND NOT ANDROID AND BUILD_SHARED_LIBS)
    message(STATUS "On 32-bit Linux Intel IPP can not currently be used with dynamic libs because of linker errors. Set BUILD_SHARED_LIBS=OFF")
    return()
endif()

set(IPP_X64 0)
if(CMAKE_CXX_SIZEOF_DATA_PTR EQUAL 8)
    set(IPP_X64 1)
endif()
if(CMAKE_CL_64)
    set(IPP_X64 1)
endif()

# This function detects Intel IPP version by analyzing .h file
macro(ipp_get_version VERSION_FILE)
  unset(_VERSION_STR)
  unset(_MAJOR)
  unset(_MINOR)
  unset(_BUILD)

  # read Intel IPP version info from file
  file(STRINGS ${VERSION_FILE} STR1 REGEX "IPP_VERSION_MAJOR")
  file(STRINGS ${VERSION_FILE} STR2 REGEX "IPP_VERSION_MINOR")
  file(STRINGS ${VERSION_FILE} STR3 REGEX "IPP_VERSION_BUILD")
  if("${STR3}" STREQUAL "")
    file(STRINGS ${VERSION_FILE} STR3 REGEX "IPP_VERSION_UPDATE")
  endif()
  file(STRINGS ${VERSION_FILE} STR4 REGEX "IPP_VERSION_STR")

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
endmacro()

macro(_ipp_not_supported)
  message(STATUS ${ARGN})
  unset(HAVE_IPP)
  unset(HAVE_IPP_ICV)
  unset(IPP_VERSION_STR)
  return()
endmacro()

# This macro uses IPP_ROOT_DIR variable
# TODO Cleanup code after ICV package stabilization
macro(ipp_detect_version)
  set(IPP_INCLUDE_DIRS ${IPP_ROOT_DIR}/include)

  set(__msg)
  if(EXISTS ${IPP_ROOT_DIR}/include/ippicv_redefs.h)
    set(__msg " (ICV version)")
    set(HAVE_IPP_ICV 1)
  elseif(EXISTS ${IPP_ROOT_DIR}/include/ipp.h)
    # nothing
  else()
    _ipp_not_supported("Can't resolve Intel IPP directory: ${IPP_ROOT_DIR}")
  endif()

  ipp_get_version(${IPP_INCLUDE_DIRS}/ippversion.h)
  ocv_assert(IPP_VERSION_STR VERSION_GREATER "1.0")

  message(STATUS "found Intel IPP${__msg}: ${_MAJOR}.${_MINOR}.${_BUILD} [${IPP_VERSION_STR}]")
  message(STATUS "at: ${IPP_ROOT_DIR}")

  if(IPP_VERSION_STR VERSION_LESS "7.0")
    _ipp_not_supported("Intel IPP ${IPP_VERSION_STR} is not supported")
  endif()

  set(HAVE_IPP 1)

  macro(_ipp_set_library_dir DIR)
    if(NOT EXISTS ${DIR})
      _ipp_not_supported("Intel IPP library directory not found")
    endif()
    set(IPP_LIBRARY_DIR ${DIR})
  endmacro()

  if(APPLE AND NOT HAVE_IPP_ICV)
    _ipp_set_library_dir(${IPP_ROOT_DIR}/lib)
  elseif(IPP_X64)
    _ipp_set_library_dir(${IPP_ROOT_DIR}/lib/intel64)
  else()
    _ipp_set_library_dir(${IPP_ROOT_DIR}/lib/ia32)
  endif()

  macro(_ipp_add_library name)
    # dynamic linking is only supported for standalone version of Intel IPP
    if (BUILD_WITH_DYNAMIC_IPP AND NOT HAVE_IPP_ICV)
      if (WIN32)
        set(IPP_LIB_PREFIX ${CMAKE_IMPORT_LIBRARY_PREFIX})
        set(IPP_LIB_SUFFIX ${CMAKE_IMPORT_LIBRARY_SUFFIX})
      else (WIN32)
        set(IPP_LIB_PREFIX ${CMAKE_SHARED_LIBRARY_PREFIX})
        set(IPP_LIB_SUFFIX ${CMAKE_SHARED_LIBRARY_SUFFIX})
      endif (WIN32)
    else ()
      set(IPP_LIB_PREFIX ${CMAKE_STATIC_LIBRARY_PREFIX})
      set(IPP_LIB_SUFFIX ${CMAKE_STATIC_LIBRARY_SUFFIX})
    endif ()
    if (EXISTS ${IPP_LIBRARY_DIR}/${IPP_LIB_PREFIX}${IPP_PREFIX}${name}${IPP_SUFFIX}${IPP_LIB_SUFFIX})
      if (BUILD_WITH_DYNAMIC_IPP AND NOT HAVE_IPP_ICV)
        # When using dynamic libraries from standalone Intel IPP it is your responsibility to install those on the target system
        list(APPEND IPP_LIBRARIES ${IPP_LIBRARY_DIR}/${IPP_LIB_PREFIX}${IPP_PREFIX}${name}${IPP_SUFFIX}${IPP_LIB_SUFFIX})
      else ()
        add_library(ipp${name} STATIC IMPORTED)
        set_target_properties(ipp${name} PROPERTIES
          IMPORTED_LINK_INTERFACE_LIBRARIES ""
          IMPORTED_LOCATION ${IPP_LIBRARY_DIR}/${IPP_LIB_PREFIX}${IPP_PREFIX}${name}${IPP_SUFFIX}${IPP_LIB_SUFFIX}
        )
        list(APPEND IPP_LIBRARIES ipp${name})
        if (NOT BUILD_SHARED_LIBS)
          # CMake doesn't support "install(TARGETS ${IPP_PREFIX}${name} " command with imported targets
          install(FILES ${IPP_LIBRARY_DIR}/${IPP_LIB_PREFIX}${IPP_PREFIX}${name}${IPP_SUFFIX}${IPP_LIB_SUFFIX}
                  DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev)
          string(TOUPPER ${name} uname)
          set(IPP${uname}_INSTALL_PATH "${CMAKE_INSTALL_PREFIX}/${OPENCV_3P_LIB_INSTALL_PATH}/${IPP_LIB_PREFIX}${IPP_PREFIX}${name}${IPP_SUFFIX}${IPP_LIB_SUFFIX}" CACHE INTERNAL "" FORCE)
          set(IPP${uname}_LOCATION_PATH "${IPP_LIBRARY_DIR}/${IPP_LIB_PREFIX}${IPP_PREFIX}${name}${IPP_SUFFIX}${IPP_LIB_SUFFIX}" CACHE INTERNAL "" FORCE)
        endif()
      endif()
    else()
      message(STATUS "Can't find Intel IPP library: ${name} at ${IPP_LIBRARY_DIR}/${IPP_LIB_PREFIX}${IPP_PREFIX}${name}${IPP_SUFFIX}${IPP_LIB_SUFFIX}")
    endif()
  endmacro()

  set(IPP_PREFIX "ipp")
  if(IPP_VERSION_STR VERSION_LESS "8.0")
    if (BUILD_WITH_DYNAMIC_IPP AND NOT HAVE_IPP_ICV)
      set(IPP_SUFFIX "")      # dynamic not threaded libs suffix Intel IPP 7.x
    else ()
      set(IPP_SUFFIX "_l")    # static not threaded libs suffix Intel IPP 7.x
    endif ()
  else ()
    if(WIN32)
      if (BUILD_WITH_DYNAMIC_IPP AND NOT HAVE_IPP_ICV)
        set(IPP_SUFFIX "")    # dynamic not threaded libs suffix Intel IPP 8.x for Windows
      else ()
        set(IPP_SUFFIX "mt")  # static not threaded libs suffix Intel IPP 8.x for Windows
      endif ()
    else()
      set(IPP_SUFFIX "")      # static not threaded libs suffix Intel IPP 8.x for Linux/OS X
    endif()
  endif()

  if(HAVE_IPP_ICV)
    _ipp_add_library(icv)
  else()
    _ipp_add_library(cv)
    _ipp_add_library(i)
    _ipp_add_library(cc)
    _ipp_add_library(s)
    _ipp_add_library(vm)
    _ipp_add_library(core)

    if(UNIX AND IPP_VERSION_MAJOR LESS 2017)
      get_filename_component(INTEL_COMPILER_LIBRARY_DIR ${IPP_ROOT_DIR}/../lib REALPATH)
      if(NOT EXISTS ${INTEL_COMPILER_LIBRARY_DIR})
        get_filename_component(INTEL_COMPILER_LIBRARY_DIR ${IPP_ROOT_DIR}/../compiler/lib REALPATH)
      endif()
      if(NOT EXISTS ${INTEL_COMPILER_LIBRARY_DIR})
        _ipp_not_supported("Intel IPP configuration error: can't find Intel compiler library dir ${INTEL_COMPILER_LIBRARY_DIR}")
      endif()
      if(NOT APPLE)
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
      endif()

      macro(_ipp_add_compiler_library name)
        if (EXISTS ${INTEL_COMPILER_LIBRARY_DIR}/${IPP_LIB_PREFIX}${name}${CMAKE_SHARED_LIBRARY_SUFFIX})
          list(APPEND IPP_LIBRARIES ${INTEL_COMPILER_LIBRARY_DIR}/${IPP_LIB_PREFIX}${name}${CMAKE_SHARED_LIBRARY_SUFFIX})
        else()
          message(STATUS "Can't find compiler library: ${name} at ${INTEL_COMPILER_LIBRARY_DIR}/${IPP_LIB_PREFIX}${name}${CMAKE_SHARED_LIBRARY_SUFFIX}")
        endif()
      endmacro()

      _ipp_add_compiler_library(irc)
      _ipp_add_compiler_library(imf)
      _ipp_add_compiler_library(svml)
    endif()
  endif()

  #message(STATUS "Intel IPP libs: ${IPP_LIBRARIES}")
endmacro()

# OPENCV_IPP_PATH is an environment variable for internal usage only, do not use it
if(DEFINED ENV{OPENCV_IPP_PATH} AND NOT DEFINED IPPROOT)
  set(IPPROOT "$ENV{OPENCV_IPP_PATH}")
endif()

if(NOT DEFINED IPPROOT)
  include("${OpenCV_SOURCE_DIR}/3rdparty/ippicv/ippicv.cmake")
  download_ippicv(ICV_PACKAGE_ROOT)
  if(NOT ICV_PACKAGE_ROOT)
    return()
  endif()
  set(IPPROOT "${ICV_PACKAGE_ROOT}/icv")
  ocv_install_3rdparty_licenses(ippicv "${IPPROOT}/readme.htm" "${ICV_PACKAGE_ROOT}/EULA.txt")
endif()

file(TO_CMAKE_PATH "${IPPROOT}" __IPPROOT)
if(EXISTS "${__IPPROOT}/include/ippversion.h")
  set(IPP_ROOT_DIR ${__IPPROOT})
  ipp_detect_version()
endif()


if(WIN32 AND MINGW AND NOT IPP_VERSION_MAJOR LESS 7)
    # Since Intel IPP built with Microsoft compiler and /GS option
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
