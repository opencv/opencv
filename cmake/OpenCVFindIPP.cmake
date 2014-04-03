#
# The script to detect Intel(R) Integrated Performance Primitives (IPP)
# installation/package
#
<<<<<<< HEAD
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
# IPP_FOUND        - True if Intel IPP found
# IPP_ROOT_DIR     - root of IPP installation
# IPP_INCLUDE_DIRS - IPP include folder
# IPP_LIBRARY_DIRS - IPP libraries folder
# IPP_LIBRARIES    - IPP libraries names that are used by OpenCV
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
=======
=======
# IPP_LATEST_VERSION_STR   - string with the newest detected IPP version
# IPP_LATEST_VERSION_MAJOR - numbers of IPP version (MAJOR.MINOR.BUILD)
# IPP_LATEST_VERSION_MINOR
# IPP_LATEST_VERSION_BUILD
#
# Created: 30 Dec 2010 by Vladimir Dudnik (vladimir.dudnik@intel.com)
#
>>>>>>> Undo changes ipp to ippicv prefix of function names

set(IPP_FOUND)
set(IPP_VERSION_STR "5.3.0.0") # will not detect earlier versions
set(IPP_VERSION_MAJOR 0)
set(IPP_VERSION_MINOR 0)
set(IPP_VERSION_BUILD 0)
set(IPP_ROOT_DIR)
set(IPP_INCLUDE_DIRS)
set(IPP_LIBRARY_DIRS)
set(IPP_LIBRARIES)
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
if (CMAKE_CXX_SIZEOF_DATA_PTR EQUAL 8)
    set(IPP_X64 1)
endif()
if (CMAKE_CL_64)
    set(IPP_X64 1)
endif()

# ------------------------------------------------------------------------
# This function detect IPP version by analyzing ippversion.h file
# Note, ippversion.h file was inroduced since IPP 5.3
# ------------------------------------------------------------------------
function(get_ipp_version _ROOT_DIR)
    set(_VERSION_STR)
    set(_MAJOR)
    set(_MINOR)
    set(_BUILD)

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
    set(IPP_VERSION_STR   ${_VERSION_STR} PARENT_SCOPE)
    set(IPP_VERSION_MAJOR ${_MAJOR}       PARENT_SCOPE)
    set(IPP_VERSION_MINOR ${_MINOR}       PARENT_SCOPE)
    set(IPP_VERSION_BUILD ${_BUILD}       PARENT_SCOPE)

    message(STATUS "found IPP: ${_MAJOR}.${_MINOR}.${_BUILD} [${_VERSION_STR}]")
    message(STATUS "at: ${_ROOT_DIR}")

    return()

endfunction()


# ------------------------------------------------------------------------
# This is auxiliary function called from set_ipp_variables()
# to set IPP_LIBRARIES variable in IPP 6.x style (IPP 5.3 should also work)
# ------------------------------------------------------------------------
function(set_ipp_old_libraries)
    set(IPP_PREFIX "ipp")
    set(IPP_SUFFIX)            # old style static core libs suffix
    set(IPP_ARCH)              # architecture suffix
    set(IPP_DISP   "emerged")  # old style dipatcher and cpu-specific
    set(IPP_MRGD   "merged")   #   static libraries
    set(IPPCORE    "core")     # core functionality
    set(IPPSP      "s")        # signal processing
    set(IPPIP      "i")        # image processing
    set(IPPCC      "cc")       # color conversion
    set(IPPCV      "cv")       # computer vision
    set(IPPVM      "vm")       # vector math

    if (IPP_X64)
        set(IPP_ARCH "em64t")
    endif()

    if(WIN32)
        set(IPP_SUFFIX "l")
    endif()

    set(IPP_LIBRARIES
        ${IPP_LIB_PREFIX}${IPP_PREFIX}${IPPVM}${IPP_MRGD}${IPP_ARCH}${IPP_LIB_SUFFIX}
        ${IPP_LIB_PREFIX}${IPP_PREFIX}${IPPVM}${IPP_DISP}${IPP_ARCH}${IPP_LIB_SUFFIX}
        ${IPP_LIB_PREFIX}${IPP_PREFIX}${IPPCC}${IPP_MRGD}${IPP_ARCH}${IPP_LIB_SUFFIX}
        ${IPP_LIB_PREFIX}${IPP_PREFIX}${IPPCC}${IPP_DISP}${IPP_ARCH}${IPP_LIB_SUFFIX}
        ${IPP_LIB_PREFIX}${IPP_PREFIX}${IPPCV}${IPP_MRGD}${IPP_ARCH}${IPP_LIB_SUFFIX}
        ${IPP_LIB_PREFIX}${IPP_PREFIX}${IPPCV}${IPP_DISP}${IPP_ARCH}${IPP_LIB_SUFFIX}
        ${IPP_LIB_PREFIX}${IPP_PREFIX}${IPPIP}${IPP_MRGD}${IPP_ARCH}${IPP_LIB_SUFFIX}
        ${IPP_LIB_PREFIX}${IPP_PREFIX}${IPPIP}${IPP_DISP}${IPP_ARCH}${IPP_LIB_SUFFIX}
        ${IPP_LIB_PREFIX}${IPP_PREFIX}${IPPSP}${IPP_MRGD}${IPP_ARCH}${IPP_LIB_SUFFIX}
        ${IPP_LIB_PREFIX}${IPP_PREFIX}${IPPSP}${IPP_DISP}${IPP_ARCH}${IPP_LIB_SUFFIX}
        ${IPP_LIB_PREFIX}${IPP_PREFIX}${IPPCORE}${IPP_ARCH}${IPP_SUFFIX}${IPP_LIB_SUFFIX}
        PARENT_SCOPE)

    return()

endfunction()


# ------------------------------------------------------------------------
# This is auxiliary function called from set_ipp_variables()
# to set IPP_LIBRARIES variable in IPP 7.x and 8.x style
# ------------------------------------------------------------------------
function(set_ipp_new_libraries _LATEST_VERSION)
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
    set(IPPCORE    "core")     # core functionality
    set(IPPSP      "s")        # signal processing
    set(IPPIP      "i")        # image processing
    set(IPPCC      "cc")       # color conversion
    set(IPPCV      "cv")       # computer vision
    set(IPPVM      "vm")       # vector math

    set(IPP_LIBRARIES
        ${IPP_LIB_PREFIX}${IPP_PREFIX}${IPPVM}${IPP_SUFFIX}${IPP_LIB_SUFFIX}
        ${IPP_LIB_PREFIX}${IPP_PREFIX}${IPPCC}${IPP_SUFFIX}${IPP_LIB_SUFFIX}
        ${IPP_LIB_PREFIX}${IPP_PREFIX}${IPPCV}${IPP_SUFFIX}${IPP_LIB_SUFFIX}
        ${IPP_LIB_PREFIX}${IPP_PREFIX}${IPPI}${IPP_SUFFIX}${IPP_LIB_SUFFIX}
        ${IPP_LIB_PREFIX}${IPP_PREFIX}${IPPS}${IPP_SUFFIX}${IPP_LIB_SUFFIX}
        ${IPP_LIB_PREFIX}${IPP_PREFIX}${IPPCORE}${IPP_SUFFIX}${IPP_LIB_SUFFIX})

    if (UNIX)
        set(IPP_LIBRARIES
            ${IPP_LIBRARIES}
            ${IPP_LIB_PREFIX}irc${CMAKE_SHARED_LIBRARY_SUFFIX}
            ${IPP_LIB_PREFIX}imf${CMAKE_SHARED_LIBRARY_SUFFIX}
            ${IPP_LIB_PREFIX}svml${CMAKE_SHARED_LIBRARY_SUFFIX})
    endif()
    set(IPP_LIBRARIES ${IPP_LIBRARIES} PARENT_SCOPE)
    return()

endfunction()


# ------------------------------------------------------------------------
# This function will set
# IPP_INCLUDE_DIRS, IPP_LIBRARY_DIRS and IPP_LIBRARIES variables depending
# on IPP version parameter.
# Since IPP 7.0 version library names and install folder structure
# was changed
# ------------------------------------------------------------------------
function(set_ipp_variables _LATEST_VERSION)
    if(${_LATEST_VERSION} VERSION_LESS "7.0")
#        message(STATUS "old")

        # set INCLUDE and LIB folders
        set(IPP_INCLUDE_DIRS ${IPP_ROOT_DIR}/include PARENT_SCOPE)
        set(IPP_LIBRARY_DIRS ${IPP_ROOT_DIR}/lib     PARENT_SCOPE)

        if (IPP_X64)
            if(NOT EXISTS ${IPP_ROOT_DIR}/../em64t)
                message(SEND_ERROR "IPP EM64T libraries not found")
            endif()
        else()
            if(NOT EXISTS ${IPP_ROOT_DIR}/../ia32)
                message(SEND_ERROR "IPP IA32 libraries not found")
            endif()
        endif()

<<<<<<< HEAD
# read IPP version info from file
set(_VERSION_STR)
set(_MAJOR)
set(_MINOR)
set(_BUILD)
file(STRINGS ${IPP_INCLUDE_DIR}/ippversion.h STR1 REGEX "IPP_VERSION_MAJOR")
file(STRINGS ${IPP_INCLUDE_DIR}/ippversion.h  STR2 REGEX "IPP_VERSION_MINOR")
file(STRINGS ${IPP_INCLUDE_DIR}/ippversion.h  STR3 REGEX "IPP_VERSION_BUILD")
if("${STR3}" STREQUAL "")
    file(STRINGS ${IPP_INCLUDE_DIR}/ippversion.h STR3 REGEX "IPP_VERSION_UPDATE")
>>>>>>> Prepare codes for ippicv library
=======
        # set IPP_LIBRARIES variable (6.x lib names)
        set_ipp_old_libraries()
        set(IPP_LIBRARIES ${IPP_LIBRARIES} PARENT_SCOPE)
        message(STATUS "IPP libs: ${IPP_LIBRARIES}")

    else()
#        message(STATUS "new")

        # set INCLUDE and LIB folders
        set(IPP_INCLUDE_DIRS ${IPP_ROOT_DIR}/include PARENT_SCOPE)

        if (APPLE)
            set(IPP_LIBRARY_DIRS ${IPP_ROOT_DIR}/lib)
        elseif (IPP_X64)
            if(NOT EXISTS ${IPP_ROOT_DIR}/lib/intel64)
                message(SEND_ERROR "IPP EM64T libraries not found")
            endif()
            set(IPP_LIBRARY_DIRS ${IPP_ROOT_DIR}/lib/intel64)
        else()
            if(NOT EXISTS ${IPP_ROOT_DIR}/lib/ia32)
                message(SEND_ERROR "IPP IA32 libraries not found")
            endif()
            set(IPP_LIBRARY_DIRS ${IPP_ROOT_DIR}/lib/ia32)
        endif()

        if (UNIX)
            get_filename_component(INTEL_COMPILER_LIBRARY_DIR ${IPP_ROOT_DIR}/../lib REALPATH)
            if (IPP_X64)
                if(NOT EXISTS ${INTEL_COMPILER_LIBRARY_DIR}/intel64)
                    message(SEND_ERROR "Intel compiler EM64T libraries not found")
                endif()
                set(IPP_LIBRARY_DIRS
                    ${IPP_LIBRARY_DIRS}
                    ${INTEL_COMPILER_LIBRARY_DIR}/intel64)
            else()
                if(NOT EXISTS ${INTEL_COMPILER_LIBRARY_DIR}/ia32)
                    message(SEND_ERROR "Intel compiler IA32 libraries not found")
                endif()
                set(IPP_LIBRARY_DIRS
                    ${IPP_LIBRARY_DIRS}
                    ${INTEL_COMPILER_LIBRARY_DIR}/ia32)
            endif()
        endif()
        set(IPP_LIBRARY_DIRS ${IPP_LIBRARY_DIRS} PARENT_SCOPE)

        # set IPP_LIBRARIES variable (7.x or 8.x lib names)
        set_ipp_new_libraries(${_LATEST_VERSION})
        set(IPP_LIBRARIES ${IPP_LIBRARIES} PARENT_SCOPE)
        message(STATUS "IPP libs: ${IPP_LIBRARIES}")

    endif()

    return()

endfunction()


# ------------------------------------------------------------------------
# This section will look for IPP through IPPROOT env variable
# Note, IPPROOT is not set by IPP installer, you may need to set it manually
# ------------------------------------------------------------------------
find_path(
    IPP_H_PATH
    NAMES ippversion.h
    PATHS $ENV{IPPROOT}
    PATH_SUFFIXES include
    DOC "The path to Intel(R) IPP header files"
    NO_DEFAULT_PATH
    NO_CMAKE_PATH)

if(IPP_H_PATH)
    set(IPP_FOUND 1)

    # traverse up to IPPROOT level
    get_filename_component(IPP_ROOT_DIR ${IPP_H_PATH} PATH)

    # extract IPP version info
    get_ipp_version(${IPP_ROOT_DIR})

    # keep info in the same vars for auto search and search by IPPROOT
    set(IPP_LATEST_VERSION_STR   ${IPP_VERSION_STR})
    set(IPP_LATEST_VERSION_MAJOR ${IPP_VERSION_MAJOR})
    set(IPP_LATEST_VERSION_MINOR ${IPP_VERSION_MINOR})
    set(IPP_LATEST_VERSION_BUILD ${IPP_VERSION_BUILD})

    # set IPP INCLUDE, LIB dirs and library names
    set_ipp_variables(${IPP_LATEST_VERSION_STR})
>>>>>>> Undo changes ipp to ippicv prefix of function names
endif()


<<<<<<< HEAD
<<<<<<< HEAD
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
=======
# export info to parent scope
set(IPP_VERSION_STR   ${_VERSION_STR})
set(IPP_VERSION_MAJOR ${_MAJOR})
set(IPP_VERSION_MINOR ${_MINOR})
set(IPP_VERSION_BUILD ${_BUILD})

message(STATUS "found IPP: ${_MAJOR}.${_MINOR}.${_BUILD} [${_VERSION_STR}]")
>>>>>>> Prepare codes for ippicv library
=======
if(NOT IPP_FOUND)
    # reset var from previous search
    set(IPP_H_PATH)


    # ------------------------------------------------------------------------
    # This section will look for IPP through system program folders
    # Note, if several IPP installations found the newest version will be
    # selected
    # ------------------------------------------------------------------------
    foreach(curdir ${CMAKE_SYSTEM_PREFIX_PATH})
        set(curdir ${curdir}/intel)
        file(TO_CMAKE_PATH ${curdir} CURDIR)

        if(EXISTS ${curdir})
            file(GLOB_RECURSE IPP_H_DIR ${curdir}/ippversion.h)

            if(IPP_H_DIR)
                set(IPP_FOUND 1)
            endif()

            # init IPP_LATEST_VERSION  version with oldest detectable version (5.3.0.0)
            #   IPP prior 5.3 did not have ippversion.h file
            set(IPP_LATEST_VERSION_STR ${IPP_VERSION_STR})

            # look through all dirs where ippversion.h was found
            foreach(item ${IPP_H_DIR})

                # traverse up to IPPROOT level
                get_filename_component(_FILE_PATH ${item} PATH)
                get_filename_component(_ROOT_DIR ${_FILE_PATH} PATH)

                # extract IPP version info
                get_ipp_version(${_ROOT_DIR})

                # remember the latest version (if many found)
                if(${IPP_LATEST_VERSION_STR} VERSION_LESS ${IPP_VERSION_STR})
                    set(IPP_LATEST_VERSION_STR   ${IPP_VERSION_STR})
                    set(IPP_LATEST_VERSION_MAJOR ${IPP_VERSION_MAJOR})
                    set(IPP_LATEST_VERSION_MINOR ${IPP_VERSION_MINOR})
                    set(IPP_LATEST_VERSION_BUILD ${IPP_VERSION_BUILD})
                    set(IPP_ROOT_DIR ${_ROOT_DIR})
                endif()
            endforeach()
        endif()
    endforeach()
endif()

if(IPP_FOUND)
    # set IPP INCLUDE, LIB dirs and library names
    set_ipp_variables(${IPP_LATEST_VERSION_STR})

    # set CACHE variable IPP_H_PATH,
    # path to IPP header files for the latest version
    find_path(
        IPP_H_PATH
        NAMES ippversion.h
        PATHS ${IPP_ROOT_DIR}
        PATH_SUFFIXES include
        DOC "The path to Intel(R) IPP header files"
        NO_DEFAULT_PATH
        NO_CMAKE_PATH)
endif()

if(WIN32 AND MINGW AND NOT IPP_LATEST_VERSION_MAJOR LESS 7)
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
>>>>>>> Undo changes ipp to ippicv prefix of function names
