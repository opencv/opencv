cmake_minimum_required(VERSION 2.8.9)
if(POLICY CMP0022)
  cmake_policy(SET CMP0022 OLD)
endif()
if(POLICY CMP0025)
  cmake_policy(SET CMP0025 NEW)
endif()
if(POLICY CMP0042)
  cmake_policy(SET CMP0042 NEW)
endif()
project(GDCM)
mark_as_advanced(CMAKE_BACKWARDS_COMPATIBILITY CMAKE_BUILD_TYPE CMAKE_INSTALL_PREFIX)
set(GDCM_CMAKE_DIR "${GDCM_SOURCE_DIR}/CMake" CACHE INTERNAL "")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${GDCM_CMAKE_DIR}")

set(GDCM_PACKAGE_DESCRIPTION_SUMMARY "GDCM - Grassroots DICOM. GDCM is yet another DICOM library.")
set(GDCM_PACKAGE_CONTACT "GDCM Developers <gdcm-developers@lists.sourceforge.net>")

# TODO
# http://www.vtk.org/pipermail/vtkusers/2007-May/090968.html
#
# False memory leak reports are caused by VTK dlls loading *before* MFC dlls.
# You have to use the linker's /delayload flag to avoid this issue.
#
# The /delayload flag should be correct by default in the MFC examples if you
# are using CVS VTK... If you are using a previous version of VTK (5.0 or
# earlier) then you will have to figure out a way to link with that flag.
#
# See the CVS version of files in VTK/GUISupport/MFC for details. Or grep the
# VTK source tree for "DELAYLOAD"

#-----------------------------------------------------------------------------
# Disallow insource build since I never test that
string(COMPARE EQUAL "${GDCM_SOURCE_DIR}" "${GDCM_BINARY_DIR}" INSOURCE)
get_filename_component(PARENTDIR ${GDCM_BINARY_DIR} PATH)
string(COMPARE EQUAL "${GDCM_SOURCE_DIR}" "${PARENTDIR}" INSOURCESUBDIR)
if(INSOURCE OR INSOURCESUBDIR)
  message(FATAL_ERROR "GDCM requires an out of source Build. "
    "Please create a separate binary directory and run CMake there.")
endif()

#----------------------------------------------------------------------------
set(GDCM_MAJOR_VERSION 2)
set(GDCM_MINOR_VERSION 4)
set(GDCM_BUILD_VERSION 4)
set(GDCM_VERSION
  "${GDCM_MAJOR_VERSION}.${GDCM_MINOR_VERSION}.${GDCM_BUILD_VERSION}")
# let advanced user the option to define GDCM_API_VERSION:
if(NOT DEFINED GDCM_API_VERSION)
  set(GDCM_API_VERSION "${GDCM_MAJOR_VERSION}.${GDCM_MINOR_VERSION}")
endif()
set(GDCM_LIBRARY_PROPERTIES ${GDCM_LIBRARY_PROPERTIES}
  VERSION "${GDCM_VERSION}"
  SOVERSION "${GDCM_API_VERSION}"
)
#set(GDCM_EXECUTABLE_PROPERTIES ${GDCM_EXECUTABLE_PROPERTIES}
#  VERSION "${GDCM_MAJOR_VERSION}.${GDCM_MINOR_VERSION}"
#)
set(GDCM_EXECUTABLE_PROPERTIES)
if(GDCM_NO_EXECUTABLE_PROPERTIES)
  set(GDCM_EXECUTABLE_PROPERTIES)
  set(python_site_package "python/dist-packages")
else()
  set(python_site_package "")
endif()

set(GDCM_STANDALONE 0)
# Top level project (eg. ITK) should set GDCM_TARGETS_NAME
# to define a particular behavior where GDCM does not call
# install(EXPORT...)
# This sets the default value for GDCM_STANDALONE
if(NOT GDCM_TARGETS_NAME)
  set(GDCM_TARGETS_NAME GDCMTargets)
  set(GDCM_STANDALONE 1)
endif()
#-----------------------------------------------------------------------------
# PDB handling + Module handling
include(${GDCM_SOURCE_DIR}/CMake/InstallMacros.cmake)
#-----------------------------------------------------------------------------
include(${GDCM_SOURCE_DIR}/CMake/UseCopyright.cmake)
CREATE_COPYRIGHT_FILE(${CMAKE_CURRENT_BINARY_DIR}/Copyright.txt)
APPEND_COPYRIGHT(${CMAKE_CURRENT_SOURCE_DIR}/Copyright.txt)
APPEND_COPYRIGHT(${CMAKE_CURRENT_SOURCE_DIR}/CMake/COPYING-CMAKE-SCRIPTS)

#-----------------------------------------------------------------------------
if(GDCM_MINOR_VERSION MATCHES "[02468]$")
  # Are we building a release branch / tag (read: even number)?
  # By default dashboard are expected to run with Design by Contract on
  # to trigger any of the assert, but on the other hand no user really
  # can figure out they need to change this value
  # So unless the user *specifically* requested a particular cmake_build_type
  # do the work internally and append the NDEBUG def flag (hopefully portable)
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG")
  endif()
  # Since we are on a release branch, chance is that people don't care about testing
  # let's disable it for them
  set(GDCM_BUILD_TESTING_DEFAULT OFF)
else()
  set(GDCM_BUILD_TESTING_DEFAULT ON)
endif()

#-----------------------------------------------------------------------------
# Disable deprecation warnings for standard C and STL functions in VS2005
# and later
if(MSVC_VERSION EQUAL 1400 OR MSVC_VERSION GREATER 1400)
  add_definitions(-D_CRT_SECURE_NO_DEPRECATE -D_CRT_NONSTDC_NO_DEPRECATE -D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-D_SCL_SECURE_NO_DEPRECATE -D_SCL_SECURE_NO_WARNINGS)
endif()

#-----------------------------------------------------------------------------
# Build shared lib by default
if(GDCM_STANDALONE)
  option(GDCM_BUILD_SHARED_LIBS "Build GDCM with shared libraries." OFF)
  set(BUILD_SHARED_LIBS ${GDCM_BUILD_SHARED_LIBS})
else()
  set(GDCM_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
endif()

if(BUILD_SHARED_LIBS)
if(${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} GREATER 2.6)
  set(NAMELINK_ONLY NAMELINK_ONLY)
  set(NAMELINK_SKIP NAMELINK_SKIP)
else()
  set(NAMELINK_ONLY)
  set(NAMELINK_SKIP)
endif()
endif()


#-----------------------------------------------------------------------------
if(NOT EXECUTABLE_OUTPUT_PATH)
  set(EXECUTABLE_OUTPUT_PATH ${GDCM_BINARY_DIR}/bin CACHE PATH "Single output directory for building all executables.")
  mark_as_advanced(EXECUTABLE_OUTPUT_PATH)
endif()
if(NOT LIBRARY_OUTPUT_PATH)
  set(LIBRARY_OUTPUT_PATH ${GDCM_BINARY_DIR}/bin CACHE PATH "Single output directory for building all libraries.")
  mark_as_advanced(LIBRARY_OUTPUT_PATH)
endif()

# Use dependent options.
INCLUDE(CMakeDependentOption)

#-----------------------------------------------------------------------------
# Adding GDCM_DATA_ROOT
if(GDCM_STANDALONE)
  find_path(GDCM_DATA_ROOT test.acr
    ${GDCM_SOURCE_DIR}/Testing/Data
    $ENV{GDCM_DATA_ROOT}
    NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH
    )
  mark_as_advanced(GDCM_DATA_ROOT)
# You can define a path where you extra the famous D. Clunie spacing dataset
# http://www.dclunie.com/images/pixelspacingtestimages.zip
# for example:
# find $HOME/pixelspacingtestimages
# $HOME/pixelspacingtestimages
# $HOME/pixelspacingtestimages/DISCIMG
# $HOME/pixelspacingtestimages/DISCIMG/DICOMDIR
# $HOME/pixelspacingtestimages/DISCIMG/IMAGES
# $HOME/pixelspacingtestimages/DISCIMG/IMAGES/DXIMAGE
# $HOME/pixelspacingtestimages/DISCIMG/IMAGES/MGIMAGE
# $HOME/pixelspacingtestimages/DISCIMG/IMAGES/CRIMAGE
  mark_as_advanced(GDCM_PIXEL_SPACING_DATA_ROOT)

#-----------------------------------------------------------------------------
  find_path(GDCM_DATA_EXTRA_ROOT gdcmData.tar.gz
    ${GDCM_SOURCE_DIR}/../gdcmDataExtra
    $ENV{GDCM_DATA_EXTRA_ROOT}
    $ENV{PUB_DICT_PATH}/../../gdcmDataExtra
    NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH
    )
  mark_as_advanced(GDCM_DATA_EXTRA_ROOT)
endif()

# Define a temp directory in which we can output stuff
set(GDCM_TEMP_DIRECTORY "${GDCM_BINARY_DIR}/Testing/Temporary" CACHE PATH "Path to a valid temp directory")
mark_as_advanced(GDCM_TEMP_DIRECTORY)

#-----------------------------------------------------------------------------
include(${CMAKE_ROOT}/Modules/CheckIncludeFile.cmake)
include(${CMAKE_ROOT}/Modules/CheckIncludeFiles.cmake)
# Check if header file exists and add it to the list.
macro(CHECK_INCLUDE_FILE_CONCAT FILE VARIABLE)
  CHECK_INCLUDE_FILES("${UUID_INCLUDES};${FILE}" ${VARIABLE})
  if(${VARIABLE})
    set(UUID_INCLUDES ${UUID_INCLUDES} ${FILE})
  endif()
endmacro()

CHECK_INCLUDE_FILE("stdint.h"       GDCM_HAVE_STDINT_H)
if(UNIX) #Avoid polluting Win32 cmakecache
  CHECK_INCLUDE_FILE("inttypes.h"     GDCM_HAVE_INTTYPES_H)
endif()

#include(${GDCM_SOURCE_DIR}/CMake/gdcmPlatformCxxTests.cmake)
#
#GDCM_PLATFORM_CXX_TEST(GDCM_CXX_HAS_FUNCTION
#    "Checking whether compiler has __FUNCTION__" DIRECT)


#-----------------------------------------------------------------------------
# Build the main lib...
if(NOT GDCM_HAVE_STDINT_H)
  if(MSVC)
    include_directories(
      "${GDCM_SOURCE_DIR}/Utilities/C99"
      )
    # Process the install rules from C99
    subdirs(Utilities/C99)
  endif()
endif()

# --------------------------------------------------------------------------
# Configure the export configuration

# You will also need to define a value for the following variables:
# GDCM_INSTALL_BIN_DIR          - binary dir (executables)
# GDCM_INSTALL_LIB_DIR          - library dir (libs)
# GDCM_INSTALL_DATA_DIR         - share dir (say, examples, data, etc)
# GDCM_INSTALL_INCLUDE_DIR      - include dir (headers)
# GDCM_INSTALL_PACKAGE_DIR      - package/export configuration files
# GDCM_VTK_INSTALL_PACKAGE_DIR  - VTK package/export configuration files
# GDCM_INSTALL_NO_DEVELOPMENT   - do not install development files
# GDCM_INSTALL_NO_RUNTIME       - do not install runtime files
# GDCM_INSTALL_NO_DOCUMENTATION - do not install documentation files

# GDCM_INSTALL_PYTHONMODULE_DIR     - Python Module install dir
# GDCM_VTK_INSTALL_PYTHONMODULE_DIR - VTK/Python Module install dir
# GDCM_INSTALL_CSHARPMODULE_DIR     - C# Module install dir
# GDCM_VTK_INSTALL_CSHARPMODULE_DIR - VTK/C# Module install dir
# GDCM_INSTALL_JAVAMODULE_DIR       - Java Module install dir (JNI glue)
# GDCM_INSTALL_JARMODULE_DIR        - Java Module install dir (JAR)
# GDCM_VTK_INSTALL_JAVAMODULE_DIR   - VTK/Java Module install dir (JNI glue)
# GDCM_VTK_INSTALL_JARMODULE_DIR    - VTK/Java Module install dir (JAR)

# --------------------------------------------------------------------------
# Install directories

string(TOLOWER ${PROJECT_NAME} projectname)
set(subdir "${projectname}-${GDCM_MAJOR_VERSION}.${GDCM_MINOR_VERSION}")

if(NOT GDCM_INSTALL_BIN_DIR)
  set(GDCM_INSTALL_BIN_DIR "bin")
endif()

if(NOT GDCM_INSTALL_LIB_DIR)
  #set(GDCM_INSTALL_LIB_DIR "lib/${PROJECT_NAME}")
  set(GDCM_INSTALL_LIB_DIR "lib")
endif()

if(NOT GDCM_INSTALL_DATA_DIR)
  set(GDCM_INSTALL_DATA_DIR "share/${subdir}")
endif()

if(NOT GDCM_INSTALL_INCLUDE_DIR)
  set(GDCM_INSTALL_INCLUDE_DIR "include/${subdir}")
endif()

if(NOT GDCM_INSTALL_DOC_DIR)
  set(GDCM_INSTALL_DOC_DIR "share/doc/${subdir}")
endif()

if(NOT GDCM_INSTALL_MAN_DIR)
  set(GDCM_INSTALL_MAN_DIR "share/man")
endif()

if(NOT GDCM_INSTALL_PACKAGE_DIR)
  set(GDCM_INSTALL_PACKAGE_DIR ${GDCM_INSTALL_LIB_DIR}/${subdir}
    CACHE INTERNAL "")
endif()

if(NOT GDCM_VTK_INSTALL_PACKAGE_DIR)
  set(GDCM_VTK_INSTALL_PACKAGE_DIR ${VTK_INSTALL_PACKAGE_DIR})
endif()

if(NOT GDCM_INSTALL_NO_DEVELOPMENT)
  set(GDCM_INSTALL_NO_DEVELOPMENT 0)
endif()

if(NOT GDCM_INSTALL_NO_RUNTIME)
  set(GDCM_INSTALL_NO_RUNTIME 0)
endif()

if(NOT GDCM_INSTALL_NO_DOCUMENTATION)
  set(GDCM_INSTALL_NO_DOCUMENTATION 0)
endif()

set(GDCM_INSTALL_NO_LIBRARIES)
if(GDCM_BUILD_SHARED_LIBS)
  if(GDCM_INSTALL_NO_RUNTIME AND GDCM_INSTALL_NO_DEVELOPMENT)
    set(GDCM_INSTALL_NO_LIBRARIES 1)
  endif()
else()
  if(GDCM_INSTALL_NO_DEVELOPMENT)
    set(GDCM_INSTALL_NO_LIBRARIES 1)
  endif()
endif()

if(NOT GDCM_INSTALL_PYTHONMODULE_DIR)
  set(GDCM_INSTALL_PYTHONMODULE_DIR ${GDCM_INSTALL_LIB_DIR}/${python_site_package})
endif()
if(NOT GDCM_VTK_INSTALL_PYTHONMODULE_DIR)
  set(GDCM_VTK_INSTALL_PYTHONMODULE_DIR ${GDCM_INSTALL_PYTHONMODULE_DIR})
endif()
if(NOT GDCM_INSTALL_CSHARPMODULE_DIR)
  set(GDCM_INSTALL_CSHARPMODULE_DIR ${GDCM_INSTALL_LIB_DIR})
endif()
if(NOT GDCM_VTK_INSTALL_CSHARPMODULE_DIR)
  set(GDCM_VTK_INSTALL_CSHARPMODULE_DIR ${GDCM_INSTALL_CSHARPMODULE_DIR})
endif()
if(NOT GDCM_INSTALL_JAVAMODULE_DIR)
  set(GDCM_INSTALL_JAVAMODULE_DIR ${GDCM_INSTALL_LIB_DIR})
endif()
if(NOT GDCM_VTK_INSTALL_JAVAMODULE_DIR)
  set(GDCM_VTK_INSTALL_JAVAMODULE_DIR ${GDCM_INSTALL_JAVAMODULE_DIR})
endif()
if(NOT GDCM_INSTALL_JARMODULE_DIR)
  set(GDCM_INSTALL_JARMODULE_DIR ${GDCM_INSTALL_LIB_DIR})
endif()
if(NOT GDCM_VTK_INSTALL_JARMODULE_DIR)
  set(GDCM_VTK_INSTALL_JARMODULE_DIR ${GDCM_INSTALL_JARMODULE_DIR})
endif()

#-----------------------------------------------------------------------------
#System stuff, mainly for packager or paranoid people with up-to-date lib moto
option(GDCM_USE_SYSTEM_ZLIB "Use system zlib" OFF)
option(GDCM_USE_SYSTEM_OPENSSL  "Use system OpenSSL" OFF)
if(UNIX)
  # Will search for the uuid_generate symbols.
  # Can be in libSystem.dylib or libuuid.so
  option(GDCM_USE_SYSTEM_UUID "Use system uuid" OFF)
endif()
option(GDCM_USE_SYSTEM_EXPAT "Use system expat" OFF)
option(GDCM_USE_SYSTEM_JSON "Use system json" OFF)
option(GDCM_USE_SYSTEM_PAPYRUS3 "Use system papyrus3" OFF)
option(GDCM_USE_SYSTEM_SOCKETXX "Use system socket++" OFF)
option(GDCM_USE_SYSTEM_LJPEG "Use system ljpeg (ijg lib)" OFF)
option(GDCM_USE_SYSTEM_OPENJPEG "Use system openjpeg" OFF)
option(GDCM_USE_SYSTEM_CHARLS "Use system charls" OFF)
mark_as_advanced(
  GDCM_USE_SYSTEM_ZLIB
  GDCM_USE_SYSTEM_OPENSSL
  GDCM_USE_SYSTEM_UUID
  GDCM_USE_SYSTEM_EXPAT
  GDCM_USE_SYSTEM_JSON
  GDCM_USE_SYSTEM_PAPYRUS3
  GDCM_USE_SYSTEM_SOCKETXX
  GDCM_USE_SYSTEM_LJPEG
  GDCM_USE_SYSTEM_OPENJPEG
  GDCM_USE_SYSTEM_CHARLS
  )
option(GDCM_USE_SYSTEM_POPPLER "Use system poppler (pdf)" OFF)
if(GDCM_USE_SYSTEM_POPPLER)
  find_package(Poppler REQUIRED)
endif()
mark_as_advanced(GDCM_USE_SYSTEM_POPPLER)

option(GDCM_USE_SYSTEM_LIBXML2 "Use LibXml2" OFF)
if(GDCM_USE_SYSTEM_LIBXML2)
  find_package(LibXml2)
endif()
mark_as_advanced(GDCM_USE_SYSTEM_LIBXML2)

if(GDCM_USE_SYSTEM_LJPEG)
  find_package(LJPEG REQUIRED)
  set(GDCM_LJPEG_LIBRARIES ${LJPEG_LIBRARIES})
else()
  set(GDCM_LJPEG_LIBRARIES gdcmjpeg8 gdcmjpeg12 gdcmjpeg16)
endif()

if(GDCM_USE_SYSTEM_CHARLS)
  find_package(CharLS REQUIRED)
  set(GDCM_CHARLS_LIBRARIES ${CHARLS_LIBRARIES})
else()
  set(GDCM_CHARLS_LIBRARIES gdcmcharls)
endif()

if(GDCM_USE_SYSTEM_OPENJPEG)
  # For some reason I cannot specify the version I want.
  # find_package(OpenJPEG 2.0 REQUIRED)
  # oh well we should handle both 1.0 and 2.0 anyway...
  find_package(OpenJPEG REQUIRED)
  set(GDCM_OPENJPEG_LIBRARIES ${OPENJPEG_LIBRARIES})
else()
  set(GDCM_OPENJPEG_LIBRARIES gdcmopenjpeg)
  option(GDCM_USE_OPENJPEG "Use OpenJGPEG library." OFF)
  cmake_dependent_option(GDCM_USE_OPENJPEG_V2 "Use openjpeg v2 version (advanced users only)." OFF "GDCM_USE_OPENJPEG" OFF)  
  mark_as_advanced(GDCM_USE_OPENJPEG GDCM_USE_OPENJPEG_V2)
  if(GDCM_USE_OPENJPEG)
	add_definitions(-DGDCM_USE_OPENJPEG)
  endif()
endif()

option(GDCM_USE_MEXD_MODULE "Use message exchange definition module" OFF)

# Very advanced user option:
# This will cause building of the broken JPEG library released by the Standford PVRG group:
option(GDCM_USE_PVRG "Use pvrg lib, only turn it on if you know what you are doing." OFF)
mark_as_advanced(GDCM_USE_PVRG)

option(GDCM_USE_KAKADU "Use kakadu lib, only turn it on if you know what you are doing." OFF)
mark_as_advanced(GDCM_USE_KAKADU)

if(GDCM_USE_PVRG)
  option(GDCM_USE_SYSTEM_PVRG "Use system PVRG" OFF)
  mark_as_advanced(GDCM_USE_SYSTEM_PVRG)
  if(GDCM_USE_SYSTEM_PVRG)
    find_package(PVRGJPEG REQUIRED)
  endif()
endif()

# Very advanced behavior only use if you want to keep backward compatible
# behavior but possibly incorrect behavior.
option(GDCMV2_0_COMPATIBILITY "Preserve compatibility with GDCM 2.0 release" ON)
mark_as_advanced(GDCMV2_0_COMPATIBILITY)

if(GDCM_USE_KAKADU)
  option(GDCM_USE_SYSTEM_KAKADU "Use system KAKADU " ON)
  mark_as_advanced(GDCM_USE_SYSTEM_KAKADU)
  if(GDCM_USE_SYSTEM_KAKADU)
    find_package(KAKADU REQUIRED)
  else()
    message(FATAL_ERROR "Not Implemented")
  endif()
endif()

if(GDCM_USE_SYSTEM_ZLIB)
  # If user say so, then this is a requirement !
  find_package(ZLIB REQUIRED)
  set(GDCM_ZLIB_LIBRARIES ${ZLIB_LIBRARIES})
else()
  set(GDCM_ZLIB_LIBRARIES "gdcmzlib")
endif()

if(GDCM_USE_SYSTEM_OPENSSL)
  find_package(OpenSSL REQUIRED)
endif()

if(GDCM_USE_SYSTEM_UUID)
  # If user say so, then this is a requirement !
  find_package(UUID REQUIRED)
  set(GDCM_UUID_LIBRARIES ${UUID_LIBRARIES})
else()
  set(GDCM_UUID_LIBRARIES "gdcmuuid")
endif()

if(GDCM_USE_SYSTEM_EXPAT)
  # If user say so, then this is a requirement !
  find_package(EXPAT REQUIRED)
  set(GDCM_EXPAT_LIBRARIES ${EXPAT_LIBRARIES})
else()
  set(GDCM_EXPAT_LIBRARIES "gdcmexpat")
endif()

if(GDCM_USE_SYSTEM_JSON)
  find_package(JSON REQUIRED)
endif()
if(GDCM_USE_SYSTEM_PAPYRUS3)
  find_package(PAPYRUS3 REQUIRED)
endif()

#-----------------------------------------------------------------------------
if(GDCM_STANDALONE)
  option(GDCM_BUILD_EXAMPLES "Build GDCM examples." OFF)
  set(BUILD_EXAMPLES ${GDCM_BUILD_EXAMPLES})
else()
  set(GDCM_BUILD_EXAMPLES OFF)
  set(BUILD_EXAMPLES OFF)
endif()
#-----------------------------------------------------------------------------
# Add the testing directories
if(GDCM_STANDALONE)
  option(GDCM_BUILD_TESTING "Build testing." ${GDCM_BUILD_TESTING_DEFAULT})
  set(BUILD_TESTING ${GDCM_BUILD_TESTING}) # CACHE BOOL "" FORCE)
else()
  set(GDCM_BUILD_TESTING OFF)
  set(BUILD_TESTING OFF)
endif()
# Hide BUILD_TESTING as user tend to always turn all options on and then complains when something
# does not work 'by default'
mark_as_advanced(BUILD_TESTING) # GDCM_BUILD_TESTING)

#-----------------------------------------------------------------------------
if(GDCM_TESTING_USE_LC_NUMERIC)
  set(CMAKE_TESTDRIVER_BEFORE_TESTMAIN
    "setlocale(LC_ALL,\"fr_FR.UTF-8\");std::locale::global(std::locale(\"fr_FR.UTF-8\"));"
    )
endif()

#-----------------------------------------------------------------------------
# Python install
#find_package(PythonInterp REQUIRED)
#macro(GET_PYTHON_SITE_PACKAGE dir)
#  execute_process(
#  COMMAND ${PYTHON_EXECUTABLE} "-c" "from distutils import sysconfig; print sysconfig.get_python_lib()"
#  #WORKING_DIRECTORY @LIBRARY_OUTPUT_PATH@
#  RESULT_VARIABLE import_res
#  OUTPUT_VARIABLE import_output
#  ERROR_VARIABLE  import_error
#    OUTPUT_STRIP_TRAILING_WHITESPACE
#  )
#  #set(dir ${import_output})
#endmacro()
#
#
#GET_PYTHON_SITE_PACKAGE(python_site_package)
#if(import_output)
#string(LENGTH ${import_output} len)
## let's remove the "/usr/lib" part...
#math(EXPR fileend "${len} - 9")
#string(SUBSTRING ${import_output} 9 ${fileend} dummy1)
#if(UNIX)
#  set(python_site_package ${dummy1})
#  message(${python_site_package})
#endif()
#endif()

# On unix one have: "python2.4/site-packages"
# while on Win32: "c:/Python24/Lib/site-packages/"
# give up for now and place python modules stuff in a general 'python' subdir

# Typical runtime env should be then
# (assuming CMAKE_INSTALL_PREFIX:PATH=/tmp/local)
# One would do:
#   export PYTHONPATH=/tmp/local/lib/python/site-packages/gdcm-2.1
#   export LD_LIBRARY_PATH=/tmp/local/lib/
#   python
#   > import gdcm


#-----------------------------------------------------------------------------
# Wrapping
if(GDCM_STANDALONE)
  option(GDCM_WRAP_PYTHON "build python wrapping" OFF)
  option(GDCM_WRAP_PERL "build perl wrapping (experimental !)" OFF)
  option(GDCM_WRAP_PHP "build php wrapping" OFF)
  option(GDCM_WRAP_JAVA "build java wrapping" OFF)
  option(GDCM_WRAP_CSHARP "build csharp wrapping" OFF)
  mark_as_advanced(GDCM_WRAP_PERL)
  mark_as_advanced(GDCM_USE_RLE)
  mark_as_advanced(GDCM_USE_ACTIVIZ)
  option(GDCM_USE_JPEGLS "Build GDCM with JPEG-LS support" ON)
  mark_as_advanced(GDCM_USE_JPEGLS)
  if(GDCM_WRAP_CSHARP)
    find_package(CSharp REQUIRED)
    include(${CSharp_USE_FILE})
  endif()
endif()

set(GDCM_LIBRARY_DIR ${LIBRARY_OUTPUT_PATH}/${CMAKE_CFG_INTDIR})
set(GDCM_EXECUTABLE_DIR ${EXECUTABLE_OUTPUT_PATH}/${CMAKE_CFG_INTDIR})

#-----------------------------------------------------------------------------
# we need to link against CoreFoundation so that we can use CFBundle to get the executable path.
if(APPLE)
  find_library(COREFOUNDATION_LIBRARY CoreFoundation )
endif()

#-----------------------------------------------------------------------------
# Allow user to set a postfix symbol to a target library name (eg. 'd')
if(WIN32)
  set(GDCM_DEBUG_POSTFIX "" CACHE STRING "Globally append a debug postfix symbols on all libraries")
  if(GDCM_DEBUG_POSTFIX)
    set(CMAKE_DEBUG_POSTFIX "${GDCM_DEBUG_POSTFIX}")
  endif()
  mark_as_advanced(GDCM_DEBUG_POSTFIX)
endif()
#-----------------------------------------------------------------------------

# Need to subdirs in Source/Common before Wrapping
# to have gdcmConfigure.h around
subdirs(Utilities)
add_subdirectory(Source)

if(GDCM_STANDALONE)
  subdirs(Wrapping)
  if(GDCM_WRAP_CSHARP)
    add_subdirectory(Wrapping/Csharp)
  endif()
endif()

if(GDCM_STANDALONE)
  # After Wrapping please
  if(BUILD_EXAMPLES)
    subdirs(Examples)
  endif()
endif()

#-----------------------------------------------------------------------------
# Special CMake Module required when doing Python Testing
if(GDCM_STANDALONE)
  if(BUILD_TESTING AND GDCM_WRAP_PYTHON)
    include(${GDCM_SOURCE_DIR}/CMake/UsePythonTest.cmake)
  endif()

# Special CMake Module required when doing C# Testing
  if(BUILD_TESTING AND GDCM_WRAP_CSHARP)
    include(${GDCM_SOURCE_DIR}/CMake/UseCSharpTest.cmake)
  endif()

# Special CMake Module required when doing Java Testing
  if(BUILD_TESTING AND GDCM_WRAP_JAVA)
    include(${GDCM_SOURCE_DIR}/CMake/UseJavaTest.cmake)
  endif()
endif()
#-----------------------------------------------------------------------------
# Need pthread for the following class:
CHECK_INCLUDE_FILE("pthread.h"      GDCM_HAVE_PTHREAD_H)

# Big endian thing:
if(GDCM_STANDALONE)
  include(${CMAKE_ROOT}/Modules/TestBigEndian.cmake)
  TEST_BIG_ENDIAN(GDCM_WORDS_BIGENDIAN)
endif()

if(GDCM_STANDALONE)
  if(BUILD_TESTING)
    configure_file(${GDCM_SOURCE_DIR}/CMake/CTestCustom.ctest.in
        ${GDCM_BINARY_DIR}/CMake/CTestCustom.ctest @ONLY)
    file(WRITE ${GDCM_BINARY_DIR}/CTestCustom.cmake
      "include(\"${GDCM_BINARY_DIR}/CMake/CTestCustom.ctest\")\n")
    mark_as_advanced(DART_TESTING_TIMEOUT)
    enable_testing()
    include(CTest)
    subdirs(Testing)
    if(NOT GDCM_DATA_ROOT)
      message("If you want to build the test suite, you must set GDCM_DATA_ROOT (advanced option) "
      "to the full path name of the gdcmData directory; if you don't want, disable GDCM_BUILD_TESTING.\n"
      "What is gdcmData? Please read: http://gdcm.sourceforge.net/wiki/index.php/General_questions#What_is_gdcmData_.3F")
    endif()
  endif()
endif()

#-----------------------------------------------------------------------------
if(GDCM_STANDALONE)
  option(GDCM_DOCUMENTATION "Build source documentation using doxygen." OFF)
  if(GDCM_DOCUMENTATION)
    option(GDCM_PDF_DOCUMENTATION "Build source doxygen using doxygen as pdf" ON)
    mark_as_advanced(GDCM_PDF_DOCUMENTATION)
  endif()
endif()

#-----------------------------------------------------------------------------
if(GDCM_STANDALONE)
  option(GDCM_USE_VTK "vtk bridge ?" OFF)
  if(GDCM_USE_VTK AND GDCM_WRAP_CSHARP)
    option(GDCM_USE_ACTIVIZ "vtk/Activiz bridge ?" OFF)
  endif()

  if(GDCM_USE_VTK)
    option(GDCM_USE_PARAVIEW "paraview plugin ?" OFF)
    # needed here so that we have VTK_WRAP_PYTHON and al. available
    find_package(VTK REQUIRED)
    mark_as_advanced(VTK_DIR)
    mark_as_advanced(GDCM_USE_PARAVIEW)
    subdirs(Utilities/VTK)
  endif()
endif()
#-----------------------------------------------------------------------------
option(GDCM_USE_WXWIDGETS "wxWidgets bridge ?" OFF)
mark_as_advanced(GDCM_USE_WXWIDGETS)
if(GDCM_USE_WXWIDGETS)
  subdirs(Utilities/wxWidgets)
endif()

#-----------------------------------------------------------------------------
if(GDCM_STANDALONE)
  option(GDCM_BUILD_APPLICATIONS "apps ?" OFF)
  set(BUILD_APPLICATIONS ${GDCM_BUILD_APPLICATIONS})
  if(BUILD_APPLICATIONS)
    subdirs(Applications)
  endif()
else()
  set(BUILD_APPLICATIONS OFF)
  set(GDCM_BUILD_APPLICATIONS OFF)
endif()
#-----------------------------------------------------------------------------
if(GDCM_USE_VTK)
if(WIN32)
include(CMake/InstallRequiredVTKLibraries.cmake)
endif()
endif()

#-----------------------------------------------------------------------------
# CPack stuff
if(GDCM_STANDALONE) # disabled for ITK distribution of gdcm
  if(EXISTS "${CMAKE_ROOT}/Modules/CPack.cmake")
  if(EXISTS "${CMAKE_ROOT}/Modules/InstallRequiredSystemLibraries.cmake")
    set(CMAKE_INSTALL_MFC_LIBRARIES 0)
    set(CMAKE_INSTALL_DEBUG_LIBRARIES 0)
    if(NOT DEFINED CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS_NO_WARNINGS)
      set(CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS_NO_WARNINGS ON)
    endif()
    include(${CMAKE_ROOT}/Modules/InstallRequiredSystemLibraries.cmake)
  endif()

  set(CPACK_PACKAGE_DESCRIPTION_SUMMARY ${GDCM_PACKAGE_DESCRIPTION_SUMMARY})
  set(CPACK_PACKAGE_VENDOR "GDCM")
  set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_BINARY_DIR}/Copyright.txt")
  set(CPACK_RESOURCE_FILE_LICENSE    "${CMAKE_CURRENT_BINARY_DIR}/Copyright.txt")
  set(CPACK_PACKAGE_VERSION_MAJOR "${GDCM_MAJOR_VERSION}")
  set(CPACK_PACKAGE_VERSION_MINOR "${GDCM_MINOR_VERSION}")
  set(CPACK_PACKAGE_VERSION_PATCH "${GDCM_BUILD_VERSION}")
  set(CPACK_PACKAGE_INSTALL_DIRECTORY "GDCM ${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}")
  set(CPACK_SOURCE_PACKAGE_FILE_NAME "gdcm-${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}")

  if(WIN32 AND NOT UNIX)
    # There is a bug in NSIS that does not handle full UNIX paths properly. Make
    # sure there is at least one set of four (4) backlashes.
    set(CPACK_NSIS_INSTALLED_ICON_NAME "bin\\\\gdcmviewer.exe")
    set(CPACK_NSIS_DISPLAY_NAME "${CPACK_PACKAGE_INSTALL_DIRECTORY}")
    set(CPACK_NSIS_HELP_LINK "http://gdcm.sourceforge.net")
    set(CPACK_NSIS_URL_INFO_ABOUT "http://gdcm.sourceforge.net")
    set(CPACK_NSIS_MODIFY_PATH ON)
    # if(${CMAKE_INSTALL_PREFIX} MATCHES ${CPACK_PACKAGE_INSTALL_DIRECTORY})
    # else()
    # string(REPLACE ${PROJECT_NAME} ${CPACK_PACKAGE_INSTALL_DIRECTORY} dummy ${CMAKE_INSTALL_PREFIX} )
    # set(CMAKE_INSTALL_PREFIX ${dummy} CACHE PATH "" FORCE)
    # endif()

  else()
    set(CPACK_STRIP_FILES TRUE)
    set(CPACK_SOURCE_STRIP_FILES "")
    set(CPACK_PACKAGE_EXECUTABLES "gdcmviewer" "VIEWER")
  endif()

  # cygwin stff also copied from cmake
  #if(NOT DEFINED CPACK_PACKAGE_FILE_NAME)
  #  # if the CPACK_PACKAGE_FILE_NAME is not defined by the cache
  #  # default to source package - system, on cygwin system is not
  #  # needed
  #  if(CYGWIN)
  #    set(CPACK_PACKAGE_FILE_NAME "${CPACK_SOURCE_PACKAGE_FILE_NAME}")
  #  else()
  #    set(CPACK_PACKAGE_FILE_NAME
  #      "${CPACK_SOURCE_PACKAGE_FILE_NAME}-${CPACK_SYSTEM_NAME}")
  #  endif()
  #endif()

  #Cygwin stuff copied from cmake
  if(NOT DEFINED CPACK_SYSTEM_NAME)
    # make sure package is not Cygwin-unknown, for Cygwin just
    # cygwin is good for the system name
    if("${CMAKE_SYSTEM_NAME}" STREQUAL "CYGWIN")
      set(CPACK_SYSTEM_NAME Cygwin)
    else()
      #set(CMAKE_SYSTEM_PROCESSOR "x86_64")
      #set(CPACK_GENERATOR "TGZ;TBZ2")
      #set(CPACK_SOURCE_GENERATOR "TGZ;TBZ2")
      set(CPACK_SYSTEM_NAME ${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR})
    endif()
  endif()

  # Need to set the architecture for debian package
  set(CPACK_PACKAGE_CONTACT ${GDCM_PACKAGE_CONTACT})
  set(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6, libstdc++6 (>= 4.0.2-4), libuuid1, zlib1g (>= 1:1.2.1), libgcc1 (>= 1:4.0.2), libexpat1, swig") # bug: missing dep to python...
  set(CPACK_DEBIAN_PACKAGE_SUGGESTS "dcmtk")
  set(CPACK_SOURCE_IGNORE_FILES "/\\\\.gitmodules" "/\\\\.git/" "/\\\\.gitignore" "TODO" "/Testing/Data/")
  set(CPACK_IGNORE_FILES ${CPACK_SOURCE_IGNORE_FILES})

  # List executables
  #set(CPACK_PACKAGE_EXECUTABLES "gdcmviewer" "VIEWER")
  if(CYGWIN)
    set(CPACK_CYGWIN_PATCH_NUMBER 1)
    set(CPACK_CYGWIN_BUILD_SCRIPT
    "${GDCM_BINARY_DIR}/@CPACK_PACKAGE_FILE_NAME@-@CPACK_CYGWIN_PATCH_NUMBER@.sh")
    set(CPACK_CYGWIN_PATCH_FILE
  "${GDCM_BINARY_DIR}/@CPACK_PACKAGE_FILE_NAME@-@CPACK_CYGWIN_PATCH_NUMBER@.patch")
    configure_file(${GDCM_SOURCE_DIR}/CMake/Release/cygwin-patch.diff.in ${CPACK_CYGWIN_PATCH_FILE})
    configure_file(${GDCM_SOURCE_DIR}/CMake/Release/cygwin-package.sh.in ${CPACK_CYGWIN_BUILD_SCRIPT})
  endif()

  # Tell CPack all of the components to install. The "ALL"
  # refers to the fact that this is the set of components that
  # will be included when CPack is instructed to put everything
  # into the binary installer (the default behavior).
  set(components)
  if(GDCM_BUILD_APPLICATIONS)
    list(APPEND components Applications)
  endif()
  list(APPEND components Libraries)
  list(APPEND components Headers)
  list(APPEND components DebugDevel)
  if(GDCM_WRAP_PYTHON)
    list(APPEND components PythonModule)
  endif()
  if(GDCM_WRAP_CSHARP)
    list(APPEND components CSharpModule)
  endif()
  if(GDCM_WRAP_JAVA)
    list(APPEND components JavaModule)
  endif()
  if(GDCM_WRAP_PHP)
    list(APPEND components PHPModule)
  endif()
  if(GDCM_USE_VTK)
    foreach(comp ${components})
      if( "${comp}" STREQUAL "PythonModule" )
        if(VTK_WRAP_PYTHON)
          list(APPEND components VTK${comp})
        endif()
      elseif( "${comp}" STREQUAL "JavaModule" )
        if(VTK_WRAP_JAVA)
          list(APPEND components VTK${comp})
        endif()
      else()
        list(APPEND components VTK${comp})
      endif()
    endforeach()
  endif()
  if(GDCM_USE_PARAVIEW)
    list(APPEND components ParaViewModule)
  endif()
  # Ok this is the complete list of all components:
  set(CPACK_COMPONENTS_ALL ${components})

  # Set the displayed names for each of the components to install.
  # These will be displayed in the list of components inside the installer.
  set(CPACK_COMPONENT_APPLICATIONS_DISPLAY_NAME "GDCM Applications")
  set(CPACK_COMPONENT_LIBRARIES_DISPLAY_NAME "GDCM Libraries")
  set(CPACK_COMPONENT_HEADERS_DISPLAY_NAME "GDCM C/C++ Headers")
  set(CPACK_COMPONENT_DEBUGDEVEL_DISPLAY_NAME "GDCM Debug Symbols (PDB)")
  set(CPACK_COMPONENT_PYTHONMODULE_DISPLAY_NAME "GDCM Python Module")
  set(CPACK_COMPONENT_CSHARPMODULE_DISPLAY_NAME "GDCM C# Module")
  set(CPACK_COMPONENT_JAVAMODULE_DISPLAY_NAME "GDCM Java Module")
  set(CPACK_COMPONENT_PHPMODULE_DISPLAY_NAME "GDCM PHP Module")
  set(CPACK_COMPONENT_VTKAPPLICATIONS_DISPLAY_NAME "VTK/GDCM Applications")
  set(CPACK_COMPONENT_VTKLIBRARIES_DISPLAY_NAME "VTK/GDCM Libraries")
  set(CPACK_COMPONENT_VTKHEADERS_DISPLAY_NAME "VTK/GDCM C/C++ Headers")
  set(CPACK_COMPONENT_VTKPYTHONMODULE_DISPLAY_NAME "VTK/GDCM Python Module")
  set(CPACK_COMPONENT_VTKCSHARPMODULE_DISPLAY_NAME "VTK/GDCM C# Module")
  set(CPACK_COMPONENT_VTKJAVAMODULE_DISPLAY_NAME "VTK/GDCM Java Module")
  set(CPACK_COMPONENT_VTKPHPMODULE_DISPLAY_NAME "VTK/GDCM PHP Module")
  set(CPACK_COMPONENT_PARAVIEWMODULE_DISPLAY_NAME "ParaView Module")

  # Provide descriptions for each of the components to install.
  # When the user hovers the mouse over the name of a component,
  # the description will be shown in the "Description" box in the
  # installer. If no descriptions are provided, the "Description"
  # box will be removed.
  set(CPACK_COMPONENT_APPLICATIONS_DESCRIPTION
    "Command line applications that uses GDCM: gdcmconv, gdcmscu, gdcmdump, gdcminfo, gdcmscanner, gdcmimg, gdcmanon")
  set(CPACK_COMPONENT_LIBRARIES_DESCRIPTION
    "Libraries used to build programs with GDCM")
  set(CPACK_COMPONENT_HEADERS_DESCRIPTION
    "C/C++ header files for use with GDCM")
  set(CPACK_COMPONENT_DEBUGDEVEL_DESCRIPTION
    "Program Database files for use with GDCM")
  set(CPACK_COMPONENT_PYTHONMODULE_DESCRIPTION
    "Python Module for GDCM")
  set(CPACK_COMPONENT_CSHARPMODULE_DESCRIPTION
    "C# Module for GDCM")
  set(CPACK_COMPONENT_JAVAMODULE_DESCRIPTION
    "Java Module for GDCM")
  set(CPACK_COMPONENT_PHPMODULE_DESCRIPTION
    "PHP Module for GDCM")
  set(CPACK_COMPONENT_VTKAPPLICATIONS_DESCRIPTION
    "Command line applications that uses GDCM: gdcmviewer & gdcm2vtk")
  set(CPACK_COMPONENT_VTKLIBRARIES_DESCRIPTION
    "Libraries used to build programs with VTK/GDCM")
  set(CPACK_COMPONENT_VTKHEADERS_DESCRIPTION
    "C/C++ header files for use with VTK/GDCM")
  set(CPACK_COMPONENT_VTKPYTHONMODULE_DESCRIPTION
    "Python Module for VTK/GDCM")
  set(CPACK_COMPONENT_VTKCSHARPMODULE_DESCRIPTION
    "C# Module for VTK/GDCM")
  set(CPACK_COMPONENT_VTKJAVAMODULE_DESCRIPTION
    "Java Module for VTK/GDCM")
  set(CPACK_COMPONENT_VTKPHPMODULE_DESCRIPTION
    "PHP Module for VTK/GDCM")
  set(CPACK_COMPONENT_PARAVIEWMODULE_DESCRIPTION
    "ParaView Module for VTK/GDCM")


  # Put the components into two different groups: "Runtime" and "Development"
  set(CPACK_COMPONENT_APPLICATIONS_GROUP "Runtime")
  set(CPACK_COMPONENT_LIBRARIES_GROUP "Development")
  set(CPACK_COMPONENT_HEADERS_GROUP "Development")
  set(CPACK_COMPONENT_DEBUGDEVEL_GROUP "Development")
  set(CPACK_COMPONENT_PYTHONMODULE_GROUP "Runtime")
  set(CPACK_COMPONENT_CSHARPMODULE_GROUP "Runtime")
  set(CPACK_COMPONENT_JAVAMODULE_GROUP "Runtime")
  set(CPACK_COMPONENT_PHPMODULE_GROUP "Runtime")
  set(CPACK_COMPONENT_VTKAPPLICATIONS_GROUP "Runtime")
  set(CPACK_COMPONENT_VTKPYTHONMODULE_GROUP "Runtime")
  set(CPACK_COMPONENT_VTKCSHARPMODULE_GROUP "Runtime")
  set(CPACK_COMPONENT_VTKJAVAMODULE_GROUP "Runtime")
  set(CPACK_COMPONENT_VTKPHPMODULE_GROUP "Runtime")
  set(CPACK_COMPONENT_VTKLIBRARIES_GROUP "Development")
  set(CPACK_COMPONENT_VTKHEADERS_GROUP "Development")
  set(CPACK_COMPONENT_PARAVIEWMODULE_GROUP "Runtime")
  # Expand the "Development" group by default, since we have so few components.
  # Also, provide this group with a description.
  if(CMAKE_VERSION VERSION_EQUAL 2.8.3)
    # The following is needed for CMake 2.8.3 and above to preserve backward compat
    set(CPACK_MONOLITHIC_INSTALL 1)
  endif()
  set(CPACK_COMPONENT_GROUP_DEVELOPMENT_EXPANDED ON)
  set(CPACK_COMPONENT_GROUP_DEVELOPMENT_DESCRIPTION
    "All of the tools you'll ever need to develop software using GDCM")

  # It doesn't make sense to install the headers without the libraries
  # (because you could never use the headers!), so make the headers component
  # depend on the libraries component.
  set(CPACK_COMPONENT_HEADERS_DEPENDS Libraries)
  set(CPACK_COMPONENT_DEBUGDEVEL_DEPENDS Libraries)
  set(CPACK_COMPONENT_APPLICATIONS_DEPENDS Libraries)
  set(CPACK_COMPONENT_PYTHONMODULE_DEPENDS Libraries)
  set(CPACK_COMPONENT_CSHARPMODULE_DEPENDS Libraries)
  set(CPACK_COMPONENT_JAVAMODULE_DEPENDS Libraries)
  set(CPACK_COMPONENT_PHPMODULE_DEPENDS Libraries)
  set(CPACK_COMPONENT_VTKHEADERS_DEPENDS VTKLibraries)
  set(CPACK_COMPONENT_VTKLIBRARIES_DEPENDS Libraries)
  set(CPACK_COMPONENT_VTKAPPLICATIONS_DEPENDS VTKLibraries)
  set(CPACK_COMPONENT_VTKPYTHONMODULE_DEPENDS VTKLibraries)
  set(CPACK_COMPONENT_VTKCSHARPMODULE_DEPENDS VTKLibraries)
  set(CPACK_COMPONENT_VTKJAVAMODULE_DEPENDS VTKLibraries)
  set(CPACK_COMPONENT_VTKPHPMODULE_DEPENDS VTKLibraries)
  set(CPACK_COMPONENT_PARAVIEWMODULE_DEPENDS VTKLibraries)

  # Create two installation types with pre-selected components.
  # The "Developer" installation has just the library and headers,
  # while the "Full" installation has everything.
  set(CPACK_ALL_INSTALL_TYPES Full Developer)
  set(CPACK_INSTALL_TYPE_FULL_DISPLAY_NAME "Everything")
  set(CPACK_COMPONENT_APPLICATIONS_INSTALL_TYPES Full)
  set(CPACK_COMPONENT_LIBRARIES_INSTALL_TYPES Developer Full)
  set(CPACK_COMPONENT_HEADERS_INSTALL_TYPES Developer Full)
  set(CPACK_COMPONENT_DEBUGDEVEL_INSTALL_TYPES Developer Full)
  set(CPACK_COMPONENT_PYTHONMODULE_INSTALL_TYPES Developer Full)
  set(CPACK_COMPONENT_CSHARPMODULE_INSTALL_TYPES Developer Full)
  set(CPACK_COMPONENT_JAVAMODULE_INSTALL_TYPES Developer Full)
  set(CPACK_COMPONENT_PHPMODULE_INSTALL_TYPES Developer Full)
  set(CPACK_COMPONENT_VTKLIBRARIES_INSTALL_TYPES Developer Full)
  set(CPACK_COMPONENT_VTKHEADERS_INSTALL_TYPES Developer Full)
  set(CPACK_COMPONENT_VTKAPPLICATIONS_INSTALL_TYPES Full)
  set(CPACK_COMPONENT_VTKPYTHONMODULE_INSTALL_TYPES Developer Full)
  set(CPACK_COMPONENT_VTKCSHARPMODULE_INSTALL_TYPES Developer Full)
  set(CPACK_COMPONENT_VTKJAVAMODULE_INSTALL_TYPES Developer Full)
  set(CPACK_COMPONENT_VTKPHPMODULE_INSTALL_TYPES Developer Full)
  set(CPACK_COMPONENT_PARAVIEWMODULE_INSTALL_TYPES Developer Full)

  include(CPack)
  endif()

endif()


macro(PROCESSONEDIR DIRNAME myoutput)
  file(GLOB files
    ${DIRNAME}/*
  )
  foreach(file ${files})
    if(IS_DIRECTORY ${file})
      if("${file}" MATCHES ".git")
        #message("${file} is git dir")
      else()
        #message("${file} is dir")
        PROCESSONEDIR(${file} mytests2)
        list(APPEND ${myoutput} ${mytests2})
      endif()
    else()
      #message("${file} is file")
      get_filename_component(filename ${file} NAME)
      if("${filename}" MATCHES "Test" AND ${filename} MATCHES ".cxx")
        string(LENGTH ${filename} filename_length)
        #message("${filename} is test, ${filename_length}")
        math(EXPR fileend "${filename_length} - 4 - 4") # Need to remove 'Test' and '.cxx'
        string(SUBSTRING ${filename} 4 ${fileend} classname)
        #message("${classname} is tested")
        list(APPEND ${myoutput} ${classname})
      endif()
    endif()
  endforeach()
endmacro()

#PROCESSONEDIR(
#  ${CMAKE_CURRENT_SOURCE_DIR}/Testing/Source/Common
#  theoutput
#)
#message("${theoutput}")

# For DICOM Q/R testing
if(GDCM_BUILD_TESTING)
  set(GDCM_DICOM_CLIENT_AETITLE "" CACHE STRING "DICOM CLIENT AETITLE")
  set(GDCM_DICOM_SERVER_AETITLE "" CACHE STRING "DICOM SERVER AETITLE")
  set(GDCM_DICOM_SERVER_PEER    "" CACHE STRING "DICOM SERVER PEER")
  set(GDCM_DICOM_SERVER_PORT    "" CACHE STRING "DICOM SERVER PORT")
  set(GDCM_DICOM_CLIENT_PORT    "" CACHE STRING "DICOM CLIENT PORT")
  mark_as_advanced(
    GDCM_DICOM_CLIENT_AETITLE
    GDCM_DICOM_SERVER_AETITLE
    GDCM_DICOM_SERVER_PEER
    GDCM_DICOM_SERVER_PORT
    GDCM_DICOM_CLIENT_PORT
    )
endif()

#-----------------------------------------------------------------------------
# Need to be the last operation:
set(GDCM_INCLUDE_PATH
  "${GDCM_SOURCE_DIR}/Source/Common"
  "${GDCM_BINARY_DIR}/Source/Common"
  "${GDCM_SOURCE_DIR}/Source/DataStructureAndEncodingDefinition"
  "${GDCM_SOURCE_DIR}/Source/MediaStorageAndFileFormat"
  "${GDCM_SOURCE_DIR}/Source/MessageExchangeDefinition"
  "${GDCM_SOURCE_DIR}/Source/DataDictionary"
  "${GDCM_SOURCE_DIR}/Source/InformationObjectDefinition"
  )
if(NOT GDCM_HAVE_STDINT_H)
  if(MSVC)
    set(GDCM_INCLUDE_PATH
      ${GDCM_INCLUDE_PATH}
      "${GDCM_SOURCE_DIR}/Utilities/C99"
    )
  endif()
endif()
if(GDCM_USE_VTK)
    set(GDCM_INCLUDE_PATH
      ${GDCM_INCLUDE_PATH}
      "${GDCM_SOURCE_DIR}/Utilities/VTK"
      )
endif()
set(GDCM_LIBRARY_DIRS ${LIBRARY_OUTPUT_PATH})
if(GDCM_STANDALONE)
subdirs(CMake/ExportConfiguration)
endif()
