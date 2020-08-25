# Main CMakeLists.txt to build the OpenJPEG project using CMake (www.cmake.org)
# Written by Mathieu Malaterre

# This CMake project will by default create a library called openjpeg
# But if you want to use this project within your own (CMake) project
# you will eventually like to prefix the library to avoid linking confusion
# For this purpose you can define a CMake var: OPENJPEG_NAMESPACE to whatever you like
# e.g.:
# set(OPENJPEG_NAMESPACE "GDCMOPENJPEG")
cmake_minimum_required(VERSION 2.8.2)

if(COMMAND CMAKE_POLICY)
  cmake_policy(SET CMP0003 NEW)
  if (NOT (${CMAKE_VERSION} VERSION_LESS 3.0))
    cmake_policy(SET CMP0042 NEW)
  endif()
endif()

if(NOT OPENJPEG_NAMESPACE)
  set(OPENJPEG_NAMESPACE "OPENJPEG")
  set(OPENJPEG_STANDALONE 1)
endif()
# In all cases:
#string(TOLOWER ${OPENJPEG_NAMESPACE} OPENJPEG_LIBRARY_NAME)
set(OPENJPEG_LIBRARY_NAME openjp2)

project(${OPENJPEG_NAMESPACE})

# Do full dependency headers.
include_regular_expression("^.*$")

#-----------------------------------------------------------------------------
# OPENJPEG version number, useful for packaging and doxygen doc:
set(OPENJPEG_VERSION_MAJOR 2)
set(OPENJPEG_VERSION_MINOR 3)
set(OPENJPEG_VERSION_BUILD 1)
set(OPENJPEG_VERSION
  "${OPENJPEG_VERSION_MAJOR}.${OPENJPEG_VERSION_MINOR}.${OPENJPEG_VERSION_BUILD}")
set(PACKAGE_VERSION
  "${OPENJPEG_VERSION_MAJOR}.${OPENJPEG_VERSION_MINOR}.${OPENJPEG_VERSION_BUILD}")

# Because autotools does not support X.Y notation for SOVERSION, we have to use
# two numbering, one for the openjpeg version and one for openjpeg soversion
# version | soversion
#   1.0   |  0
#   1.1   |  1
#   1.2   |  2
#   1.3   |  3
#   1.4   |  4
#   1.5   |  5
#   1.5.1 |  5
#   2.0   |  6
#   2.0.1 |  6
#   2.1   |  7
#   2.1.1 |  7
#   2.1.2 |  7
#   2.2.0 |  7
#   2.3.0 |  7
#   2.3.1 |  7
# above is the recommendation by the OPJ team. If you really need to override this default,
# you can specify your own OPENJPEG_SOVERSION at cmake configuration time:
# cmake -DOPENJPEG_SOVERSION:STRING=42 /path/to/openjpeg
if(NOT OPENJPEG_SOVERSION)
  set(OPENJPEG_SOVERSION 7)
endif(NOT OPENJPEG_SOVERSION)
set(OPENJPEG_LIBRARY_PROPERTIES
  VERSION   "${OPENJPEG_VERSION_MAJOR}.${OPENJPEG_VERSION_MINOR}.${OPENJPEG_VERSION_BUILD}"
  SOVERSION "${OPENJPEG_SOVERSION}"
)

# --------------------------------------------------------------------------
# Path to additional CMake modules
set(CMAKE_MODULE_PATH
    ${${OPENJPEG_NAMESPACE}_SOURCE_DIR}/cmake
    ${CMAKE_MODULE_PATH})

# --------------------------------------------------------------------------
# On Visual Studio 8 MS deprecated C. This removes all 1.276E1265 security
# warnings
if(WIN32)
  if(NOT BORLAND)
    if(NOT CYGWIN)
      if(NOT MINGW)
        if(NOT ITK_ENABLE_VISUAL_STUDIO_DEPRECATED_C_WARNINGS)
          add_definitions(
            -D_CRT_FAR_MAPPINGS_NO_DEPRECATE
            -D_CRT_IS_WCTYPE_NO_DEPRECATE
            -D_CRT_MANAGED_FP_NO_DEPRECATE
            -D_CRT_NONSTDC_NO_DEPRECATE
            -D_CRT_SECURE_NO_DEPRECATE
            -D_CRT_SECURE_NO_DEPRECATE_GLOBALS
            -D_CRT_SETERRORMODE_BEEP_SLEEP_NO_DEPRECATE
            -D_CRT_TIME_FUNCTIONS_NO_DEPRECATE
            -D_CRT_VCCLRIT_NO_DEPRECATE
            -D_SCL_SECURE_NO_DEPRECATE
            )
        endif()
      endif()
    endif()
  endif()
endif()


# --------------------------------------------------------------------------
# Install directories
# Build DOCUMENTATION (not in ALL target and only if Doxygen is found)
option(BUILD_DOC "Build the HTML documentation (with doxygen if available)." OFF)

string(TOLOWER ${PROJECT_NAME} projectname)
set(OPENJPEG_INSTALL_SUBDIR "${projectname}-${OPENJPEG_VERSION_MAJOR}.${OPENJPEG_VERSION_MINOR}")

if(NOT OPENJPEG_INSTALL_BIN_DIR)
  set(OPENJPEG_INSTALL_BIN_DIR "bin")
endif()

if(NOT OPENJPEG_INSTALL_LIB_DIR)
  set(OPENJPEG_INSTALL_LIB_DIR "lib")
endif()

if(NOT OPENJPEG_INSTALL_SHARE_DIR)
  set(OPENJPEG_INSTALL_SHARE_DIR "share")
endif()

if(NOT OPENJPEG_INSTALL_DATA_DIR)
  set(OPENJPEG_INSTALL_DATA_DIR "${OPENJPEG_INSTALL_SHARE_DIR}/${OPENJPEG_INSTALL_SUBDIR}")
endif()

if(NOT OPENJPEG_INSTALL_INCLUDE_DIR)
  set(OPENJPEG_INSTALL_INCLUDE_DIR "include/${OPENJPEG_INSTALL_SUBDIR}")
endif()

if(BUILD_DOC)
if(NOT OPENJPEG_INSTALL_MAN_DIR)
  set(OPENJPEG_INSTALL_MAN_DIR "share/man/")
endif()

if(NOT OPENJPEG_INSTALL_DOC_DIR)
  set(OPENJPEG_INSTALL_DOC_DIR "share/doc/${OPENJPEG_INSTALL_SUBDIR}")
endif()
endif()

if(NOT OPENJPEG_INSTALL_JNI_DIR)
  if(WIN32)
    set(OPENJPEG_INSTALL_JNI_DIR ${OPENJPEG_INSTALL_BIN_DIR})
  else()
    set(OPENJPEG_INSTALL_JNI_DIR ${OPENJPEG_INSTALL_LIB_DIR})
  endif()
endif()

if(NOT OPENJPEG_INSTALL_PACKAGE_DIR)
  # We could install *.cmake files in share/ however those files contains
  # hardcoded path to libraries on a multi-arch system (fedora/debian) those
  # path will be different (lib/i386-linux-gnu vs lib/x86_64-linux-gnu)
  set(OPENJPEG_INSTALL_PACKAGE_DIR "${OPENJPEG_INSTALL_LIB_DIR}/${OPENJPEG_INSTALL_SUBDIR}")
endif()

if (APPLE)
	list(APPEND OPENJPEG_LIBRARY_PROPERTIES INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/${OPENJPEG_INSTALL_LIB_DIR}")
	option(OPJ_USE_DSYMUTIL "Call dsymutil on binaries after build." OFF)
endif()

#-----------------------------------------------------------------------------
# Big endian test:
include (${CMAKE_ROOT}/Modules/TestBigEndian.cmake)
TEST_BIG_ENDIAN(OPJ_BIG_ENDIAN)

#-----------------------------------------------------------------------------
# Setup file for setting custom ctest vars
configure_file(
  ${${OPENJPEG_NAMESPACE}_SOURCE_DIR}/cmake/CTestCustom.cmake.in
  ${${OPENJPEG_NAMESPACE}_BINARY_DIR}/CTestCustom.cmake
  @ONLY
  )

#-----------------------------------------------------------------------------
# OpenJPEG build configuration options.
option(BUILD_SHARED_LIBS "Build OpenJPEG shared library and link executables against it." ON)
option(BUILD_STATIC_LIBS "Build OpenJPEG static library." ON)
set (EXECUTABLE_OUTPUT_PATH ${OPENJPEG_BINARY_DIR}/bin CACHE PATH "Single output directory for building all executables.")
set (LIBRARY_OUTPUT_PATH ${OPENJPEG_BINARY_DIR}/bin CACHE PATH "Single output directory for building all libraries.")
mark_as_advanced(LIBRARY_OUTPUT_PATH EXECUTABLE_OUTPUT_PATH)

#-----------------------------------------------------------------------------
# configure name mangling to allow multiple libraries to coexist
# peacefully
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/openjpeg_mangle.h.in)
set(MANGLE_PREFIX ${OPENJPEG_LIBRARY_NAME})
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/openjpeg_mangle.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/openjpeg_mangle.h
               @ONLY)
endif()

#-----------------------------------------------------------------------------
# Compiler specific flags:
if(CMAKE_COMPILER_IS_GNUCC)
  # For all builds, make sure openjpeg is std99 compliant:
  # set(CMAKE_C_FLAGS "-Wall -std=c99 ${CMAKE_C_FLAGS}") # FIXME: this setting prevented us from setting a coverage build.
  # Do not use ffast-math for all build, it would produce incorrect results, only set for release:
  set(OPENJPEG_LIBRARY_COMPILE_OPTIONS ${OPENJPEG_LIBRARY_COMPILE_OPTIONS} "$<$<CONFIG:Release>:-ffast-math>")
  set(OPENJP2_COMPILE_OPTIONS ${OPENJP2_COMPILE_OPTIONS} "$<$<CONFIG:Release>:-ffast-math>" -Wall -Wextra -Wconversion -Wunused-parameter -Wdeclaration-after-statement -Werror=declaration-after-statement)
endif()

#-----------------------------------------------------------------------------
# opj_config.h generation (1/2)

# Check if some include files are provided by the system
include(EnsureFileInclude)
# These files are mandatory
ensure_file_include("string.h"   HAVE_STRING_H YES)
ensure_file_include("memory.h"   HAVE_MEMORY_H YES)
ensure_file_include("stdlib.h"   HAVE_STDLIB_H YES)
ensure_file_include("stdio.h"    HAVE_STDIO_H  YES)
ensure_file_include("math.h"     HAVE_MATH_H   YES)
ensure_file_include("float.h"    HAVE_FLOAT_H  YES)
ensure_file_include("time.h"     HAVE_TIME_H   YES)
ensure_file_include("stdarg.h"   HAVE_STDARG_H YES)
ensure_file_include("ctype.h"    HAVE_CTYPE_H  YES)
ensure_file_include("assert.h"   HAVE_ASSERT_H YES)

# For the following files, we provide an alternative, they are not mandatory
ensure_file_include("stdint.h"   OPJ_HAVE_STDINT_H   NO)
ensure_file_include("inttypes.h" OPJ_HAVE_INTTYPES_H NO)

# why check this one ? for openjpip ?
include (${CMAKE_ROOT}/Modules/CheckIncludeFile.cmake)
CHECK_INCLUDE_FILE("strings.h"      HAVE_STRINGS_H)
CHECK_INCLUDE_FILE("sys/stat.h"     HAVE_SYS_STAT_H)
CHECK_INCLUDE_FILE("sys/types.h"    HAVE_SYS_TYPES_H)
CHECK_INCLUDE_FILE("unistd.h"       HAVE_UNISTD_H)

# Enable Large file support
include(TestLargeFiles)
OPJ_TEST_LARGE_FILES(OPJ_HAVE_LARGEFILES)

# Allocating Aligned Memory Blocks
include(CheckIncludeFiles)
check_include_files(malloc.h OPJ_HAVE_MALLOC_H)
include(CheckSymbolExists)
# _aligned_alloc https://msdn.microsoft.com/en-us/library/8z34s9c6.aspx
check_symbol_exists(_aligned_malloc malloc.h OPJ_HAVE__ALIGNED_MALLOC)
# posix_memalign (needs _POSIX_C_SOURCE >= 200112L on Linux)
set(CMAKE_REQUIRED_DEFINITIONS -D_POSIX_C_SOURCE=200112L)
check_symbol_exists(posix_memalign stdlib.h OPJ_HAVE_POSIX_MEMALIGN)
unset(CMAKE_REQUIRED_DEFINITIONS)
# memalign (obsolete)
check_symbol_exists(memalign malloc.h OPJ_HAVE_MEMALIGN)
#-----------------------------------------------------------------------------
# Build Library
if(BUILD_JPIP_SERVER)
  find_package(CURL REQUIRED)
  find_package(FCGI REQUIRED)
  find_package(Threads REQUIRED)
  if(NOT CMAKE_USE_PTHREADS_INIT)
    message(FATAL_ERROR "Only pthread are supported")
  endif()
endif()
add_subdirectory(src/lib)
option(BUILD_LUTS_GENERATOR "Build utility to generate t1_luts.h" OFF)
option(BUILD_UNIT_TESTS "Build unit tests (bench_dwt, test_sparse_array, etc..)" OFF)

#-----------------------------------------------------------------------------
# Build Applications
option(BUILD_CODEC "Build the CODEC executables" ON)
option(BUILD_MJ2 "Build the MJ2 executables." OFF)
option(BUILD_JPWL "Build the JPWL library and executables" OFF)
option(BUILD_JPIP "Build the JPIP library and executables." OFF)
if(BUILD_JPIP)
  option(BUILD_JPIP_SERVER "Build the JPIP server." OFF)
endif()
option(BUILD_VIEWER "Build the OPJViewer executable (C++)" OFF)
option(BUILD_JAVA "Build the openjpeg jar (Java)" OFF)
option(BUILD_JP3D "Build the JP3D comp" OFF)
mark_as_advanced(BUILD_VIEWER)
mark_as_advanced(BUILD_JAVA)
mark_as_advanced(BUILD_JP3D)

if(BUILD_CODEC OR BUILD_MJ2)
  # OFF: It will only build 3rd party libs if they are not found on the system
  # ON: 3rd party libs will ALWAYS be build, and used
  option(BUILD_THIRDPARTY "Build the thirdparty executables if it is needed" OFF)
  add_subdirectory(thirdparty)
  add_subdirectory(src/bin)
endif ()
add_subdirectory(wrapping)

#-----------------------------------------------------------------------------
# opj_config.h generation (2/2)
configure_file(
 ${CMAKE_CURRENT_SOURCE_DIR}/src/lib/openjp2/opj_config.h.cmake.in
 ${CMAKE_CURRENT_BINARY_DIR}/src/lib/openjp2/opj_config.h
 @ONLY
 )

 configure_file(
 ${CMAKE_CURRENT_SOURCE_DIR}/src/lib/openjp2/opj_config_private.h.cmake.in
 ${CMAKE_CURRENT_BINARY_DIR}/src/lib/openjp2/opj_config_private.h
 @ONLY
 )

#-----------------------------------------------------------------------------
# build documentation in doc subdir:
if(BUILD_DOC)
  add_subdirectory(doc)
endif()

#-----------------------------------------------------------------------------
# Buld Testing
option(BUILD_TESTING "Build the tests." OFF)
if(BUILD_TESTING)
  if(BUILD_CODEC)
    enable_testing()
    include(CTest)

    # Search openjpeg data needed for the tests
    # They could be found via git on the OpenJPEG GitHub code project
    # git clone https://github.com/uclouvain/openjpeg-data.git
    find_path(OPJ_DATA_ROOT README-OPJ-Data
      PATHS $ENV{OPJ_DATA_ROOT} ${CMAKE_SOURCE_DIR}/../data
      NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH
      )

    # Add repository where to find tests
    add_subdirectory(tests)

  else()
    message(FATAL_ERROR "You need build codec to run the tests")
  endif()
endif()

#-----------------------------------------------------------------------------
# install all targets referenced as OPENJPEGTargets
install(EXPORT OpenJPEGTargets DESTINATION ${OPENJPEG_INSTALL_PACKAGE_DIR})
configure_file( ${${OPENJPEG_NAMESPACE}_SOURCE_DIR}/cmake/OpenJPEGConfig.cmake.in
  ${${OPENJPEG_NAMESPACE}_BINARY_DIR}/OpenJPEGConfig.cmake
  @ONLY
)
install( FILES ${OPENJPEG_BINARY_DIR}/OpenJPEGConfig.cmake
  DESTINATION ${OPENJPEG_INSTALL_PACKAGE_DIR}
)

#-----------------------------------------------------------------------------
# install CHANGES and LICENSE
if(BUILD_DOC)
if(EXISTS ${OPENJPEG_SOURCE_DIR}/CHANGES)
  install(FILES CHANGES DESTINATION ${OPENJPEG_INSTALL_DOC_DIR})
endif()

install(FILES LICENSE DESTINATION ${OPENJPEG_INSTALL_DOC_DIR})
endif()

include (cmake/OpenJPEGCPack.cmake)

#-----------------------------------------------------------------------------
# pkgconfig support
# enabled by default on Unix or if using GCC, disabled by default on other platforms
if(UNIX OR CMAKE_COMPILER_IS_GNUCC)
  option(BUILD_PKGCONFIG_FILES "Build and install pkg-config files" ON)
else()
  option(BUILD_PKGCONFIG_FILES "Build and install pkg-config files" OFF)
endif()
if(BUILD_PKGCONFIG_FILES)
  # install in lib and not share (see multi-arch note above)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/lib/openjp2/libopenjp2.pc.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/libopenjp2.pc @ONLY)
  install( FILES  ${CMAKE_CURRENT_BINARY_DIR}/libopenjp2.pc DESTINATION
    ${OPENJPEG_INSTALL_LIB_DIR}/pkgconfig )
#
  if(BUILD_JPWL)
  # install in lib and not share (see multi-arch note above)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/lib/openjpwl/libopenjpwl.pc.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/libopenjpwl.pc @ONLY)
  install( FILES  ${CMAKE_CURRENT_BINARY_DIR}/libopenjpwl.pc DESTINATION
    ${OPENJPEG_INSTALL_LIB_DIR}/pkgconfig )
  endif()
#
  if(BUILD_JPIP)
  # install in lib and not share (see multi-arch note above)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/lib/openjpip/libopenjpip.pc.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/libopenjpip.pc @ONLY)
  install( FILES  ${CMAKE_CURRENT_BINARY_DIR}/libopenjpip.pc DESTINATION
    ${OPENJPEG_INSTALL_LIB_DIR}/pkgconfig )
  endif()
#
  if(BUILD_JP3D)
  # install in lib and not share (see multi-arch note above)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/lib/openjp3d/libopenjp3d.pc.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/libopenjp3d.pc @ONLY)
  install( FILES  ${CMAKE_CURRENT_BINARY_DIR}/libopenjp3d.pc DESTINATION
    ${OPENJPEG_INSTALL_LIB_DIR}/pkgconfig )
  endif()
endif()

#-----------------------------------------------------------------------------

# build our version of astyle
SET (WITH_ASTYLE FALSE CACHE BOOL "If you plan to contribute you should reindent with scripts/prepare-commit.sh (using 'our' astyle)")
