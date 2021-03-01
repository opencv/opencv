# Build scripts are adopted to OpenCV project
# Main CMakeLists.txt to build the OpenJPEG project using CMake (www.cmake.org)
# Written by Mathieu Malaterre

cmake_policy(SET CMP0003 NEW)
if(POLICY CMP0042)
  cmake_policy(SET CMP0042 NEW)
endif()

set(OPENJPEG_LIBRARY_NAME libopenjp2)

project(openjpeg C)

ocv_warnings_disable(CMAKE_C_FLAGS
    -Wimplicit-const-int-float-conversion  # clang
)

#-----------------------------------------------------------------------------
# OPENJPEG version number, useful for packaging and doxygen doc:
set(OPENJPEG_VERSION_MAJOR 2)
set(OPENJPEG_VERSION_MINOR 4)
set(OPENJPEG_VERSION_BUILD 0)
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
#   2.4.0 |  7
# above is the recommendation by the OPJ team. If you really need to override this default,
# you can specify your own OPENJPEG_SOVERSION at cmake configuration time:
# cmake -DOPENJPEG_SOVERSION:STRING=42 /path/to/openjpeg
if(NOT OPENJPEG_SOVERSION)
  set(OPENJPEG_SOVERSION 7)
endif()

set(OPENJPEG_LIBRARY_PROPERTIES
  VERSION   "${OPENJPEG_VERSION_MAJOR}.${OPENJPEG_VERSION_MINOR}.${OPENJPEG_VERSION_BUILD}"
  SOVERSION "${OPENJPEG_SOVERSION}"
)

set(OPENJPEG_BUILD "opencv-${OPENCV_VERSION}-openjp2-${OPENJPEG_VERSION}")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(OPENJPEG_BUILD "${OPENJPEG_BUILD}-debug")
endif()

message(STATUS "OpenJPEG: VERSION = ${OPENJPEG_VERSION}, BUILD = ${OPENJPEG_BUILD}")

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

#-----------------------------------------------------------------------------
# Big endian test:
include(TestBigEndian)
test_big_endian(OPJ_BIG_ENDIAN)

#-----------------------------------------------------------------------------
# opj_config.h generation (1/2)

# Check if some include files are provided by the system
# These files are mandatory, so if they are not provided OpenJPEG library can't be built
include(CheckIncludeFile)
macro(ensure_file_include INCLUDE_FILENAME VARIABLE_NAME MANDATORY_STATUS)
  check_include_file(${INCLUDE_FILENAME} ${VARIABLE_NAME})
  if(NOT ${VARIABLE_NAME})
    if(${MANDATORY_STATUS})
      message(STATUS "The file '${INCLUDE_FILENAME}' is mandatory for OpenJPEG build, but not found on your system")
      return()
    else()
      message(STATUS "The file '${INCLUDE_FILENAME}' is optional for OpenJPEG build and not found on your system."
              " Internal implementation will be used.")
    endif()
  endif()
endmacro()

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
CHECK_INCLUDE_FILE("strings.h"      HAVE_STRINGS_H)
CHECK_INCLUDE_FILE("sys/stat.h"     HAVE_SYS_STAT_H)
CHECK_INCLUDE_FILE("sys/types.h"    HAVE_SYS_TYPES_H)
CHECK_INCLUDE_FILE("unistd.h"       HAVE_UNISTD_H)

# Allocating Aligned Memory Blocks
include(CheckIncludeFiles)
check_include_files(malloc.h OPJ_HAVE_MALLOC_H)
include(CheckSymbolExists)
# _aligned_alloc https://msdn.microsoft.com/en-us/library/8z34s9c6.aspx
check_symbol_exists(_aligned_malloc malloc.h OPJ_HAVE__ALIGNED_MALLOC)
# posix_memalign (needs _POSIX_C_SOURCE >= 200112L on Linux)
set(_prev_CMAKE_REQUIRED_DEFINITIONS ${CMAKE_REQUIRED_DEFINITIONS})
set(CMAKE_REQUIRED_DEFINITIONS -D_POSIX_C_SOURCE=200112L)
check_symbol_exists(posix_memalign stdlib.h OPJ_HAVE_POSIX_MEMALIGN)
set(CMAKE_REQUIRED_DEFINITIONS ${_prev_CMAKE_REQUIRED_DEFINITIONS})
unset(_prev_CMAKE_REQUIRED_DEFINITIONS)
# memalign (obsolete)
check_symbol_exists(memalign malloc.h OPJ_HAVE_MEMALIGN)
#-----------------------------------------------------------------------------
# opj_config.h generation (2/2)
configure_file(
  ${CMAKE_CURRENT_LIST_DIR}/openjp2/opj_config.h.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/openjp2/opj_config.h
  @ONLY
)

configure_file(
  ${CMAKE_CURRENT_LIST_DIR}/openjp2/opj_config_private.h.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/openjp2/opj_config_private.h
  @ONLY
)

add_subdirectory(openjp2)

set_target_properties(${OPENJPEG_LIBRARY_NAME}
  PROPERTIES
    OUTPUT_NAME              ${OPENJPEG_LIBRARY_NAME}
    DEBUG_POSTFIX            "${OPENCV_DEBUG_POSTFIX}"
    COMPILE_PDB_NAME         ${OPENJPEG_LIBRARY_NAME}
    COMPILE_PDB_NAME_DEBUG   "${OPENJPEG_LIBRARY_NAME}${OPENCV_DEBUG_POSTFIX}"
    ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH}
)

if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(${OPENJPEG_LIBRARY_NAME}
    PROPERTIES
      FOLDER "3rdparty"
  )
endif()

ocv_install_3rdparty_licenses(${OPENJPEG_LIBRARY_NAME} README.md LICENSE)

# Setting all necessary variables
set(OPENJPEG_LIBRARIES     ${OPENJPEG_LIBRARY_NAME}  PARENT_SCOPE)
set(OPENJPEG_VERSION       ${OPENJPEG_VERSION}       PARENT_SCOPE)
set(OPENJPEG_MAJOR_VERSION ${OPENJPEG_VERSION_MAJOR} PARENT_SCOPE)
set(OPENJPEG_MINOR_VERSION ${OPENJPEG_VERSION_MINOR} PARENT_SCOPE)
set(OPENJPEG_BUILD_VERSION ${OPENJPEG_VERSION_BUILD} PARENT_SCOPE)
get_target_property(_openjpeg_include_dirs ${OPENJPEG_LIBRARY_NAME} INCLUDE_DIRECTORIES)
set(OPENJPEG_INCLUDE_DIRS  ${_openjpeg_include_dirs} PARENT_SCOPE)

# OpenJPEG can't be built only if configuration script doesn't encounter any problem
if(NOT DEFINED OCV_CAN_BUILD_OPENJPEG)
  # all prerequisites are fulfilled
  set(OCV_CAN_BUILD_OPENJPEG TRUE PARENT_SCOPE)
endif()
