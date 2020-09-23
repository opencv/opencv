# CMakeLists.txt
#
# Copyright (C) 2013-2020 by
# David Turner, Robert Wilhelm, and Werner Lemberg.
#
# Written originally by John Cary <cary@txcorp.com>
#
# This file is part of the FreeType project, and may only be used, modified,
# and distributed under the terms of the FreeType project license,
# LICENSE.TXT.  By continuing to use, modify, or distribute this file you
# indicate that you have read the license and understand and accept it
# fully.
#
#
# The following will 1. create a build directory and 2. change into it and
# call cmake to configure the build with default parameters as a static
# library. See https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html
# for information about Debug, Release, etc. builds.
#
#   cmake -B build -D CMAKE_BUILD_TYPE=Release
#
# For a dynamic library, use
#
#   cmake -B build -D BUILD_SHARED_LIBS=true -D CMAKE_BUILD_TYPE=Release
#
# For a framework on OS X, use
#
#   cmake -E chdir build cmake -G Xcode -D BUILD_FRAMEWORK:BOOL=true ..
#
# For an iOS static library, use
#
#   cmake -E chdir build cmake -G Xcode -D IOS_PLATFORM=OS ..
#
# or
#
#   cmake -E chdir build cmake -G Xcode -D IOS_PLATFORM=SIMULATOR ..
#
# or
#
#   cmake -E chdir build cmake -G Xcode -D IOS_PLATFORM=SIMULATOR64 ..
#
# Finally, build the project with:
#
#   cmake --build build
#
# Install it with
#
#   (sudo) cmake --build build --target install
#
# A binary distribution can be made with
#
#   cmake --build build --config Release --target package
#
# Please refer to the cmake manual for further options, in particular, how
# to modify compilation and linking parameters.
#
# Some notes.
#
# . `cmake' creates configuration files in
#
#     <build-directory>/include/freetype/config
#
#   which should be further modified if necessary.
#
# . You can use `cmake' directly on a freshly cloned FreeType git
#   repository.
#
# . `CMakeLists.txt' is provided as-is since it is normally not used by the
#   developer team.
#
# . Set the `FT_WITH_ZLIB', `FT_WITH_BZIP2', `FT_WITH_PNG',
#   `FT_WITH_HARFBUZZ', and `FT_WITH_BROTLI' CMake variables to `ON' to
#   force using a dependency.  Leave a variable undefined (which is the
#   default) to use the dependency only if it is available. Example:
#
#     cmake -B build -D FT_WITH_ZLIB=ON \
#                    -D FT_WITH_BZIP2=ON \
#                    -D FT_WITH_PNG=ON \
#                    -D FT_WITH_HARFBUZZ=ON \
#                    -D FT_WITH_BROTLI=ON [...]
#
#   Set `CMAKE_DISABLE_FIND_PACKAGE_XXX=TRUE' to disable a dependency completely
#   (where `XXX' is a CMake package name like `BZip2').  Example for disabling all
#   dependencies:
#
#     cmake -B build -D CMAKE_DISABLE_FIND_PACKAGE_ZLIB=TRUE \
#                    -D CMAKE_DISABLE_FIND_PACKAGE_BZip2=TRUE \
#                    -D CMAKE_DISABLE_FIND_PACKAGE_PNG=TRUE \
#                    -D CMAKE_DISABLE_FIND_PACKAGE_HarfBuzz=TRUE \
#                    -D CMAKE_DISABLE_FIND_PACKAGE_BrotliDec=TRUE [...]
#
# . Installation of FreeType can be controlled with the CMake variables
#   `SKIP_INSTALL_HEADERS', `SKIP_INSTALL_LIBRARIES', and `SKIP_INSTALL_ALL'
#   (this is compatible with the same CMake variables in zlib's CMake
#   support).
project(freetype)

set(VERSION_MAJOR "2")
set(VERSION_MINOR "10")
set(VERSION_PATCH "2")

# Generate LIBRARY_VERSION and LIBRARY_SOVERSION.
set(LIBTOOL_REGEX "version_info='([0-9]+):([0-9]+):([0-9]+)'")
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/builds/unix/configure.raw"
  VERSION_INFO
  REGEX ${LIBTOOL_REGEX})
string(REGEX REPLACE
  ${LIBTOOL_REGEX} "\\1"
  LIBTOOL_CURRENT "${VERSION_INFO}")
string(REGEX REPLACE
  ${LIBTOOL_REGEX} "\\2"
  LIBTOOL_REVISION "${VERSION_INFO}")
string(REGEX REPLACE
  ${LIBTOOL_REGEX} "\\3"
  LIBTOOL_AGE "${VERSION_INFO}")

set(FREETYPE_VERSION "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}" PARENT_SCOPE)

# External dependency library detection is automatic. See the notes at the top
# of this file, for how to force or disable dependencies completely.
#option(FT_WITH_ZLIB "Use system zlib instead of internal library." OFF)
#option(FT_WITH_BZIP2 "Support bzip2 compressed fonts." OFF)
#option(FT_WITH_PNG "Support PNG compressed OpenType embedded bitmaps." OFF)
#option(FT_WITH_HARFBUZZ "Improve auto-hinting of OpenType fonts." OFF)
#option(FT_WITH_BROTLI "Support compressed WOFF2 fonts." OFF)

# Create the configuration file
if (UNIX)
  check_include_file("unistd.h" HAVE_UNISTD_H)
  check_include_file("fcntl.h" HAVE_FCNTL_H)
  check_include_file("stdint.h" HAVE_STDINT_H)

  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/builds/unix/ftconfig.in"
    FTCONFIG_H)
  if (HAVE_UNISTD_H)
    string(REGEX REPLACE
      "#undef +(HAVE_UNISTD_H)" "#define \\1 1"
      FTCONFIG_H "${FTCONFIG_H}")
  endif ()
  if (HAVE_FCNTL_H)
    string(REGEX REPLACE
      "#undef +(HAVE_FCNTL_H)" "#define \\1 1"
      FTCONFIG_H "${FTCONFIG_H}")
  endif ()
  if (HAVE_STDINT_H)
    string(REGEX REPLACE
      "#undef +(HAVE_STDINT_H)" "#define \\1 1"
      FTCONFIG_H "${FTCONFIG_H}")
  endif ()
  string(REPLACE "/undef " "#undef "
    FTCONFIG_H "${FTCONFIG_H}")
else ()
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/include/freetype/config/ftconfig.h"
    FTCONFIG_H)
endif ()

set(FTCONFIG_H_NAME "${CMAKE_CURRENT_BINARY_DIR}/include/freetype/config/ftconfig.h")
if (EXISTS "${FTCONFIG_H_NAME}")
  file(READ "${FTCONFIG_H_NAME}" ORIGINAL_FTCONFIG_H)
else ()
  set(ORIGINAL_FTCONFIG_H "")
endif ()
if (NOT (ORIGINAL_FTCONFIG_H STREQUAL FTCONFIG_H))
  file(WRITE "${FTCONFIG_H_NAME}" "${FTCONFIG_H}")
endif ()

# Create the options file
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/include/freetype/config/ftoption.h"
  FTOPTION_H)

set(FTOPTION_H_NAME "${CMAKE_CURRENT_BINARY_DIR}/include/freetype/config/ftoption.h")
if (EXISTS "${FTOPTION_H_NAME}")
  file(READ "${FTOPTION_H_NAME}" ORIGINAL_FTOPTION_H)
else ()
  set(ORIGINAL_FTOPTION_H "")
endif ()
if (NOT (ORIGINAL_FTOPTION_H STREQUAL FTOPTION_H))
  file(WRITE "${FTOPTION_H_NAME}" "${FTOPTION_H}")
endif ()

file(GLOB PUBLIC_HEADERS "include/ft2build.h" "include/freetype/*.h")
file(GLOB PUBLIC_CONFIG_HEADERS "include/freetype/config/*.h")
file(GLOB PRIVATE_HEADERS "include/freetype/internal/*.h")

set(BASE_SRCS
  src/autofit/autofit.c
  src/base/ftbase.c
  src/base/ftbbox.c
  src/base/ftbdf.c
  src/base/ftbitmap.c
  src/base/ftcid.c
  src/base/ftfstype.c
  src/base/ftgasp.c
  src/base/ftglyph.c
  src/base/ftgxval.c
  src/base/ftinit.c
  src/base/ftmm.c
  src/base/ftotval.c
  src/base/ftpatent.c
  src/base/ftpfr.c
  src/base/ftstroke.c
  src/base/ftsynth.c
  src/base/fttype1.c
  src/base/ftwinfnt.c
  src/bdf/bdf.c
  src/bzip2/ftbzip2.c
  src/cache/ftcache.c
  src/cff/cff.c
  src/cid/type1cid.c
  #src/gzip/ftgzip.c
  src/lzw/ftlzw.c
  src/pcf/pcf.c
  src/pfr/pfr.c
  src/psaux/psaux.c
  src/pshinter/pshinter.c
  src/psnames/psnames.c
  src/raster/raster.c
  src/sfnt/sfnt.c
  src/smooth/smooth.c
  src/truetype/truetype.c
  src/type1/type1.c
  src/type42/type42.c
  src/winfonts/winfnt.c
)

if (UNIX)
  list(APPEND BASE_SRCS "builds/unix/ftsystem.c")
else ()
  list(APPEND BASE_SRCS "src/base/ftsystem.c")
endif ()

if (WIN32)
  enable_language(RC)
  list(APPEND BASE_SRCS builds/windows/ftdebug.c
                        src/base/ftver.rc)
elseif (WINCE)
  list(APPEND BASE_SRCS builds/wince/ftdebug.c)
else ()
  list(APPEND BASE_SRCS src/base/ftdebug.c)
endif ()

if (NOT DISABLE_FORCE_DEBUG_POSTFIX)
  set(CMAKE_DEBUG_POSTFIX d)
endif ()


add_library(${PROJECT_NAME} STATIC
  ${PUBLIC_HEADERS}
  ${PUBLIC_CONFIG_HEADERS}
  ${PRIVATE_HEADERS}
  ${BASE_SRCS}
)

set_target_properties(
  ${PROJECT_NAME} PROPERTIES
    OUTPUT_NAME freetype
    C_VISIBILITY_PRESET hidden
    DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
    COMPILE_PDB_NAME freetype
    COMPILE_PDB_NAME_DEBUG "freetype${OPENCV_DEBUG_POSTFIX}"
    ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH}
)    

target_compile_definitions(
  ${PROJECT_NAME} PRIVATE FT2_BUILD_LIBRARY)

if (WIN32)
  target_compile_definitions(
    ${PROJECT_NAME} PRIVATE _CRT_SECURE_NO_WARNINGS _CRT_NONSTDC_NO_WARNINGS)
  if (BUILD_SHARED_LIBS)
    target_compile_definitions(
      ${PROJECT_NAME} PRIVATE DLL_EXPORT)
  endif ()
endif ()

ocv_warnings_disable(CMAKE_C_FLAGS -Wunused-parameter -Wstrict-prototypes) # clang

set(FREETYPE_INCLUDE_DIR
"${CMAKE_CURRENT_BINARY_DIR}/include/freetype/config"
"${CMAKE_CURRENT_BINARY_DIR}/include"
"${CMAKE_CURRENT_SOURCE_DIR}/include" PARENT_SCOPE)

# Pick up ftconfig.h and ftoption.h generated above, first.
target_include_directories(
  ${PROJECT_NAME}
  ${FREETYPE_INCLUDE_DIR}
    PUBLIC
      $<INSTALL_INTERFACE:include/freetype>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
      "${CMAKE_CURRENT_BINARY_DIR}/include/freetype/config"
      "${CMAKE_CURRENT_BINARY_DIR}/include"
      "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "3rdparty")
endif()

if(NOT BUILD_SHARED_LIBS)
  ocv_install_target(${PROJECT_NAME} EXPORT OpenCVModules ARCHIVE DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev)
endif()

ocv_install_3rdparty_licenses(${PROJECT_NAME} README)
