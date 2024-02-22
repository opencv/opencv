#  Copyright (c) 2020 Google LLC.
#
#  Use of this source code is governed by a BSD-style license
#  that can be found in the LICENSE file in the root of the source
#  tree. An additional intellectual property rights grant can be found
#  in the file PATENTS.  All contributing project authors may
#  be found in the AUTHORS file in the root of the source tree.

if(APPLE)
  cmake_minimum_required(VERSION 3.17)
else()
  cmake_minimum_required(VERSION 3.7)
endif()

if(POLICY CMP0072)
  cmake_policy(SET CMP0072 NEW)
endif()

project(WebP C)

# Options for coder / decoder executables.
if(BUILD_SHARED_LIBS)
  set(WEBP_LINK_STATIC_DEFAULT OFF)
else()
  set(WEBP_LINK_STATIC_DEFAULT ON)
endif()
option(WEBP_LINK_STATIC
       "Link using static libraries. If OFF, use dynamic libraries."
       ${WEBP_LINK_STATIC_DEFAULT})
if(NOT EMSCRIPTEN)
  # Disable SIMD on Emscripten by default, as it's a new unstable Wasm feature.
  # Users can still explicitly opt-in to make a SIMD-enabled build.
  set(WEBP_ENABLE_SIMD_DEFAULT ON)
endif()
option(WEBP_ENABLE_SIMD "Enable any SIMD optimization."
       ${WEBP_ENABLE_SIMD_DEFAULT})
option(WEBP_BUILD_ANIM_UTILS "Build animation utilities." ON)
option(WEBP_BUILD_CWEBP "Build the cwebp command line tool." ON)
option(WEBP_BUILD_DWEBP "Build the dwebp command line tool." ON)
option(WEBP_BUILD_GIF2WEBP "Build the gif2webp conversion tool." ON)
option(WEBP_BUILD_IMG2WEBP "Build the img2webp animation tool." ON)
option(WEBP_BUILD_VWEBP "Build the vwebp viewer tool." ON)
option(WEBP_BUILD_WEBPINFO "Build the webpinfo command line tool." ON)
option(WEBP_BUILD_LIBWEBPMUX "Build the libwebpmux library." ON)
option(WEBP_BUILD_WEBPMUX "Build the webpmux command line tool." ON)
option(WEBP_BUILD_EXTRAS "Build extras." ON)
option(WEBP_BUILD_WEBP_JS "Emscripten build of webp.js." OFF)
option(WEBP_USE_THREAD "Enable threading support" ON)
option(WEBP_NEAR_LOSSLESS "Enable near-lossless encoding" ON)
option(WEBP_ENABLE_SWAP_16BIT_CSP "Enable byte swap for 16 bit colorspaces."
       OFF)
set(WEBP_BITTRACE "0" CACHE STRING "Bit trace mode (0=none, 1=bit, 2=bytes)")
set_property(CACHE WEBP_BITTRACE PROPERTY STRINGS 0 1 2)

if(WEBP_LINK_STATIC)
  if(WIN32)
    set(CMAKE_FIND_LIBRARY_SUFFIXES .lib .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
  else()
    set(CMAKE_FIND_LIBRARY_SUFFIXES .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
  endif()
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  # vwebp does not compile on Ubuntu with static libraries so disabling it for
  # now.
  set(WEBP_BUILD_VWEBP OFF)
endif()

# Option needed for handling Unicode file names on Windows.
if(WIN32)
  option(WEBP_UNICODE "Build Unicode executables." ON)
endif()

if(WEBP_BUILD_WEBP_JS)
  set(WEBP_BUILD_ANIM_UTILS OFF)
  set(WEBP_BUILD_CWEBP OFF)
  set(WEBP_BUILD_DWEBP OFF)
  set(WEBP_BUILD_GIF2WEBP OFF)
  set(WEBP_BUILD_IMG2WEBP OFF)
  set(WEBP_BUILD_VWEBP OFF)
  set(WEBP_BUILD_WEBPINFO OFF)
  set(WEBP_BUILD_WEBPMUX OFF)
  set(WEBP_BUILD_EXTRAS OFF)
  set(WEBP_USE_THREAD OFF)

  if(WEBP_ENABLE_SIMD)
    message(NOTICE
            "wasm2js does not support SIMD, disabling webp.js generation.")
  endif()
endif()

set(SHARPYUV_DEP_LIBRARIES)
set(SHARPYUV_DEP_INCLUDE_DIRS)
set(WEBP_DEP_LIBRARIES)
set(WEBP_DEP_INCLUDE_DIRS)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release"
      CACHE STRING "Build type: Release, Debug, MinSizeRel or RelWithDebInfo"
            FORCE)
endif()

# Include dependencies.
if(WEBP_BUILD_ANIM_UTILS
   OR WEBP_BUILD_CWEBP
   OR WEBP_BUILD_DWEBP
   OR WEBP_BUILD_EXTRAS
   OR WEBP_BUILD_GIF2WEBP
   OR WEBP_BUILD_IMG2WEBP)
  set(WEBP_FIND_IMG_LIBS TRUE)
else()
  set(WEBP_FIND_IMG_LIBS FALSE)
endif()
include(cmake/deps.cmake)
include(GNUInstallDirs)

if(BUILD_SHARED_LIBS AND NOT DEFINED CMAKE_INSTALL_RPATH)
  # Set the rpath to match autoconf/libtool behavior. Note this must be set
  # before target creation.
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
endif()

# ##############################################################################
# Options.
if(WEBP_ENABLE_SWAP_16BIT_CSP)
  add_definitions(-DWEBP_SWAP_16BIT_CSP=1)
endif()

if(NOT WEBP_BITTRACE STREQUAL "0")
  add_definitions(-DBITTRACE=${WEBP_BITTRACE})
endif()

if(WEBP_UNICODE)
  # Windows recommends setting both UNICODE and _UNICODE.
  add_definitions(-DUNICODE -D_UNICODE)
endif()

if(MSVC AND BUILD_SHARED_LIBS)
  add_definitions(-DWEBP_DLL)
endif()

# pkg-config variables used by *.pc.in.
set(prefix ${CMAKE_INSTALL_PREFIX})
set(exec_prefix "\${prefix}")
if(IS_ABSOLUTE "${CMAKE_INSTALL_LIBDIR}")
  set(libdir "${CMAKE_INSTALL_LIBDIR}")
else()
  set(libdir "\${exec_prefix}/${CMAKE_INSTALL_LIBDIR}")
endif()
if(IS_ABSOLUTE "${CMAKE_INSTALL_INCLUDEDIR}")
  set(includedir "${CMAKE_INSTALL_INCLUDEDIR}")
else()
  set(includedir "\${prefix}/${CMAKE_INSTALL_INCLUDEDIR}")
endif()
set(PTHREAD_LIBS ${CMAKE_THREAD_LIBS_INIT})
set(INSTALLED_LIBRARIES)

if(MSVC)
  # match the naming convention used by nmake
  set(webp_libname_prefix "lib")
  set(CMAKE_SHARED_LIBRARY_PREFIX "${webp_libname_prefix}")
  set(CMAKE_IMPORT_LIBRARY_PREFIX "${webp_libname_prefix}")
  set(CMAKE_STATIC_LIBRARY_PREFIX "${webp_libname_prefix}")
endif()

set(CMAKE_C_VISIBILITY_PRESET hidden)

# ##############################################################################
# Android only.
if(ANDROID)
  include_directories(${ANDROID_NDK}/sources/android/cpufeatures)
  add_library(cpufeatures-webp STATIC
              ${ANDROID_NDK}/sources/android/cpufeatures/cpu-features.c)
  list(APPEND INSTALLED_LIBRARIES cpufeatures-webp)
  target_link_libraries(cpufeatures-webp dl)
  set(SHARPYUV_DEP_LIBRARIES ${SHARPYUV_DEP_LIBRARIES} cpufeatures-webp)
  set(WEBP_DEP_LIBRARIES ${WEBP_DEP_LIBRARIES} cpufeatures-webp)
  set(cpufeatures_include_dir ${ANDROID_NDK}/sources/android/cpufeatures)
  set(SHARPYUV_DEP_INCLUDE_DIRS ${SHARPYUV_DEP_INCLUDE_DIRS}
                                ${cpufeatures_include_dir})
  set(WEBP_DEP_INCLUDE_DIRS ${WEBP_DEP_INCLUDE_DIRS} ${cpufeatures_include_dir})
  add_definitions(-DHAVE_CPU_FEATURES_H=1)
  set(HAVE_CPU_FEATURES_H 1)
else()
  set(HAVE_CPU_FEATURES_H 0)
endif()

function(configure_pkg_config FILE)
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/${FILE}.in"
                 "${CMAKE_CURRENT_BINARY_DIR}/${FILE}" @ONLY)

  if(HAVE_MATH_LIBRARY)
    # MSVC doesn't have libm
    file(READ ${CMAKE_CURRENT_BINARY_DIR}/${FILE} data)
    string(REPLACE "-lm" "" data ${data})
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${FILE} ${data})
  endif()

  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${FILE}"
          DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
endfunction()

# ##############################################################################
# WebP source files. Read the Makefile.am to get the source files.

# We expect the Makefiles to define the sources as defined in the first regex.
# E.g.: libimagedec_la_SOURCES  = image_dec.c image_dec.h
function(parse_Makefile_am FOLDER VAR SRC_REGEX)
  file(READ ${FOLDER}/Makefile.am MAKEFILE_AM)
  string(REGEX MATCHALL "${SRC_REGEX}_SOURCES[ ]*\\+?=[ ]+[0-9a-z\\._ ]*"
               FILES_PER_LINE ${MAKEFILE_AM})
  set(SRCS ${${VAR}})
  foreach(FILES ${FILES_PER_LINE})
    string(FIND ${FILES} "=" OFFSET)
    math(EXPR OFFSET "${OFFSET} + 2")
    string(SUBSTRING ${FILES} ${OFFSET} -1 FILES)
    if(FILES)
      string(REGEX MATCHALL "[0-9a-z\\._]+" FILES ${FILES})
      foreach(FILE ${FILES})
        list(APPEND SRCS ${FOLDER}/${FILE})
      endforeach()
    endif()
  endforeach()
  set(${VAR} ${SRCS} PARENT_SCOPE)
endfunction()

set(WEBP_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
parse_makefile_am(${WEBP_SRC_DIR}/dec "WEBP_DEC_SRCS" "")
parse_makefile_am(${WEBP_SRC_DIR}/demux "WEBP_DEMUX_SRCS" "")
parse_makefile_am(${WEBP_SRC_DIR}/dsp "WEBP_DSP_COMMON_SRCS" "COMMON")
parse_makefile_am(${WEBP_SRC_DIR}/dsp "WEBP_DSP_ENC_SRCS" "ENC")
parse_makefile_am(${WEBP_SRC_DIR}/dsp "WEBP_DSP_ENC_SRCS" "dsp_[^ ]*")
parse_makefile_am(${WEBP_SRC_DIR}/dsp "WEBP_DSP_DEC_SRCS" "decode_[^ ]*")
parse_makefile_am(${WEBP_SRC_DIR}/enc "WEBP_ENC_SRCS" "")
parse_makefile_am(${WEBP_SRC_DIR}/utils "WEBP_UTILS_COMMON_SRCS" "COMMON")
parse_makefile_am(${WEBP_SRC_DIR}/utils "WEBP_UTILS_ENC_SRCS" "ENC")
parse_makefile_am(${WEBP_SRC_DIR}/utils "WEBP_UTILS_DEC_SRCS" "decode_[^ ]*")

# Remove the files specific to SIMD we don't use.
foreach(FILE ${WEBP_SIMD_FILES_NOT_TO_INCLUDE})
  list(REMOVE_ITEM WEBP_DSP_ENC_SRCS ${FILE})
  list(REMOVE_ITEM WEBP_DSP_DEC_SRCS ${FILE})
endforeach()

# Generate the config.h file.
configure_file(${CMAKE_CURRENT_LIST_DIR}/cmake/config.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/src/webp/config.h @ONLY)
add_definitions(-DHAVE_CONFIG_H)

# Set the version numbers.
macro(set_version FILE TARGET_NAME NAME_IN_MAKEFILE)
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/${FILE} SOURCE_FILE)
  string(REGEX MATCH
               "${NAME_IN_MAKEFILE}_la_LDFLAGS[^\n]* -version-info [0-9:]+" TMP
               ${SOURCE_FILE})
  string(REGEX MATCH "[0-9:]+" TMP ${TMP})
  string(REGEX REPLACE ":" " " LT_VERSION ${TMP})

  # See the libtool docs for more information:
  # https://www.gnu.org/software/libtool/manual/libtool.html#Updating-version-info
  #
  # c=<current>, a=<age>, r=<revision>
  #
  # libtool generates a .so file as .so.[c-a].a.r, while -version-info c:r:a is
  # passed to libtool.
  #
  # We set FULL = [c-a].a.r and MAJOR = [c-a].
  separate_arguments(LT_VERSION)
  list(GET LT_VERSION 0 LT_CURRENT)
  list(GET LT_VERSION 1 LT_REVISION)
  list(GET LT_VERSION 2 LT_AGE)
  math(EXPR LT_CURRENT_MINUS_AGE "${LT_CURRENT} - ${LT_AGE}")

  set_target_properties(
    ${TARGET_NAME}
    PROPERTIES VERSION ${LT_CURRENT_MINUS_AGE}.${LT_AGE}.${LT_REVISION}
               SOVERSION ${LT_CURRENT_MINUS_AGE})
  if(APPLE)
    # For compatibility, set MACHO_COMPATIBILITY_VERSION and
    # MACHO_CURRENT_VERSION to match libtool. These properties were introduced
    # in 3.17:
    # https://cmake.org/cmake/help/latest/prop_tgt/MACHO_COMPATIBILITY_VERSION.html
    math(EXPR LIBWEBP_MACHO_COMPATIBILITY_VERSION "${LT_CURRENT} + 1")
    set_target_properties(
      ${TARGET_NAME}
      PROPERTIES MACHO_COMPATIBILITY_VERSION
                 ${LIBWEBP_MACHO_COMPATIBILITY_VERSION}
                 MACHO_CURRENT_VERSION
                 ${LIBWEBP_MACHO_COMPATIBILITY_VERSION}.${LT_REVISION})
  endif()
endmacro()

# ##############################################################################
# Build the webpdecoder library.

# Creates a source file with an unused stub function in $CMAKE_BINARY_DIR and
# adds it to the specified target. Currently used only with Xcode.
#
# See also:
# https://cmake.org/cmake/help/v3.18/command/add_library.html#object-libraries
# "Some native build systems (such as Xcode) may not like targets that have only
# object files, so consider adding at least one real source file to any target
# that references $<TARGET_OBJECTS:objlib>."
function(libwebp_add_stub_file TARGET)
  set(stub_source_dir "${CMAKE_BINARY_DIR}")
  set(stub_source_file "${stub_source_dir}/libwebp_${TARGET}_stub.c")
  set(stub_source_code
      "// Generated file. DO NOT EDIT!\n"
      "// C source file created for target ${TARGET}.\n"
      "void libwebp_${TARGET}_stub_function(void)\;\n"
      "void libwebp_${TARGET}_stub_function(void) {}\n")
  file(WRITE "${stub_source_file}" ${stub_source_code})

  target_sources(${TARGET} PRIVATE ${stub_source_file})
endfunction()

parse_makefile_am(${CMAKE_CURRENT_SOURCE_DIR}/sharpyuv "WEBP_SHARPYUV_SRCS" "")
add_library(sharpyuv ${WEBP_SHARPYUV_SRCS})
target_link_libraries(sharpyuv ${SHARPYUV_DEP_LIBRARIES})
set_version(sharpyuv/Makefile.am sharpyuv sharpyuv)
target_include_directories(
  sharpyuv PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}
                   ${CMAKE_CURRENT_SOURCE_DIR}/src)
set_target_properties(
  sharpyuv
  PROPERTIES PUBLIC_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/sharpyuv/sharpyuv.h;\
${CMAKE_CURRENT_SOURCE_DIR}/sharpyuv/sharpyuv_csp.h")
configure_pkg_config("sharpyuv/libsharpyuv.pc")
install(
  TARGETS sharpyuv
  EXPORT ${PROJECT_NAME}Targets
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/webp/sharpyuv
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  ${CMAKE_INSTALL_INCLUDEDIR}/webp
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if(MSVC)
  # avoid security warnings for e.g., fopen() used in the examples.
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
else()
  add_compile_options(-Wall)
endif()
include_directories(${WEBP_DEP_INCLUDE_DIRS})
add_library(webpdecode OBJECT ${WEBP_DEC_SRCS})
target_include_directories(webpdecode PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
                                              ${CMAKE_CURRENT_SOURCE_DIR})
add_library(webpdspdecode OBJECT ${WEBP_DSP_COMMON_SRCS} ${WEBP_DSP_DEC_SRCS})
target_include_directories(webpdspdecode PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
                                                 ${CMAKE_CURRENT_SOURCE_DIR})
add_library(webputilsdecode OBJECT ${WEBP_UTILS_COMMON_SRCS}
                                   ${WEBP_UTILS_DEC_SRCS})
target_include_directories(webputilsdecode PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
                                                   ${CMAKE_CURRENT_SOURCE_DIR})
add_library(
  webpdecoder $<TARGET_OBJECTS:webpdecode> $<TARGET_OBJECTS:webpdspdecode>
              $<TARGET_OBJECTS:webputilsdecode>)
if(XCODE)
  libwebp_add_stub_file(webpdecoder)
endif()
target_link_libraries(webpdecoder ${WEBP_DEP_LIBRARIES})
target_include_directories(
  webpdecoder PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
set_target_properties(
  webpdecoder
  PROPERTIES PUBLIC_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/src/webp/decode.h;\
${CMAKE_CURRENT_SOURCE_DIR}/src/webp/types.h")

configure_pkg_config("src/libwebpdecoder.pc")

# Build the webp library.
add_library(webpencode OBJECT ${WEBP_ENC_SRCS})
target_include_directories(
  webpencode PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}
                     ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_library(webpdsp OBJECT ${WEBP_DSP_COMMON_SRCS} ${WEBP_DSP_DEC_SRCS}
                           ${WEBP_DSP_ENC_SRCS})
target_include_directories(webpdsp PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
                                           ${CMAKE_CURRENT_SOURCE_DIR})
add_library(webputils OBJECT ${WEBP_UTILS_COMMON_SRCS} ${WEBP_UTILS_DEC_SRCS}
                             ${WEBP_UTILS_ENC_SRCS})
target_include_directories(webputils PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
                                             ${CMAKE_CURRENT_SOURCE_DIR})
add_library(webp $<TARGET_OBJECTS:webpdecode> $<TARGET_OBJECTS:webpdsp>
                 $<TARGET_OBJECTS:webpencode> $<TARGET_OBJECTS:webputils>)
target_link_libraries(webp sharpyuv)
if(XCODE)
  libwebp_add_stub_file(webp)
endif()
target_link_libraries(webp ${WEBP_DEP_LIBRARIES})
target_include_directories(
  webp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
         $<INSTALL_INTERFACE:include>)
set_target_properties(
  webp
  PROPERTIES PUBLIC_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/src/webp/decode.h;\
${CMAKE_CURRENT_SOURCE_DIR}/src/webp/encode.h;\
${CMAKE_CURRENT_SOURCE_DIR}/src/webp/types.h")

# Make sure the OBJECT libraries are built with position independent code (it is
# not ON by default).
set_target_properties(webpdecode webpdspdecode webputilsdecode webpencode
                      webpdsp webputils PROPERTIES POSITION_INDEPENDENT_CODE ON)
configure_pkg_config("src/libwebp.pc")

# Build the webp demux library.
add_library(webpdemux ${WEBP_DEMUX_SRCS})
target_link_libraries(webpdemux webp)
target_include_directories(
  webpdemux PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}
  PUBLIC $<INSTALL_INTERFACE:include>)
set_target_properties(
  webpdemux
  PROPERTIES
    PUBLIC_HEADER
    "${CMAKE_CURRENT_SOURCE_DIR}/src/webp/decode.h;\
${CMAKE_CURRENT_SOURCE_DIR}/src/webp/demux.h;\
${CMAKE_CURRENT_SOURCE_DIR}/src/webp/mux_types.h;\
${CMAKE_CURRENT_SOURCE_DIR}/src/webp/types.h")

configure_pkg_config("src/demux/libwebpdemux.pc")

set_version(src/Makefile.am webp webp)
set_version(src/Makefile.am webpdecoder webpdecoder)
set_version(src/demux/Makefile.am webpdemux webpdemux)
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/configure.ac CONFIGURE_FILE)
string(REGEX MATCH "AC_INIT\\([^\n]*\\[[0-9\\.]+\\]" TMP ${CONFIGURE_FILE})
string(REGEX MATCH "[0-9\\.]+" PROJECT_VERSION ${TMP})

# Define the libraries to install.
list(APPEND INSTALLED_LIBRARIES webpdecoder webp webpdemux)

# Deal with SIMD. Change the compile flags for SIMD files we use.
list(LENGTH WEBP_SIMD_FILES_TO_INCLUDE WEBP_SIMD_FILES_TO_INCLUDE_LENGTH)
math(EXPR WEBP_SIMD_FILES_TO_INCLUDE_RANGE
     "${WEBP_SIMD_FILES_TO_INCLUDE_LENGTH}-1")

foreach(I_FILE RANGE ${WEBP_SIMD_FILES_TO_INCLUDE_RANGE})
  list(GET WEBP_SIMD_FILES_TO_INCLUDE ${I_FILE} FILE)
  list(GET WEBP_SIMD_FLAGS_TO_INCLUDE ${I_FILE} SIMD_COMPILE_FLAG)
  set_source_files_properties(${FILE} PROPERTIES COMPILE_FLAGS
                                                 ${SIMD_COMPILE_FLAG})
endforeach()

if(NOT WEBP_BUILD_LIBWEBPMUX)
  set(WEBP_BUILD_GIF2WEBP OFF)
  set(WEBP_BUILD_IMG2WEBP OFF)
  set(WEBP_BUILD_WEBPMUX OFF)
endif()

if(WEBP_BUILD_GIF2WEBP AND NOT GIF_FOUND)
  set(WEBP_BUILD_GIF2WEBP OFF)
endif()

if(WEBP_BUILD_ANIM_UTILS AND NOT GIF_FOUND)
  set(WEBP_BUILD_ANIM_UTILS OFF)
endif()

# Build the executables if asked for.
if(WEBP_BUILD_ANIM_UTILS
   OR WEBP_BUILD_CWEBP
   OR WEBP_BUILD_DWEBP
   OR WEBP_BUILD_GIF2WEBP
   OR WEBP_BUILD_IMG2WEBP
   OR WEBP_BUILD_VWEBP
   OR WEBP_BUILD_WEBPMUX
   OR WEBP_BUILD_WEBPINFO)
  # Example utility library.
  parse_makefile_am(${CMAKE_CURRENT_SOURCE_DIR}/examples "EXAMPLEUTIL_SRCS"
                    "example_util_[^ ]*")
  list(APPEND EXAMPLEUTIL_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/examples/stopwatch.h)
  add_library(exampleutil STATIC ${EXAMPLEUTIL_SRCS})
  target_include_directories(
    exampleutil PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)

  parse_makefile_am(${CMAKE_CURRENT_SOURCE_DIR}/imageio "IMAGEIOUTILS_SRCS"
                    "imageio_util_[^ ]*")
  add_library(imageioutil STATIC ${IMAGEIOUTILS_SRCS})
  target_link_libraries(imageioutil webp)
  target_link_libraries(exampleutil imageioutil)

  # Image-decoding utility library.
  parse_makefile_am(${CMAKE_CURRENT_SOURCE_DIR}/imageio "IMAGEDEC_SRCS"
                    "imagedec_[^ ]*")
  add_library(imagedec STATIC ${IMAGEDEC_SRCS})
  target_link_libraries(imagedec imageioutil webpdemux webp
                        ${WEBP_DEP_IMG_LIBRARIES})

  # Image-encoding utility library.
  parse_makefile_am(${CMAKE_CURRENT_SOURCE_DIR}/imageio "IMAGEENC_SRCS"
                    "imageenc_[^ ]*")
  add_library(imageenc STATIC ${IMAGEENC_SRCS})
  target_link_libraries(imageenc imageioutil webp)

  set_property(
    TARGET exampleutil imageioutil imagedec imageenc
    PROPERTY INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/src
             ${CMAKE_CURRENT_BINARY_DIR}/src)
endif()

if(WEBP_BUILD_DWEBP)
  # dwebp
  parse_makefile_am(${CMAKE_CURRENT_SOURCE_DIR}/examples "DWEBP_SRCS" "dwebp")
  add_executable(dwebp ${DWEBP_SRCS})
  target_link_libraries(dwebp exampleutil imagedec imageenc)
  target_include_directories(dwebp PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/src)
  install(TARGETS dwebp RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

if(WEBP_BUILD_CWEBP)
  # cwebp
  parse_makefile_am(${CMAKE_CURRENT_SOURCE_DIR}/examples "CWEBP_SRCS" "cwebp")
  add_executable(cwebp ${CWEBP_SRCS})
  target_link_libraries(cwebp exampleutil imagedec webp)
  target_include_directories(cwebp PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/src
                                           ${CMAKE_CURRENT_SOURCE_DIR})
  install(TARGETS cwebp RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

if(WEBP_BUILD_LIBWEBPMUX)
  parse_makefile_am(${CMAKE_CURRENT_SOURCE_DIR}/src/mux "WEBP_MUX_SRCS" "")
  add_library(libwebpmux ${WEBP_MUX_SRCS})
  target_link_libraries(libwebpmux webp)
  target_include_directories(libwebpmux PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
                                                ${CMAKE_CURRENT_SOURCE_DIR})
  set_version(src/mux/Makefile.am libwebpmux webpmux)
  set_target_properties(
    libwebpmux
    PROPERTIES PUBLIC_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/src/webp/mux.h;\
${CMAKE_CURRENT_SOURCE_DIR}/src/webp/mux_types.h;\
${CMAKE_CURRENT_SOURCE_DIR}/src/webp/types.h;")
  set_target_properties(libwebpmux PROPERTIES OUTPUT_NAME webpmux)
  list(APPEND INSTALLED_LIBRARIES libwebpmux)
  configure_pkg_config("src/mux/libwebpmux.pc")
endif()

if(WEBP_BUILD_GIF2WEBP)
  # gif2webp
  include_directories(${WEBP_DEP_GIF_INCLUDE_DIRS})
  parse_makefile_am(${CMAKE_CURRENT_SOURCE_DIR}/examples "GIF2WEBP_SRCS"
                    "gif2webp")
  add_executable(gif2webp ${GIF2WEBP_SRCS})
  target_link_libraries(gif2webp exampleutil imageioutil webp libwebpmux
                        ${WEBP_DEP_GIF_LIBRARIES})
  target_include_directories(gif2webp PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/src)
  install(TARGETS gif2webp RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

if(WEBP_BUILD_IMG2WEBP)
  # img2webp
  include_directories(${WEBP_DEP_IMG_INCLUDE_DIRS})
  parse_makefile_am(${CMAKE_CURRENT_SOURCE_DIR}/examples "IMG2WEBP_SRCS"
                    "img2webp")
  add_executable(img2webp ${IMG2WEBP_SRCS})
  target_link_libraries(img2webp exampleutil imagedec imageioutil webp
                        libwebpmux)
  target_include_directories(img2webp PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/src
                                              ${CMAKE_CURRENT_SOURCE_DIR})
  install(TARGETS img2webp RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

if(WEBP_BUILD_VWEBP)
  # vwebp
  find_package(GLUT)
  if(GLUT_FOUND)
    include_directories(${WEBP_DEP_IMG_INCLUDE_DIRS})
    parse_makefile_am(${CMAKE_CURRENT_SOURCE_DIR}/examples "VWEBP_SRCS" "vwebp")
    add_executable(vwebp ${VWEBP_SRCS})
    target_link_libraries(
      vwebp
      ${OPENGL_LIBRARIES}
      exampleutil
      GLUT::GLUT
      imageioutil
      webp
      webpdemux)
    target_include_directories(
      vwebp PRIVATE ${GLUT_INCLUDE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/src
                    ${OPENGL_INCLUDE_DIR})
    install(TARGETS vwebp RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    if(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
      check_c_compiler_flag("-Wno-deprecated-declarations" HAS_NO_DEPRECATED)
      if(HAS_NO_DEPRECATED)
        target_compile_options(vwebp PRIVATE "-Wno-deprecated-declarations")
      endif()
    endif()
  endif()
endif()

if(WEBP_BUILD_WEBPINFO)
  # webpinfo
  include_directories(${WEBP_DEP_IMG_INCLUDE_DIRS})
  parse_makefile_am(${CMAKE_CURRENT_SOURCE_DIR}/examples "WEBPINFO_SRCS"
                    "webpinfo")
  add_executable(webpinfo ${WEBPINFO_SRCS})
  target_link_libraries(webpinfo exampleutil imageioutil)
  target_include_directories(webpinfo PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/src
                                              ${CMAKE_CURRENT_SOURCE_DIR}/src)
  install(TARGETS webpinfo RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

if(WEBP_BUILD_WEBPMUX)
  # webpmux
  parse_makefile_am(${CMAKE_CURRENT_SOURCE_DIR}/examples "WEBPMUX_SRCS"
                    "webpmux")
  add_executable(webpmux ${WEBPMUX_SRCS})
  target_link_libraries(webpmux exampleutil imageioutil libwebpmux webp)
  target_include_directories(webpmux PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/src)
  install(TARGETS webpmux RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

if(WEBP_BUILD_EXTRAS)
  set(EXTRAS_MAKEFILE "${CMAKE_CURRENT_SOURCE_DIR}/extras")
  parse_makefile_am(${EXTRAS_MAKEFILE} "WEBP_EXTRAS_SRCS" "libwebpextras_la")
  parse_makefile_am(${EXTRAS_MAKEFILE} "GET_DISTO_SRCS" "get_disto")
  parse_makefile_am(${EXTRAS_MAKEFILE} "WEBP_QUALITY_SRCS" "webp_quality")
  parse_makefile_am(${EXTRAS_MAKEFILE} "VWEBP_SDL_SRCS" "vwebp_sdl")

  # libextras
  add_library(extras STATIC ${WEBP_EXTRAS_SRCS})
  target_include_directories(
    extras PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}
                   ${CMAKE_CURRENT_SOURCE_DIR}/src)

  # get_disto
  add_executable(get_disto ${GET_DISTO_SRCS})
  target_link_libraries(get_disto imagedec)
  target_include_directories(get_disto PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
                                               ${CMAKE_CURRENT_BINARY_DIR}/src)

  # webp_quality
  add_executable(webp_quality ${WEBP_QUALITY_SRCS})
  target_link_libraries(webp_quality exampleutil imagedec extras)
  target_include_directories(webp_quality PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
                                                  ${CMAKE_CURRENT_BINARY_DIR})

  # vwebp_sdl
  find_package(SDL)
  if(WEBP_BUILD_VWEBP AND SDL_FOUND)
    add_executable(vwebp_sdl ${VWEBP_SDL_SRCS})
    target_link_libraries(vwebp_sdl ${SDL_LIBRARY} imageioutil webp)
    target_include_directories(
      vwebp_sdl PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}
                        ${CMAKE_CURRENT_BINARY_DIR}/src ${SDL_INCLUDE_DIR})
    set(WEBP_HAVE_SDL 1)
    target_compile_definitions(vwebp_sdl PUBLIC WEBP_HAVE_SDL)
  endif()
endif()

if(WEBP_BUILD_WEBP_JS)
  # The default stack size changed from 5MB to 64KB in 3.1.27. See
  # https://crbug.com/webp/614.
  if(EMSCRIPTEN_VERSION VERSION_GREATER_EQUAL "3.1.27")
    # TOTAL_STACK size was renamed to STACK_SIZE in 3.1.27. The old name was
    # kept for compatibility, but prefer the new one in case it is removed in
    # the future.
    set(emscripten_stack_size "-sSTACK_SIZE=5MB")
  else()
    set(emscripten_stack_size "-sTOTAL_STACK=5MB")
  endif()
  # wasm2js does not support SIMD.
  if(NOT WEBP_ENABLE_SIMD)
    # JavaScript version
    add_executable(webp_js ${CMAKE_CURRENT_SOURCE_DIR}/extras/webp_to_sdl.c)
    target_link_libraries(webp_js webpdecoder SDL)
    target_include_directories(webp_js PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    set(WEBP_HAVE_SDL 1)
    set_target_properties(
      webp_js
      PROPERTIES LINK_FLAGS "-sWASM=0 ${emscripten_stack_size} \
         -sEXPORTED_FUNCTIONS=_WebPToSDL -sINVOKE_RUN=0 \
         -sEXPORTED_RUNTIME_METHODS=cwrap")
    set_target_properties(webp_js PROPERTIES OUTPUT_NAME webp)
    target_compile_definitions(webp_js PUBLIC EMSCRIPTEN WEBP_HAVE_SDL)
  endif()

  # WASM version
  add_executable(webp_wasm ${CMAKE_CURRENT_SOURCE_DIR}/extras/webp_to_sdl.c)
  target_link_libraries(webp_wasm webpdecoder SDL)
  target_include_directories(webp_wasm PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
  set_target_properties(
    webp_wasm
    PROPERTIES LINK_FLAGS "-sWASM=1 ${emscripten_stack_size} \
       -sEXPORTED_FUNCTIONS=_WebPToSDL -sINVOKE_RUN=0 \
       -sEXPORTED_RUNTIME_METHODS=cwrap")
  target_compile_definitions(webp_wasm PUBLIC EMSCRIPTEN WEBP_HAVE_SDL)

  target_compile_definitions(webpdspdecode PUBLIC EMSCRIPTEN)
endif()

if(WEBP_BUILD_ANIM_UTILS)
  # anim_diff
  include_directories(${WEBP_DEP_IMG_INCLUDE_DIRS} ${WEBP_DEP_GIF_INCLUDE_DIRS})
  parse_makefile_am(${CMAKE_CURRENT_SOURCE_DIR}/examples "ANIM_DIFF_SRCS"
                    "anim_diff")
  add_executable(anim_diff ${ANIM_DIFF_SRCS})
  target_link_libraries(
    anim_diff
    exampleutil
    imagedec
    imageenc
    imageioutil
    webp
    webpdemux
    ${WEBP_DEP_GIF_LIBRARIES})
  target_include_directories(anim_diff PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/src)

  # anim_dump
  include_directories(${WEBP_DEP_IMG_INCLUDE_DIRS} ${WEBP_DEP_GIF_INCLUDE_DIRS})
  parse_makefile_am(${CMAKE_CURRENT_SOURCE_DIR}/examples "ANIM_DUMP_SRCS"
                    "anim_dump")
  add_executable(anim_dump ${ANIM_DUMP_SRCS})
  target_link_libraries(
    anim_dump
    exampleutil
    imagedec
    imageenc
    imageioutil
    webp
    webpdemux
    ${WEBP_DEP_GIF_LIBRARIES})
  target_include_directories(anim_dump PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/src)
endif()

# Install the different headers and libraries.
install(
  TARGETS ${INSTALLED_LIBRARIES}
  EXPORT ${PROJECT_NAME}Targets
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/webp
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
set(ConfigPackageLocation ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/cmake/)
install(EXPORT ${PROJECT_NAME}Targets NAMESPACE ${PROJECT_NAME}::
        DESTINATION ${ConfigPackageLocation})

# Create the CMake version file.
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/WebPConfigVersion.cmake"
  VERSION ${PACKAGE_VERSION} COMPATIBILITY AnyNewerVersion)

# Create the Config file.
include(CMakePackageConfigHelpers)
# Fix libwebpmux reference. The target name libwebpmux is used for compatibility
# purposes, but the library mentioned in WebPConfig.cmake should be the
# unprefixed version. Note string(...) can be replaced with list(TRANSFORM ...)
# if cmake_minimum_required is >= 3.12.
string(REGEX REPLACE "libwebpmux" "webpmux" INSTALLED_LIBRARIES
                     "${INSTALLED_LIBRARIES}")

if(MSVC)
  # For compatibility with nmake, MSVC builds use a custom prefix (lib) that
  # needs to be included in the library name.
  string(REGEX REPLACE "[A-Za-z0-9_]+" "${CMAKE_STATIC_LIBRARY_PREFIX}\\0"
                       INSTALLED_LIBRARIES "${INSTALLED_LIBRARIES}")
endif()

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/WebPConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/WebPConfig.cmake
  INSTALL_DESTINATION ${ConfigPackageLocation}
  PATH_VARS CMAKE_INSTALL_INCLUDEDIR)

# Install the generated CMake files.
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/WebPConfigVersion.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/WebPConfig.cmake"
        DESTINATION ${ConfigPackageLocation})

# Install the man pages.
set(MAN_PAGES
    cwebp.1
    dwebp.1
    gif2webp.1
    img2webp.1
    vwebp.1
    webpmux.1
    webpinfo.1)
set(EXEC_BUILDS
    "CWEBP"
    "DWEBP"
    "GIF2WEBP"
    "IMG2WEBP"
    "VWEBP"
    "WEBPMUX"
    "WEBPINFO")
list(LENGTH MAN_PAGES MAN_PAGES_LENGTH)
math(EXPR MAN_PAGES_RANGE "${MAN_PAGES_LENGTH} - 1")

foreach(I_MAN RANGE ${MAN_PAGES_RANGE})
  list(GET EXEC_BUILDS ${I_MAN} EXEC_BUILD)
  if(WEBP_BUILD_${EXEC_BUILD})
    list(GET MAN_PAGES ${I_MAN} MAN_PAGE)
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/man/${MAN_PAGE}
            DESTINATION ${CMAKE_INSTALL_MANDIR}/man1 COMPONENT doc)
  endif()
endforeach()
