/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMCONFIGURE_H
#define GDCMCONFIGURE_H

/* This header is configured by GDCM's build process.  */

/*--------------------------------------------------------------------------*/
/* Platform Features                                                        */

/* Byte order.  */
/* All compilers that support Mac OS X define either __BIG_ENDIAN__ or
   __LITTLE_ENDIAN__ to match the endianness of the architecture being
   compiled for. This is not necessarily the same as the architecture of the
   machine doing the building. In order to support Universal Binaries on
   Mac OS X, we prefer those defines to decide the endianness.
   Elsewhere use the platform check result.  */
#if !defined(__APPLE__)
 #cmakedefine GDCM_WORDS_BIGENDIAN
#elif defined(__BIG_ENDIAN__)
# define GDCM_WORDS_BIGENDIAN
#endif

/* Allow access to UINT32_MAX , cf gdcmCommon.h */
#define __STDC_LIMIT_MACROS

/* Hard code the path to the public dictionary */
#define PUB_DICT_PATH "@GDCM_PUB_DICT_PATH@"

/* Usefull in particular for loadshared where the full path
 * to the lib is needed */
#define GDCM_SOURCE_DIR "@GDCM_SOURCE_DIR@"
#define GDCM_EXECUTABLE_OUTPUT_PATH "@EXECUTABLE_OUTPUT_PATH@"
#define GDCM_LIBRARY_OUTPUT_PATH    "@LIBRARY_OUTPUT_PATH@"

#cmakedefine GDCM_BUILD_TESTING

#cmakedefine GDCM_USE_SYSTEM_ZLIB
#cmakedefine GDCM_USE_SYSTEM_UUID
#cmakedefine GDCM_USE_SYSTEM_POPPLER
#cmakedefine GDCM_USE_SYSTEM_LIBXML2
#cmakedefine GDCM_USE_SYSTEM_OPENSSL
#cmakedefine GDCM_USE_SYSTEM_MD5
#cmakedefine GDCM_USE_SYSTEM_EXPAT
#cmakedefine GDCM_USE_SYSTEM_JSON
#cmakedefine GDCM_USE_SYSTEM_LJPEG
#cmakedefine GDCM_USE_SYSTEM_OPENJPEG
#cmakedefine GDCM_USE_OPENJPEG_V2
#cmakedefine GDCM_USE_SYSTEM_CHARLS
#cmakedefine GDCM_USE_SYSTEM_KAKADU
#cmakedefine GDCM_USE_SYSTEM_PVRG
#cmakedefine GDCMV2_0_COMPATIBILITY
#cmakedefine GDCM_USE_SYSTEM_PAPYRUS3

#ifndef GDCM_USE_SYSTEM_OPENJPEG
#ifdef GDCM_USE_OPENJPEG_V2
#define OPENJPEG_MAJOR_VERSION 2
#else // GDCM_USE_OPENJPEG_V2
#define OPENJPEG_MAJOR_VERSION 1
#endif // GDCM_USE_OPENJPEG_V2
#else
#define OPENJPEG_MAJOR_VERSION @OPENJPEG_MAJOR_VERSION@
#endif //GDCM_USE_SYSTEM_OPENJPEG

#ifndef OPENJPEG_MAJOR_VERSION
#error problem with openjpeg major version
#endif // OPENJPEG_MAJOR_VERSION

#cmakedefine GDCM_USE_PVRG
#cmakedefine GDCM_USE_KAKADU
#cmakedefine GDCM_USE_JPEGLS

#cmakedefine GDCM_AUTOLOAD_GDCMJNI

/* I guess something important */
#cmakedefine GDCM_HAVE_STDINT_H
#cmakedefine GDCM_HAVE_INTTYPES_H

/* This variable allows you to have helpful debug statement */
/* That are in between #ifdef / endif in the gdcm code */
/* That means if GDCM_DEBUG is OFF there shouldn't be any 'cout' at all ! */
/* only cerr, for instance 'invalid file' will be allowed */
#cmakedefine GDCM_DEBUG

#define GDCM_CMAKE_INSTALL_PREFIX "@CMAKE_INSTALL_PREFIX@"
#define GDCM_INSTALL_INCLUDE_DIR "@GDCM_INSTALL_INCLUDE_DIR@"
#define GDCM_INSTALL_DATA_DIR "@GDCM_INSTALL_DATA_DIR@"

/* Whether we are building shared libraries.  */
#cmakedefine GDCM_BUILD_SHARED_LIBS

/* GDCM uses __FUNCTION__ which is not ANSI C, but C99 */
#cmakedefine GDCM_CXX_HAS_FUNCTION

/* Special time structure support */
#cmakedefine GDCM_HAVE_SYS_TIME_H
#cmakedefine GDCM_HAVE_WINSOCK_H
#cmakedefine GDCM_HAVE_BYTESWAP_H
#cmakedefine GDCM_HAVE_RPC_H
// CMS with PBE (added in OpenSSL 1.0.0 ~ Fri Nov 27 15:33:25 CET 2009)
#cmakedefine GDCM_HAVE_CMS_RECIPIENT_PASSWORD
#cmakedefine GDCM_HAVE_LANGINFO_H
#cmakedefine GDCM_HAVE_NL_LANGINFO

#cmakedefine GDCM_HAVE_STRCASECMP
#cmakedefine GDCM_HAVE_STRNCASECMP
#cmakedefine GDCM_HAVE_SNPRINTF
#cmakedefine GDCM_HAVE_STRPTIME
#cmakedefine GDCM_HAVE__STRICMP
#cmakedefine GDCM_HAVE__STRNICMP
#cmakedefine GDCM_HAVE__SNPRINTF
#cmakedefine GDCM_HAVE_GETTIMEOFDAY

// MM: I have a feeling that if GDCM_HAVE_WCHAR_IFSTREAM, then UNICODE filename
// are expected to be specified as UTF-16, but if no API exist for UTF-16
// at std::ifstream level, then const char* of std::ifstream might accept
// UTF-8
#cmakedefine GDCM_HAVE_WCHAR_IFSTREAM

#cmakedefine GDCM_FORCE_BIGENDIAN_EMULATION

/* To Remove code that support broken DICOM implementation and therefore
 * add some over head. Turn Off at your own risk
 */
#cmakedefine GDCM_SUPPORT_BROKEN_IMPLEMENTATION

#define GDCM_PVRG_JPEG_EXECUTABLE "@PVRGJPEG_EXECUTABLE@"
#define GDCM_KAKADU_EXPAND_EXECUTABLE "@KDU_EXPAND_EXECUTABLE@"

/*--------------------------------------------------------------------------*/
/* GDCM Versioning                                                          */

/* Version number.  */
#define GDCM_MAJOR_VERSION @GDCM_MAJOR_VERSION@
#define GDCM_MINOR_VERSION @GDCM_MINOR_VERSION@
#define GDCM_BUILD_VERSION @GDCM_BUILD_VERSION@
#define GDCM_VERSION "@GDCM_VERSION@"
#define GDCM_API_VERSION "@GDCM_API_VERSION@"

/*
#define GDCM_FILE_META_INFORMATION_VERSION "\0\1"
// echo "gdcm" | od -b
#define GDCM_IMPLEMENTATION_CLASS_UID "107.104.103.115";
#define GDCM_IMPLEMENTATION_VERSION_NAME "GDCM " GDCM_VERSION
#define GDCM_SOURCE_APPLICATION_ENTITY_TITLE "GDCM"
*/


/*--------------------------------------------------------------------------*/
/* GDCM deprecation mechanism                                               */
#cmakedefine GDCM_LEGACY_REMOVE
#cmakedefine GDCM_LEGACY_SILENT

#cmakedefine GDCM_ALWAYS_TRACE_MACRO

#endif
