# - Try to find the OpenSSL encryption library
# Once done this will define
#
#  OPENSSL_ROOT_DIR - Set this variable to the root installation of OpenSSL
#
# Read-Only variables:
#  OPENSSL_FOUND - system has the OpenSSL library
#  OPENSSL_INCLUDE_DIR - the OpenSSL include directory
#  OPENSSL_LIBRARIES - The libraries needed to use OpenSSL

#=============================================================================
# Copyright 2006-2009 Kitware, Inc.
# Copyright 2006 Alexander Neundorf <neundorf@kde.org>
# Copyright 2009-2011 Mathieu Malaterre <mathieu.malaterre@gmail.com>
#
# Distributed under the OSI-approved BSD License (the "License");
# see accompanying file Copyright.txt for details.
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#=============================================================================
# (To distributed this file outside of CMake, substitute the full
#  License text for the above reference.)

# http://www.slproweb.com/products/Win32OpenSSL.html
set(_OPENSSL_ROOT_HINTS
  "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\OpenSSL (32-bit)_is1;Inno Setup: App Path]"
  "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\OpenSSL (64-bit)_is1;Inno Setup: App Path]"
  )
set(_OPENSSL_ROOT_PATHS
  "C:/OpenSSL/"
  "C:/OpenSSL-Win32/"
  )
find_path(OPENSSL_ROOT_DIR
  NAMES include/openssl/ssl.h
  HINTS ${_OPENSSL_ROOT_HINTS}
  PATHS ${_OPENSSL_ROOT_PATHS}
)
mark_as_advanced(OPENSSL_ROOT_DIR)

# Re-use the previous path:
find_path(OPENSSL_INCLUDE_DIR openssl/ssl.h
  PATHS ${OPENSSL_ROOT_DIR}/include
)

if(WIN32 AND NOT CYGWIN)
  # MINGW should go here too
  if(MSVC)
    # /MD and /MDd are the standard values - if someone wants to use
    # others, the libnames have to change here too
    # use also ssl and ssleay32 in debug as fallback for openssl < 0.9.8b
    # TODO: handle /MT and static lib
    # In Visual C++ naming convention each of these four kinds of Windows libraries has it's standard suffix:
    #   * MD for dynamic-release
    #   * MDd for dynamic-debug
    #   * MT for static-release
    #   * MTd for static-debug

    # Implementation details:
    # We are using the libraries located in the VC subdir instead of the parent directory eventhough :
    # libeay32MD.lib is identical to ../libeay32.lib, and
    # ssleay32MD.lib is identical to ../ssleay32.lib
    find_library(LIB_EAY_DEBUG NAMES libeay32MDd libeay32
      PATHS ${OPENSSL_ROOT_DIR}/lib/VC
      )
    find_library(LIB_EAY_RELEASE NAMES libeay32MD libeay32
      PATHS ${OPENSSL_ROOT_DIR}/lib/VC
      )
    find_library(SSL_EAY_DEBUG NAMES ssleay32MDd ssleay32 ssl
      PATHS ${OPENSSL_ROOT_DIR}/lib/VC
      )
    find_library(SSL_EAY_RELEASE NAMES ssleay32MD ssleay32 ssl
      PATHS ${OPENSSL_ROOT_DIR}/lib/VC
      )
    if( CMAKE_CONFIGURATION_TYPES OR CMAKE_BUILD_TYPE )
      set( OPENSSL_LIBRARIES
        optimized ${SSL_EAY_RELEASE} ${LIB_EAY_RELEASE}
        debug ${SSL_EAY_DEBUG} ${LIB_EAY_DEBUG}
        )
    else()
      set( OPENSSL_LIBRARIES ${SSL_EAY_RELEASE} ${LIB_EAY_RELEASE} )
    endif()
    mark_as_advanced(SSL_EAY_DEBUG SSL_EAY_RELEASE)
    mark_as_advanced(LIB_EAY_DEBUG LIB_EAY_RELEASE)
  elseif(MINGW)
    # same player, for MingW
    find_library(LIB_EAY NAMES libeay32
      PATHS ${OPENSSL_ROOT_DIR}/lib/MinGW
      )
    find_library(SSL_EAY NAMES ssleay32
      PATHS ${OPENSSL_ROOT_DIR}/lib/MinGW
      )
    mark_as_advanced(SSL_EAY LIB_EAY)
    set( OPENSSL_LIBRARIES ${SSL_EAY} ${LIB_EAY} )
  else()
    # Not sure what to pick for -say- intel, let's use the toplevel ones and hope someone report issues:
    find_library(LIB_EAY NAMES libeay32
      PATHS ${OPENSSL_ROOT_DIR}/lib
      )
    find_library(SSL_EAY NAMES ssleay32
      PATHS ${OPENSSL_ROOT_DIR}/lib
      )
    mark_as_advanced(SSL_EAY LIB_EAY)
    set( OPENSSL_LIBRARIES ${SSL_EAY} ${LIB_EAY} )
  endif()
else()

  find_library(OPENSSL_SSL_LIBRARIES NAMES ssl ssleay32 ssleay32MD)
  find_library(OPENSSL_CRYPTO_LIBRARIES NAMES crypto)
  mark_as_advanced(OPENSSL_CRYPTO_LIBRARIES OPENSSL_SSL_LIBRARIES)

  set(OPENSSL_LIBRARIES ${OPENSSL_SSL_LIBRARIES} ${OPENSSL_CRYPTO_LIBRARIES})

endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenSSL DEFAULT_MSG
  OPENSSL_LIBRARIES
  OPENSSL_INCLUDE_DIR
)

mark_as_advanced(OPENSSL_INCLUDE_DIR OPENSSL_LIBRARIES)
