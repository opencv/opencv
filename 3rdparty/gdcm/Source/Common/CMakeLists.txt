#-----------------------------------------------------------------------------
# Rebuild gdcm whenever a file starting with gdcm* is modified
include_regular_expression("^gdcm.*$")

# configure the .h file
option(GDCM_ALWAYS_TRACE_MACRO "When set to ON, gdcm::Trace macros will dumps message (override NDEBUG settings)" OFF)
option(GDCM_SUPPORT_BROKEN_IMPLEMENTATION "Handle broken DICOM" ON)
mark_as_advanced(
  GDCM_ALWAYS_TRACE_MACRO
  GDCM_SUPPORT_BROKEN_IMPLEMENTATION
  GDCM_AUTOLOAD_GDCMJNI
  )

#if(WIN32)
#  if (BUILD_SHARED_LIBS)
#    add_definitions(-DPOLARSSL_DLL)
#  endif ()
#endif()


CHECK_INCLUDE_FILE_CONCAT("sys/time.h"     GDCM_HAVE_SYS_TIME_H)
CHECK_INCLUDE_FILE_CONCAT("winsock.h"       GDCM_HAVE_WINSOCK_H)
CHECK_INCLUDE_FILE_CONCAT("byteswap.h"       GDCM_HAVE_BYTESWAP_FILE_H)
# see bug #324
if(GDCM_HAVE_BYTESWAP_FILE_H)
  include(CheckCXXSourceCompiles)
  CHECK_CXX_SOURCE_COMPILES(
    "\#include <byteswap.h>\nint main() { return bswap_32( 42 ); }"
    GDCM_HAVE_BYTESWAP_H)
endif()
CHECK_INCLUDE_FILE("rpc.h"       GDCM_HAVE_RPC_H)
CHECK_INCLUDE_FILE("langinfo.h"       GDCM_HAVE_LANGINFO_H)

include(CheckFunctionExists)
# See http://public.kitware.com/Bug/view.php?id=8246
include(CheckSymbolExists)
CHECK_SYMBOL_EXISTS(nl_langinfo "langinfo.h" GDCM_HAVE_NL_LANGINFO)
#C99
#CHECK_FUNCTION_EXISTS(strcasecmp  GDCM_HAVE_STRCASECMP)
CHECK_SYMBOL_EXISTS(strcasecmp "strings.h" GDCM_HAVE_STRCASECMP)
#CHECK_FUNCTION_EXISTS(strncasecmp GDCM_HAVE_STRNCASECMP)
CHECK_SYMBOL_EXISTS(strncasecmp "strings.h" GDCM_HAVE_STRNCASECMP)
#CHECK_FUNCTION_EXISTS(snprintf    GDCM_HAVE_SNPRINTF)
CHECK_SYMBOL_EXISTS(snprintf "stdio.h"    GDCM_HAVE_SNPRINTF)
#CHECK_SYMBOL_EXISTS(strptime "time.h"    GDCM_HAVE_STRPTIME)
CHECK_FUNCTION_EXISTS(strptime    GDCM_HAVE_STRPTIME)
#M$ extension:
CHECK_FUNCTION_EXISTS(_stricmp    GDCM_HAVE__STRICMP)
CHECK_FUNCTION_EXISTS(_strnicmp   GDCM_HAVE__STRNICMP)
CHECK_FUNCTION_EXISTS(_snprintf   GDCM_HAVE__SNPRINTF)

#include(CheckSymbolExists)
CHECK_FUNCTION_EXISTS(gettimeofday GDCM_HAVE_GETTIMEOFDAY)

include(CheckCXXSourceCompiles)
CHECK_CXX_SOURCE_COMPILES(
  "int main() { const char *f = __FUNCTION__; return 0;}"
  GDCM_CXX_HAS_FUNCTION)
CHECK_CXX_SOURCE_COMPILES(
  "\#include <fstream>\nint main() { const wchar_t fn[10] = {}; std::ifstream is( fn ); return 0;}"
  GDCM_HAVE_WCHAR_IFSTREAM)
if(GDCM_USE_SYSTEM_OPENSSL)
set(CMAKE_REQUIRED_INCLUDES ${OPENSSL_INCLUDE_DIR})
set(CMAKE_REQUIRED_LIBRARIES ${OPENSSL_LIBRARIES}
  ${CMAKE_DL_LIBS} # FIXME ??
  )
CHECK_CXX_SOURCE_COMPILES(
  #  "\#include <openssl/cms.h>\nint main() { CMS_add0_recipient_key(0); return 0;}"
  #HAVE_CMS_RECIPIENT_KEY)
    "\#include <openssl/cms.h>\nint main() { CMS_add0_recipient_password(0,0,0,0,0,0,0); return 0;}"
  GDCM_HAVE_CMS_RECIPIENT_PASSWORD)
endif()

#-----------------------------------------------------------------------------
# Provide compatibility options.
option(GDCM_LEGACY_REMOVE "Remove all legacy code completely." OFF)
option(GDCM_LEGACY_SILENT "Silence all legacy code messages." OFF)
mark_as_advanced(GDCM_LEGACY_REMOVE GDCM_LEGACY_SILENT)


configure_file(
  "${GDCM_SOURCE_DIR}/Source/Common/gdcmConfigure.h.in"
  "${GDCM_BINARY_DIR}/Source/Common/gdcmConfigure.h"
  )

# Add the include paths
include_directories(
  "${GDCM_BINARY_DIR}/Source/Common"
  "${GDCM_SOURCE_DIR}/Source/Common"
  "${GDCM_SOURCE_DIR}/Source/DataStructureAndEncodingDefinition"

  "${GDCM_BINARY_DIR}/Testing/Source/Data"
  "${GDCM_SOURCE_DIR}/Testing/Source/Data"

  "${GDCM_SOURCE_DIR}/Utilities"
  )

if(GDCM_BUILD_TESTING)
include_directories(
  "${GDCM_BINARY_DIR}/Utilities/gdcmmd5"
)
endif()

if(NOT GDCM_USE_SYSTEM_ZLIB)
include_directories(
  "${GDCM_BINARY_DIR}/Utilities/gdcmzlib"
)
endif()
#if(NOT GDCM_USE_SYSTEM_POLARSSL)
#include_directories(
#  "${GDCM_SOURCE_DIR}/Utilities/gdcmpolarssl/include"
#)
#endif()

set(Common_SRCS
  gdcmVersion.cxx
  gdcmRegion.cxx
  gdcmBoxRegion.cxx
  gdcmEvent.cxx
  gdcmDataEvent.cxx
  gdcmProgressEvent.cxx
  gdcmFileNameEvent.cxx
  gdcmCommand.cxx
  gdcmMD5.cxx
  gdcmBase64.cxx
  gdcmSHA1.cxx
  gdcmDummyValueGenerator.cxx
  #gdcmCryptographicMessageSyntax.cxx

  gdcmCryptographicMessageSyntax.cxx
  gdcmCryptoFactory.cxx

  gdcmASN1.cxx
  gdcmObject.cxx
  gdcmSubject.cxx
  gdcmDirectory.cxx
  gdcmTerminal.cxx
  gdcmString.cxx
  gdcmFilename.cxx
  gdcmFilenameGenerator.cxx
  gdcmSwapCode.cxx
  gdcmSystem.cxx
  gdcmTrace.cxx
  gdcmException.cxx
  gdcmDeflateStream.cxx
  gdcmByteSwap.cxx
  gdcmUnpacker12Bits.cxx
  )

if(WIN32)
set(Common_SRCS ${Common_SRCS}
  gdcmCAPICryptographicMessageSyntax.cxx
  gdcmCAPICryptoFactory.cxx
)
endif()

if(GDCM_USE_SYSTEM_OPENSSL)
set(Common_SRCS ${Common_SRCS}
  gdcmOpenSSLP7CryptoFactory.cxx
  gdcmOpenSSLP7CryptographicMessageSyntax.cxx
)
endif()

if(GDCM_USE_SYSTEM_OPENSSL AND GDCM_HAVE_CMS_RECIPIENT_PASSWORD)
set(Common_SRCS ${Common_SRCS}
  gdcmOpenSSLCryptoFactory.cxx
  gdcmOpenSSLCryptographicMessageSyntax.cxx
)
endif()

if(GDCM_BUILD_TESTING)
set(Common_SRCS ${Common_SRCS}
  gdcmTesting.cxx
)
endif()

add_library(gdcmCommon ${Common_SRCS})
set_target_properties(gdcmCommon PROPERTIES ${GDCM_LIBRARY_PROPERTIES} LINK_INTERFACE_LIBRARIES "")

if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(gdcmCommon PROPERTIES FOLDER "3rdparty")
endif()

if(NOT BUILD_SHARED_LIBS)
  ocv_install_target(gdcmCommon EXPORT OpenCVModules ARCHIVE DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev)
endif()

#target_link_libraries(gdcmCommon ${GDCM_ZLIB_LIBRARIES}) # ${GDCM_POLARSSL_LIBRARIES})
if(GDCM_BUILD_TESTING)
target_link_libraries(gdcmCommon gdcmmd5 )
endif()
if(APPLE)
  target_link_libraries(gdcmCommon ${COREFOUNDATION_LIBRARY})
endif()
if(UNIX)
  target_link_libraries(gdcmCommon ${CMAKE_DL_LIBS})
endif()

if(WIN32)
  target_link_libraries(gdcmCommon ws2_32)
  target_link_libraries(gdcmCommon crypt32)
endif()

if(GDCM_USE_SYSTEM_OPENSSL)
  include_directories(${OPENSSL_INCLUDE_DIR})
  target_link_libraries(gdcmCommon ${OPENSSL_LIBRARIES})
endif()

# libs
install_library(gdcmCommon)
# PDB
install_pdb(gdcmCommon)
# include files
install_includes("*.h" "*.txx")

# Install generated files
if(NOT GDCM_INSTALL_NO_DEVELOPMENT)
  install(FILES ${header_files}
    "${GDCM_BINARY_DIR}/Source/Common/gdcmConfigure.h"
    DESTINATION ${GDCM_INSTALL_INCLUDE_DIR} COMPONENT Headers
    )
endif()
