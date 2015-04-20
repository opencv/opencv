/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMTRACE_H
#define GDCMTRACE_H

#include "gdcmTypes.h"
#include "gdcmSystem.h"

#include <iosfwd>
#include <cassert>

namespace gdcm
{

/**
 * \brief Trace
 * \details Debug / Warning and Error are encapsulated in this class
 * by default the Trace class will redirect any debug/warning/error
 * to std::cerr. Unless SetStream was specified with another (open) stream or
 * SetStreamToFile was specified to a writable file on the system.
 *
 * \warning
 * All string messages are removed during compilation time when compiled with
 * CMAKE_BUILD_TYPE being set to either:
 * - Release
 * - MinSizeRel
 * It is recommended to compile with RelWithDebInfo and/or Debug during
 * prototyping of applications.
 */
class GDCM_EXPORT Trace
{
public :
  Trace();
  ~Trace();

  /// Explicitely set the ostream for gdcm::Trace to report to
  /// This will set the DebugStream, WarningStream and ErrorStream at once:
  static void SetStream(std::ostream &os);
  static std::ostream &GetStream();

  /// Explicitely set the stream which receive Debug messages:
  static void SetDebugStream(std::ostream &os);
  static std::ostream &GetDebugStream();

  /// Explicitely set the stream which receive Warning messages:
  static void SetWarningStream(std::ostream &os);
  static std::ostream &GetWarningStream();

  /// Explicitely set the stream which receive Error messages:
  static void SetErrorStream(std::ostream &os);
  static std::ostream &GetErrorStream();

  /// Explicitely set the filename for gdcm::Trace to report to
  /// The file will be created (it will not append to existing file)
  static void SetStreamToFile( const char *filename );

  /// Turn debug messages on (default: false)
  static void SetDebug(bool debug);
  static void DebugOn();
  static void DebugOff();
  static bool GetDebugFlag();

  /// Turn warning messages on (default: true)
  static void SetWarning(bool debug);
  static void WarningOn();
  static void WarningOff();
  static bool GetWarningFlag();

  /// Turn error messages on (default: true)
  static void SetError(bool debug);
  static void ErrorOn();
  static void ErrorOff();
  static bool GetErrorFlag();

protected:
private:
};

// Here we define function this is the only way to be able to pass
// stuff with indirection like:
// gdcmDebug( "my message:" << i << '\t' );
// You cannot use function unless you use vnsprintf ...

// __FUNCTION is not always defined by preprocessor
// In c++ we should use __PRETTY_FUNCTION__ instead...
#ifdef GDCM_CXX_HAS_FUNCTION
// Handle particular case for GNU C++ which also defines __PRETTY_FUNCTION__
// which is a lot nice in C++
#ifdef __BORLANDC__
#  define __FUNCTION__ __FUNC__
#endif
#ifdef __GNUC__
#  define GDCM_FUNCTION __PRETTY_FUNCTION__
#else
#  define GDCM_FUNCTION __FUNCTION__
#endif //__GNUC__
#else
#  define GDCM_FUNCTION "<unknow>"
#endif //GDCM_CXX_HAS_FUNCTION

/**
 * \brief   Debug
 * @param msg message part
 */
#if defined(NDEBUG) && !defined(GDCM_ALWAYS_TRACE_MACRO)
#define gdcmDebugMacro(msg) {}
#else
#define gdcmDebugMacro(msg)                                       \
{                                                                 \
   if( gdcm::Trace::GetDebugFlag() )                              \
   {                                                              \
   std::ostringstream osmacro;                                    \
   osmacro << "Debug: In " __FILE__ ", line " << __LINE__         \
           << ", function " << GDCM_FUNCTION << '\n'              \
           << "Last system error was: "                           \
           << gdcm::System::GetLastSystemError() << '\n' << msg;  \
   std::ostream &_os = gdcm::Trace::GetDebugStream();             \
   _os << osmacro.str() << "\n\n" << std::endl;                   \
   }                                                              \
}
#endif //NDEBUG

/**
 * \brief   Warning
 * @param msg message part
 */
#if defined(NDEBUG) && !defined(GDCM_ALWAYS_TRACE_MACRO)
#define gdcmWarningMacro(msg) {}
#else
#define gdcmWarningMacro(msg)                                     \
{                                                                 \
   if( gdcm::Trace::GetWarningFlag() )                            \
   {                                                              \
   std::ostringstream osmacro;                                    \
   osmacro << "Warning: In " __FILE__ ", line " << __LINE__       \
           << ", function " << GDCM_FUNCTION << "\n"              \
           << msg << "\n\n";                                      \
   std::ostream &_os = gdcm::Trace::GetWarningStream();           \
   _os << osmacro.str() << std::endl;                             \
   }                                                              \
}
#endif //NDEBUG

/**
 * \brief   Error this is pretty bad, more than just warning
 * It could mean lost of data, something not handle...
 * @param msg second message part
 */
#if defined(NDEBUG) && !defined(GDCM_ALWAYS_TRACE_MACRO)
#define gdcmErrorMacro(msg) {}
#else
#define gdcmErrorMacro(msg)                                       \
{                                                                 \
   if( gdcm::Trace::GetErrorFlag() )                              \
   {                                                              \
   std::ostringstream osmacro;                                    \
   osmacro << "Error: In " __FILE__ ", line " << __LINE__         \
           << ", function " << GDCM_FUNCTION << '\n'              \
           << msg << "\n\n";                                      \
   std::ostream &_os = gdcm::Trace::GetErrorStream();             \
   _os << osmacro.str() << std::endl;                             \
   }                                                              \
}
#endif //NDEBUG

/**
 * \brief   Assert
 * @param arg argument to test
 *        An easy solution to pass also a message is to do:
 *        gdcmAssertMacro( "my message" && 2 < 3 )
 */
#if defined(NDEBUG) && !defined(GDCM_ALWAYS_TRACE_MACRO)
#define gdcmAssertMacro(arg) {}
#else
#define gdcmAssertMacro(arg)                                      \
{                                                                 \
   if( !(arg) )                                                   \
   {                                                              \
   std::ostringstream osmacro;                                    \
   osmacro << "Assert: In " __FILE__ ", line " << __LINE__        \
           << ", function " << GDCM_FUNCTION                      \
           << "\n\n";                                             \
   std::ostream &_os = gdcm::Trace::GetErrorStream();             \
   _os << osmacro.str() << std::endl;                             \
   assert ( arg );                                                \
   }                                                              \
}
#endif //NDEBUG

/**
 * \brief   AssertAlways
 * @param arg argument to test
 *        An easy solution to pass also a message is to do:
 *        gdcmAssertMacro( "my message" && 2 < 3 )
 */
#if defined(NDEBUG) && !defined(GDCM_ALWAYS_TRACE_MACRO)
// User asked for release compilation, but still need to report
// if grave issue.
#define gdcmAssertAlwaysMacro(arg) \
{                                                                 \
   if( !(arg) )                                                   \
   {                                                              \
   std::ostringstream osmacro;                                    \
   osmacro << "Assert: In " __FILE__ ", line " << __LINE__        \
           << ", function " << GDCM_FUNCTION                      \
           << "\n\n";                                             \
   throw osmacro.str();                                           \
   }                                                              \
}
#else
// Simply reproduce gdcmAssertMacro behavior:
#define gdcmAssertAlwaysMacro(arg) gdcmAssertMacro(arg)
#endif //NDEBUG

} // end namespace gdcm
//-----------------------------------------------------------------------------
#endif //GDCMTRACE_H
