// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CORE_EXCEPTION_HPP
#define OPENCV_CORE_EXCEPTION_HPP

#include "opencv2/core/cvdef.h"
#include "opencv2/core/cvstd.hpp"

namespace cv {

//! @addtogroup core_utils
//! @{

namespace Error {
//! error codes
enum Code {
    StsOk=                       0,  //!< everything is ok
    StsBackTrace=               -1,  //!< pseudo error for back trace
    StsError=                   -2,  //!< unknown /unspecified error
    StsInternal=                -3,  //!< internal error (bad state)
    StsNoMem=                   -4,  //!< insufficient memory
    StsBadArg=                  -5,  //!< function arg/param is bad
    StsBadFunc=                 -6,  //!< unsupported function
    StsNoConv=                  -7,  //!< iteration didn't converge
    StsAutoTrace=               -8,  //!< tracing
    HeaderIsNull=               -9,  //!< image header is NULL
    BadImageSize=              -10,  //!< image size is invalid
    BadOffset=                 -11,  //!< offset is invalid
    BadDataPtr=                -12,  //!<
    BadStep=                   -13,  //!< image step is wrong, this may happen for a non-continuous matrix.
    BadModelOrChSeq=           -14,  //!<
    BadNumChannels=            -15,  //!< bad number of channels, for example, some functions accept only single channel matrices.
    BadNumChannel1U=           -16,  //!<
    BadDepth=                  -17,  //!< input image depth is not supported by the function
    BadAlphaChannel=           -18,  //!<
    BadOrder=                  -19,  //!< number of dimensions is out of range
    BadOrigin=                 -20,  //!< incorrect input origin
    BadAlign=                  -21,  //!< incorrect input align
    BadCallBack=               -22,  //!<
    BadTileSize=               -23,  //!<
    BadCOI=                    -24,  //!< input COI is not supported
    BadROISize=                -25,  //!< incorrect input roi
    MaskIsTiled=               -26,  //!<
    StsNullPtr=                -27,  //!< null pointer
    StsVecLengthErr=           -28,  //!< incorrect vector length
    StsFilterStructContentErr= -29,  //!< incorrect filter structure content
    StsKernelStructContentErr= -30,  //!< incorrect transform kernel content
    StsFilterOffsetErr=        -31,  //!< incorrect filter offset value
    StsBadSize=                -201, //!< the input/output structure size is incorrect
    StsDivByZero=              -202, //!< division by zero
    StsInplaceNotSupported=    -203, //!< in-place operation is not supported
    StsObjectNotFound=         -204, //!< request can't be completed
    StsUnmatchedFormats=       -205, //!< formats of input/output arrays differ
    StsBadFlag=                -206, //!< flag is wrong or not supported
    StsBadPoint=               -207, //!< bad CvPoint
    StsBadMask=                -208, //!< bad format of mask (neither 8uC1 nor 8sC1)
    StsUnmatchedSizes=         -209, //!< sizes of input/output structures do not match
    StsUnsupportedFormat=      -210, //!< the data format/type is not supported by the function
    StsOutOfRange=             -211, //!< some of parameters are out of range
    StsParseError=             -212, //!< invalid syntax/structure of the parsed file
    StsNotImplemented=         -213, //!< the requested function/feature is not implemented
    StsBadMemBlock=            -214, //!< an allocated block has been corrupted
    StsAssert=                 -215, //!< assertion failed
    GpuNotSupported=           -216, //!< no CUDA support
    GpuApiCallError=           -217, //!< GPU API call error
    OpenGlNotSupported=        -218, //!< no OpenGL support
    OpenGlApiCallError=        -219, //!< OpenGL API call error
    OpenCLApiCallError=        -220, //!< OpenCL API call error
    OpenCLDoubleNotSupported=  -221,
    OpenCLInitError=           -222, //!< OpenCL initialization error
    OpenCLNoAMDBlasFft=        -223
};

} // Error::

/*! @brief Class passed to an error.

This class encapsulates all or almost all necessary
information about the error happened in the program. The exception is
usually constructed and thrown implicitly via CV_Error and CV_Error_ macros.
@see error
 */
class CV_EXPORTS Exception : public std::exception
{
public:
    /*!
     Default constructor
     */
    Exception();
    /*!
     Full constructor. Normally the constructor is not called explicitly.
     Instead, the macros CV_Error(), CV_Error_() and CV_Assert() are used.
    */
    Exception(Error::Code _code, const std::string& _err, const std::string& _func, const std::string& _file, int _line);
    virtual ~Exception() CV_NOEXCEPT;

    /*!
     \return the error description and the context as a text string.
    */
    virtual const char *what() const CV_NOEXCEPT CV_OVERRIDE;
    void formatMessage();
    const char * codeMessage() const;

    std::string msg; ///< the formatted error message

    Error::Code code; ///< error code @see CVStatus
    std::string err; ///< error description
    std::string func; ///< function name. Available only when the compiler supports getting it
    std::string file; ///< source file name where the error has occurred
    int line; ///< line number in the source file where the error has occurred
};

/*! @brief Signals an error and raises the exception.

By default the function prints information about the error to stderr,
then it either stops if cv::setBreakOnError() had been called before or raises the exception.
It is possible to alternate error processing by using #redirectError().
@param exc the exception raisen.
@deprecated drop this version
 */
CV_EXPORTS CV_NORETURN void error(const Exception& exc);

/*! @brief Signals an error and raises the exception.

By default the function prints information about the error to stderr,
then it either stops if setBreakOnError() had been called before or raises the exception.
It is possible to alternate error processing by using redirectError().
@param _code - error code (Error::Code)
@param _err - error description
@param _func - function name. Available only when the compiler supports getting it
@param _file - source file name where the error has occurred
@param _line - line number in the source file where the error has occurred
@see CV_Error, CV_Error_, CV_Assert, CV_DbgAssert
 */
CV_EXPORTS CV_NORETURN void error(Error::Code _code, const String& _err, const char* _func, const char* _file, int _line);

/*! @brief Signals an error and terminate application.

By default the function prints information about the error to stderr, then it terminates application
with std::terminate. The function is designed for invariants check in functions and methods with
noexcept attribute.
@param _code - error code (Error::Code)
@param _err - error description
@param _func - function name. Available only when the compiler supports getting it
@param _file - source file name where the error has occurred
@param _line - line number in the source file where the error has occurred
@see CV_AssertTerminate
 */
CV_EXPORTS CV_NORETURN void terminate(Error::Code _code, const String& _err, const char* _func, const char* _file, int _line) CV_NOEXCEPT;


#ifdef CV_STATIC_ANALYSIS

// In practice, some macro are not processed correctly (noreturn is not detected).
// We need to use simplified definition for them.
#define CV_Error(code, msg) do { (void)(code); (void)(msg); abort(); } while (0)
#define CV_Error_(code, args) do { (void)(code); (void)(cv::format args); abort(); } while (0)
#define CV_Assert( expr ) do { if (!(expr)) abort(); } while (0)

#else // CV_STATIC_ANALYSIS

/** @brief Call the error handler.

Currently, the error handler prints the error code and the error message to the standard
error stream `stderr`. In the Debug configuration, it then provokes memory access violation, so that
the execution stack and all the parameters can be analyzed by the debugger. In the Release
configuration, the exception is thrown.

@param code one of Error::Code
@param msg error message
*/
#define CV_Error( code, msg ) cv::error( code, msg, CV_Func, __FILE__, __LINE__ )

/**  @brief Call the error handler.

This macro can be used to construct an error message on-fly to include some dynamic information,
for example:
@code
    // note the extra parentheses around the formatted text message
    CV_Error_(Error::StsOutOfRange,
    ("the value at (%d, %d)=%g is out of range", badPt.x, badPt.y, badValue));
@endcode
@param code one of Error::Code
@param args printf-like formatted error message in parentheses
*/
#define CV_Error_( code, args ) cv::error( code, cv::format args, CV_Func, __FILE__, __LINE__ )

/** @brief Checks a condition at runtime and throws exception if it fails

The macros CV_Assert (and CV_DbgAssert(expr)) evaluate the specified expression. If it is 0, the macros
raise an error (see cv::error). The macro CV_Assert checks the condition in both Debug and Release
configurations while CV_DbgAssert is only retained in the Debug configuration.
CV_AssertTerminate is analog of CV_Assert for invariants check in functions with noexcept attribute.
It does not throw exception, but terminates the application.
*/
#define CV_Assert( expr ) do { if(!!(expr)) ; else cv::error( cv::Error::StsAssert, #expr, CV_Func, __FILE__, __LINE__ ); } while(0)
#define CV_AssertTerminate( expr ) do { if(!!(expr)) ; else cv::terminate( #expr, CV_Func, __FILE__, __LINE__ ); } while(0)

#endif // CV_STATIC_ANALYSIS

//! @cond IGNORED
#if !defined(__OPENCV_BUILD)  // TODO: backward compatibility only
#ifndef CV_ErrorNoReturn
#define CV_ErrorNoReturn CV_Error
#endif
#ifndef CV_ErrorNoReturn_
#define CV_ErrorNoReturn_ CV_Error_
#endif
#endif

#define CV_Assert_1 CV_Assert
#define CV_Assert_2( expr, ... ) CV_Assert_1(expr); __CV_EXPAND(CV_Assert_1( __VA_ARGS__ ))
#define CV_Assert_3( expr, ... ) CV_Assert_1(expr); __CV_EXPAND(CV_Assert_2( __VA_ARGS__ ))
#define CV_Assert_4( expr, ... ) CV_Assert_1(expr); __CV_EXPAND(CV_Assert_3( __VA_ARGS__ ))
#define CV_Assert_5( expr, ... ) CV_Assert_1(expr); __CV_EXPAND(CV_Assert_4( __VA_ARGS__ ))
#define CV_Assert_6( expr, ... ) CV_Assert_1(expr); __CV_EXPAND(CV_Assert_5( __VA_ARGS__ ))
#define CV_Assert_7( expr, ... ) CV_Assert_1(expr); __CV_EXPAND(CV_Assert_6( __VA_ARGS__ ))
#define CV_Assert_8( expr, ... ) CV_Assert_1(expr); __CV_EXPAND(CV_Assert_7( __VA_ARGS__ ))
#define CV_Assert_9( expr, ... ) CV_Assert_1(expr); __CV_EXPAND(CV_Assert_8( __VA_ARGS__ ))
#define CV_Assert_10( expr, ... ) CV_Assert_1(expr); __CV_EXPAND(CV_Assert_9( __VA_ARGS__ ))

#define CV_Assert_N(...) do { __CV_EXPAND(__CV_CAT(CV_Assert_, __CV_VA_NUM_ARGS(__VA_ARGS__)) (__VA_ARGS__)); } while(0)

//! @endcond

#if defined _DEBUG || defined CV_STATIC_ANALYSIS
#  define CV_DbgAssert(expr) CV_Assert(expr)
#else
/** replaced with CV_Assert(expr) in Debug configuration */
#  define CV_DbgAssert(expr)
#endif

//! @} core_utils

} // cv::

#endif // OPENCV_CORE_EXCEPTION_HPP
