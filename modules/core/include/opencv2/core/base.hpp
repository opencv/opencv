/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Copyright (C) 2014, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_CORE_BASE_HPP
#define OPENCV_CORE_BASE_HPP

#ifndef __cplusplus
#  error base.hpp header must be compiled as C++
#endif

#include "opencv2/opencv_modules.hpp"

#include <climits>
#include <algorithm>

#include "opencv2/core/cvdef.h"
#include "opencv2/core/cvstd.hpp"

namespace cv
{

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
} //Error

//! @} core_utils

//! @addtogroup core_array
//! @{

//! matrix decomposition types
enum DecompTypes {
    /** Gaussian elimination with the optimal pivot element chosen. */
    DECOMP_LU       = 0,
    /** singular value decomposition (SVD) method; the system can be over-defined and/or the matrix
    src1 can be singular */
    DECOMP_SVD      = 1,
    /** eigenvalue decomposition; the matrix src1 must be symmetrical */
    DECOMP_EIG      = 2,
    /** Cholesky \f$LL^T\f$ factorization; the matrix src1 must be symmetrical and positively
    defined */
    DECOMP_CHOLESKY = 3,
    /** QR factorization; the system can be over-defined and/or the matrix src1 can be singular */
    DECOMP_QR       = 4,
    /** while all the previous flags are mutually exclusive, this flag can be used together with
    any of the previous; it means that the normal equations
    \f$\texttt{src1}^T\cdot\texttt{src1}\cdot\texttt{dst}=\texttt{src1}^T\texttt{src2}\f$ are
    solved instead of the original system
    \f$\texttt{src1}\cdot\texttt{dst}=\texttt{src2}\f$ */
    DECOMP_NORMAL   = 16
};

/** norm types

src1 and src2 denote input arrays.
*/

enum NormTypes {
                /**
                \f[
                norm =  \forkthree
                {\|\texttt{src1}\|_{L_{\infty}} =  \max _I | \texttt{src1} (I)|}{if  \(\texttt{normType} = \texttt{NORM_INF}\) }
                {\|\texttt{src1}-\texttt{src2}\|_{L_{\infty}} =  \max _I | \texttt{src1} (I) -  \texttt{src2} (I)|}{if  \(\texttt{normType} = \texttt{NORM_INF}\) }
                {\frac{\|\texttt{src1}-\texttt{src2}\|_{L_{\infty}}    }{\|\texttt{src2}\|_{L_{\infty}} }}{if  \(\texttt{normType} = \texttt{NORM_RELATIVE | NORM_INF}\) }
                \f]
                */
                NORM_INF       = 1,
                /**
                \f[
                norm =  \forkthree
                {\| \texttt{src1} \| _{L_1} =  \sum _I | \texttt{src1} (I)|}{if  \(\texttt{normType} = \texttt{NORM_L1}\)}
                { \| \texttt{src1} - \texttt{src2} \| _{L_1} =  \sum _I | \texttt{src1} (I) -  \texttt{src2} (I)|}{if  \(\texttt{normType} = \texttt{NORM_L1}\) }
                { \frac{\|\texttt{src1}-\texttt{src2}\|_{L_1} }{\|\texttt{src2}\|_{L_1}} }{if  \(\texttt{normType} = \texttt{NORM_RELATIVE | NORM_L1}\) }
                \f]*/
                 NORM_L1        = 2,
                 /**
                 \f[
                 norm =  \forkthree
                 { \| \texttt{src1} \| _{L_2} =  \sqrt{\sum_I \texttt{src1}(I)^2} }{if  \(\texttt{normType} = \texttt{NORM_L2}\) }
                 { \| \texttt{src1} - \texttt{src2} \| _{L_2} =  \sqrt{\sum_I (\texttt{src1}(I) - \texttt{src2}(I))^2} }{if  \(\texttt{normType} = \texttt{NORM_L2}\) }
                 { \frac{\|\texttt{src1}-\texttt{src2}\|_{L_2} }{\|\texttt{src2}\|_{L_2}} }{if  \(\texttt{normType} = \texttt{NORM_RELATIVE | NORM_L2}\) }
                 \f]
                 */
                 NORM_L2        = 4,
                 /**
                 \f[
                 norm =  \forkthree
                 { \| \texttt{src1} \| _{L_2} ^{2} = \sum_I \texttt{src1}(I)^2} {if  \(\texttt{normType} = \texttt{NORM_L2SQR}\)}
                 { \| \texttt{src1} - \texttt{src2} \| _{L_2} ^{2} =  \sum_I (\texttt{src1}(I) - \texttt{src2}(I))^2 }{if  \(\texttt{normType} = \texttt{NORM_L2SQR}\) }
                 { \left(\frac{\|\texttt{src1}-\texttt{src2}\|_{L_2} }{\|\texttt{src2}\|_{L_2}}\right)^2 }{if  \(\texttt{normType} = \texttt{NORM_RELATIVE | NORM_L2SQR}\) }
                 \f]
                 */
                 NORM_L2SQR     = 5,
                 /**
                 In the case of one input array, calculates the Hamming distance of the array from zero,
                 In the case of two input arrays, calculates the Hamming distance between the arrays.
                 */
                 NORM_HAMMING   = 6,
                 /**
                 Similar to NORM_HAMMING, but in the calculation, each two bits of the input sequence will
                 be added and treated as a single bit to be used in the same calculation as NORM_HAMMING.
                 */
                 NORM_HAMMING2  = 7,
                 NORM_TYPE_MASK = 7, //!< bit-mask which can be used to separate norm type from norm flags
                 NORM_RELATIVE  = 8, //!< flag
                 NORM_MINMAX    = 32 //!< flag
               };

//! comparison types
enum CmpTypes { CMP_EQ = 0, //!< src1 is equal to src2.
                CMP_GT = 1, //!< src1 is greater than src2.
                CMP_GE = 2, //!< src1 is greater than or equal to src2.
                CMP_LT = 3, //!< src1 is less than src2.
                CMP_LE = 4, //!< src1 is less than or equal to src2.
                CMP_NE = 5  //!< src1 is unequal to src2.
              };

//! generalized matrix multiplication flags
enum GemmFlags { GEMM_1_T = 1, //!< transposes src1
                 GEMM_2_T = 2, //!< transposes src2
                 GEMM_3_T = 4 //!< transposes src3
               };

enum DftFlags {
    /** performs an inverse 1D or 2D transform instead of the default forward
        transform. */
    DFT_INVERSE        = 1,
    /** scales the result: divide it by the number of array elements. Normally, it is
        combined with DFT_INVERSE. */
    DFT_SCALE          = 2,
    /** performs a forward or inverse transform of every individual row of the input
        matrix; this flag enables you to transform multiple vectors simultaneously and can be used to
        decrease the overhead (which is sometimes several times larger than the processing itself) to
        perform 3D and higher-dimensional transformations and so forth.*/
    DFT_ROWS           = 4,
    /** performs a forward transformation of 1D or 2D real array; the result,
        though being a complex array, has complex-conjugate symmetry (*CCS*, see the function
        description below for details), and such an array can be packed into a real array of the same
        size as input, which is the fastest option and which is what the function does by default;
        however, you may wish to get a full complex array (for simpler spectrum analysis, and so on) -
        pass the flag to enable the function to produce a full-size complex output array. */
    DFT_COMPLEX_OUTPUT = 16,
    /** performs an inverse transformation of a 1D or 2D complex array; the
        result is normally a complex array of the same size, however, if the input array has
        conjugate-complex symmetry (for example, it is a result of forward transformation with
        DFT_COMPLEX_OUTPUT flag), the output is a real array; while the function itself does not
        check whether the input is symmetrical or not, you can pass the flag and then the function
        will assume the symmetry and produce the real output array (note that when the input is packed
        into a real array and inverse transformation is executed, the function treats the input as a
        packed complex-conjugate symmetrical array, and the output will also be a real array). */
    DFT_REAL_OUTPUT    = 32,
    /** specifies that input is complex input. If this flag is set, the input must have 2 channels.
        On the other hand, for backwards compatibility reason, if input has 2 channels, input is
        already considered complex. */
    DFT_COMPLEX_INPUT  = 64,
    /** performs an inverse 1D or 2D transform instead of the default forward transform. */
    DCT_INVERSE        = DFT_INVERSE,
    /** performs a forward or inverse transform of every individual row of the input
        matrix. This flag enables you to transform multiple vectors simultaneously and can be used to
        decrease the overhead (which is sometimes several times larger than the processing itself) to
        perform 3D and higher-dimensional transforms and so forth.*/
    DCT_ROWS           = DFT_ROWS
};

//! Various border types, image boundaries are denoted with `|`
//! @see borderInterpolate, copyMakeBorder
enum BorderTypes {
    BORDER_CONSTANT    = 0, //!< `iiiiii|abcdefgh|iiiiiii`  with some specified `i`
    BORDER_REPLICATE   = 1, //!< `aaaaaa|abcdefgh|hhhhhhh`
    BORDER_REFLECT     = 2, //!< `fedcba|abcdefgh|hgfedcb`
    BORDER_WRAP        = 3, //!< `cdefgh|abcdefgh|abcdefg`
    BORDER_REFLECT_101 = 4, //!< `gfedcb|abcdefgh|gfedcba`
    BORDER_TRANSPARENT = 5, //!< `uvwxyz|abcdefgh|ijklmno` - Treats outliers as transparent.

    BORDER_REFLECT101  = BORDER_REFLECT_101, //!< same as BORDER_REFLECT_101
    BORDER_DEFAULT     = BORDER_REFLECT_101, //!< same as BORDER_REFLECT_101
    BORDER_ISOLATED    = 16 //!< Interpolation restricted within the ROI boundaries.
};

//! @} core_array

//! @addtogroup core_utils
//! @{

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
CV_EXPORTS CV_NORETURN void error(int _code, const String& _err, const char* _func, const char* _file, int _line);

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
*/
#define CV_Assert( expr ) do { if(!!(expr)) ; else cv::error( cv::Error::StsAssert, #expr, CV_Func, __FILE__, __LINE__ ); } while(0)

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

/*
 * Hamming distance functor - counts the bit differences between two strings - useful for the Brief descriptor
 * bit count of A exclusive XOR'ed with B
 */
struct CV_EXPORTS Hamming
{
    static const NormTypes normType = NORM_HAMMING;
    typedef unsigned char ValueType;
    typedef int ResultType;

    /** this will count the bits in a ^ b
     */
    ResultType operator()( const unsigned char* a, const unsigned char* b, int size ) const;
};

typedef Hamming HammingLUT;

/////////////////////////////////// inline norms ////////////////////////////////////

template<typename _Tp> inline _Tp cv_abs(_Tp x) { return std::abs(x); }
inline int cv_abs(uchar x) { return x; }
inline int cv_abs(schar x) { return std::abs(x); }
inline int cv_abs(ushort x) { return x; }
inline int cv_abs(short x) { return std::abs(x); }

template<typename _Tp, typename _AccTp> static inline
_AccTp normL2Sqr(const _Tp* a, int n)
{
    _AccTp s = 0;
    int i=0;
#if CV_ENABLE_UNROLLED
    for( ; i <= n - 4; i += 4 )
    {
        _AccTp v0 = a[i], v1 = a[i+1], v2 = a[i+2], v3 = a[i+3];
        s += v0*v0 + v1*v1 + v2*v2 + v3*v3;
    }
#endif
    for( ; i < n; i++ )
    {
        _AccTp v = a[i];
        s += v*v;
    }
    return s;
}

template<typename _Tp, typename _AccTp> static inline
_AccTp normL1(const _Tp* a, int n)
{
    _AccTp s = 0;
    int i = 0;
#if CV_ENABLE_UNROLLED
    for(; i <= n - 4; i += 4 )
    {
        s += (_AccTp)cv_abs(a[i]) + (_AccTp)cv_abs(a[i+1]) +
            (_AccTp)cv_abs(a[i+2]) + (_AccTp)cv_abs(a[i+3]);
    }
#endif
    for( ; i < n; i++ )
        s += cv_abs(a[i]);
    return s;
}

template<typename _Tp, typename _AccTp> static inline
_AccTp normInf(const _Tp* a, int n)
{
    _AccTp s = 0;
    for( int i = 0; i < n; i++ )
        s = std::max(s, (_AccTp)cv_abs(a[i]));
    return s;
}

template<typename _Tp, typename _AccTp> static inline
_AccTp normL2Sqr(const _Tp* a, const _Tp* b, int n)
{
    _AccTp s = 0;
    int i= 0;
#if CV_ENABLE_UNROLLED
    for(; i <= n - 4; i += 4 )
    {
        _AccTp v0 = _AccTp(a[i] - b[i]), v1 = _AccTp(a[i+1] - b[i+1]), v2 = _AccTp(a[i+2] - b[i+2]), v3 = _AccTp(a[i+3] - b[i+3]);
        s += v0*v0 + v1*v1 + v2*v2 + v3*v3;
    }
#endif
    for( ; i < n; i++ )
    {
        _AccTp v = _AccTp(a[i] - b[i]);
        s += v*v;
    }
    return s;
}

static inline float normL2Sqr(const float* a, const float* b, int n)
{
    float s = 0.f;
    for( int i = 0; i < n; i++ )
    {
        float v = a[i] - b[i];
        s += v*v;
    }
    return s;
}

template<typename _Tp, typename _AccTp> static inline
_AccTp normL1(const _Tp* a, const _Tp* b, int n)
{
    _AccTp s = 0;
    int i= 0;
#if CV_ENABLE_UNROLLED
    for(; i <= n - 4; i += 4 )
    {
        _AccTp v0 = _AccTp(a[i] - b[i]), v1 = _AccTp(a[i+1] - b[i+1]), v2 = _AccTp(a[i+2] - b[i+2]), v3 = _AccTp(a[i+3] - b[i+3]);
        s += std::abs(v0) + std::abs(v1) + std::abs(v2) + std::abs(v3);
    }
#endif
    for( ; i < n; i++ )
    {
        _AccTp v = _AccTp(a[i] - b[i]);
        s += std::abs(v);
    }
    return s;
}

inline float normL1(const float* a, const float* b, int n)
{
    float s = 0.f;
    for( int i = 0; i < n; i++ )
    {
        s += std::abs(a[i] - b[i]);
    }
    return s;
}

inline int normL1(const uchar* a, const uchar* b, int n)
{
    int s = 0;
    for( int i = 0; i < n; i++ )
    {
        s += std::abs(a[i] - b[i]);
    }
    return s;
}

template<typename _Tp, typename _AccTp> static inline
_AccTp normInf(const _Tp* a, const _Tp* b, int n)
{
    _AccTp s = 0;
    for( int i = 0; i < n; i++ )
    {
        _AccTp v0 = a[i] - b[i];
        s = std::max(s, std::abs(v0));
    }
    return s;
}

/** @brief Computes the cube root of an argument.

 The function cubeRoot computes \f$\sqrt[3]{\texttt{val}}\f$. Negative arguments are handled correctly.
 NaN and Inf are not handled. The accuracy approaches the maximum possible accuracy for
 single-precision data.
 @param val A function argument.
 */
CV_EXPORTS_W float cubeRoot(float val);

/** @overload

cubeRoot with argument of `double` type calls `std::cbrt(double)`
*/
static inline
double cubeRoot(double val)
{
    return std::cbrt(val);
}

/** @brief Calculates the angle of a 2D vector in degrees.

 The function fastAtan2 calculates the full-range angle of an input 2D vector. The angle is measured
 in degrees and varies from 0 to 360 degrees. The accuracy is about 0.3 degrees.
 @param x x-coordinate of the vector.
 @param y y-coordinate of the vector.
 */
CV_EXPORTS_W float fastAtan2(float y, float x);

/** proxy for hal::LU */
CV_EXPORTS int LU(float* A, size_t astep, int m, float* b, size_t bstep, int n);
/** proxy for hal::LU */
CV_EXPORTS int LU(double* A, size_t astep, int m, double* b, size_t bstep, int n);
/** proxy for hal::Cholesky */
CV_EXPORTS bool Cholesky(float* A, size_t astep, int m, float* b, size_t bstep, int n);
/** proxy for hal::Cholesky */
CV_EXPORTS bool Cholesky(double* A, size_t astep, int m, double* b, size_t bstep, int n);

////////////////// forward declarations for important OpenCV types //////////////////

//! @cond IGNORED

template<typename _Tp, int cn> class Vec;
template<typename _Tp, int m, int n> class Matx;

template<typename _Tp> class Complex;
template<typename _Tp> class Point_;
template<typename _Tp> class Point3_;
template<typename _Tp> class Size_;
template<typename _Tp> class Rect_;
template<typename _Tp> class Scalar_;

class CV_EXPORTS RotatedRect;
class CV_EXPORTS Range;
class CV_EXPORTS TermCriteria;
class CV_EXPORTS KeyPoint;
class CV_EXPORTS DMatch;
class CV_EXPORTS RNG;

class CV_EXPORTS Mat;
class CV_EXPORTS MatExpr;

class CV_EXPORTS UMat;

class CV_EXPORTS SparseMat;
typedef Mat MatND;

template<typename _Tp> class Mat_;
template<typename _Tp> class SparseMat_;

class CV_EXPORTS MatConstIterator;
class CV_EXPORTS SparseMatIterator;
class CV_EXPORTS SparseMatConstIterator;
template<typename _Tp> class MatIterator_;
template<typename _Tp> class MatConstIterator_;
template<typename _Tp> class SparseMatIterator_;
template<typename _Tp> class SparseMatConstIterator_;

namespace ogl
{
    class CV_EXPORTS Buffer;
    class CV_EXPORTS Texture2D;
    class CV_EXPORTS Arrays;
}

namespace cuda
{
    class CV_EXPORTS GpuMat;
    class CV_EXPORTS HostMem;
    class CV_EXPORTS Stream;
    class CV_EXPORTS Event;
}

namespace cudev
{
    template <typename _Tp> class GpuMat_;
}

namespace ipp
{
CV_EXPORTS   unsigned long long getIppFeatures();
CV_EXPORTS   void setIppStatus(int status, const char * const funcname = NULL, const char * const filename = NULL,
                             int line = 0);
CV_EXPORTS   int getIppStatus();
CV_EXPORTS   String getIppErrorLocation();
CV_EXPORTS_W bool   useIPP();
CV_EXPORTS_W void   setUseIPP(bool flag);
CV_EXPORTS_W String getIppVersion();

// IPP Not-Exact mode. This function may force use of IPP then both IPP and OpenCV provide proper results
// but have internal accuracy differences which have too much direct or indirect impact on accuracy tests.
CV_EXPORTS_W bool useIPP_NotExact();
CV_EXPORTS_W void setUseIPP_NotExact(bool flag);
#ifndef DISABLE_OPENCV_3_COMPATIBILITY
static inline bool useIPP_NE() { return useIPP_NotExact(); }
static inline void setUseIPP_NE(bool flag) { setUseIPP_NotExact(flag); }
#endif

} // ipp

//! @endcond

//! @} core_utils




} // cv

#include "opencv2/core/neon_utils.hpp"
#include "opencv2/core/vsx_utils.hpp"
#include "opencv2/core/check.hpp"

#endif //OPENCV_CORE_BASE_HPP
