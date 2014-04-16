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

#ifndef __OPENCV_CORE_BASE_HPP__
#define __OPENCV_CORE_BASE_HPP__

#ifndef __cplusplus
#  error base.hpp header must be compiled as C++
#endif

#include <climits>

#include "opencv2/core/cvdef.h"
#include "opencv2/core/cvstd.hpp"

namespace cv
{

// error codes
namespace Error {
enum {
    StsOk=                       0,  /* everithing is ok                */
    StsBackTrace=               -1,  /* pseudo error for back trace     */
    StsError=                   -2,  /* unknown /unspecified error      */
    StsInternal=                -3,  /* internal error (bad state)      */
    StsNoMem=                   -4,  /* insufficient memory             */
    StsBadArg=                  -5,  /* function arg/param is bad       */
    StsBadFunc=                 -6,  /* unsupported function            */
    StsNoConv=                  -7,  /* iter. didn't converge           */
    StsAutoTrace=               -8,  /* tracing                         */
    HeaderIsNull=               -9,  /* image header is NULL            */
    BadImageSize=              -10,  /* image size is invalid           */
    BadOffset=                 -11,  /* offset is invalid               */
    BadDataPtr=                -12,  /**/
    BadStep=                   -13,  /**/
    BadModelOrChSeq=           -14,  /**/
    BadNumChannels=            -15,  /**/
    BadNumChannel1U=           -16,  /**/
    BadDepth=                  -17,  /**/
    BadAlphaChannel=           -18,  /**/
    BadOrder=                  -19,  /**/
    BadOrigin=                 -20,  /**/
    BadAlign=                  -21,  /**/
    BadCallBack=               -22,  /**/
    BadTileSize=               -23,  /**/
    BadCOI=                    -24,  /**/
    BadROISize=                -25,  /**/
    MaskIsTiled=               -26,  /**/
    StsNullPtr=                -27,  /* null pointer */
    StsVecLengthErr=           -28,  /* incorrect vector length */
    StsFilterStructContentErr= -29,  /* incorr. filter structure content */
    StsKernelStructContentErr= -30,  /* incorr. transform kernel content */
    StsFilterOffsetErr=        -31,  /* incorrect filter ofset value */
    StsBadSize=                -201, /* the input/output structure size is incorrect  */
    StsDivByZero=              -202, /* division by zero */
    StsInplaceNotSupported=    -203, /* in-place operation is not supported */
    StsObjectNotFound=         -204, /* request can't be completed */
    StsUnmatchedFormats=       -205, /* formats of input/output arrays differ */
    StsBadFlag=                -206, /* flag is wrong or not supported */
    StsBadPoint=               -207, /* bad CvPoint */
    StsBadMask=                -208, /* bad format of mask (neither 8uC1 nor 8sC1)*/
    StsUnmatchedSizes=         -209, /* sizes of input/output structures do not match */
    StsUnsupportedFormat=      -210, /* the data format/type is not supported by the function*/
    StsOutOfRange=             -211, /* some of parameters are out of range */
    StsParseError=             -212, /* invalid syntax/structure of the parsed file */
    StsNotImplemented=         -213, /* the requested function/feature is not implemented */
    StsBadMemBlock=            -214, /* an allocated block has been corrupted */
    StsAssert=                 -215, /* assertion failed */
    GpuNotSupported=           -216,
    GpuApiCallError=           -217,
    OpenGlNotSupported=        -218,
    OpenGlApiCallError=        -219,
    OpenCLApiCallError=        -220,
    OpenCLDoubleNotSupported=  -221,
    OpenCLInitError=           -222,
    OpenCLNoAMDBlasFft=        -223
};
} //Error

// matrix decomposition types
enum { DECOMP_LU       = 0,
       DECOMP_SVD      = 1,
       DECOMP_EIG      = 2,
       DECOMP_CHOLESKY = 3,
       DECOMP_QR       = 4,
       DECOMP_NORMAL   = 16
     };

// norm types
enum { NORM_INF       = 1,
       NORM_L1        = 2,
       NORM_L2        = 4,
       NORM_L2SQR     = 5,
       NORM_HAMMING   = 6,
       NORM_HAMMING2  = 7,
       NORM_TYPE_MASK = 7,
       NORM_RELATIVE  = 8,
       NORM_MINMAX    = 32
     };

// comparison types
enum { CMP_EQ = 0,
       CMP_GT = 1,
       CMP_GE = 2,
       CMP_LT = 3,
       CMP_LE = 4,
       CMP_NE = 5
     };

enum { GEMM_1_T = 1,
       GEMM_2_T = 2,
       GEMM_3_T = 4
     };

enum { DFT_INVERSE        = 1,
       DFT_SCALE          = 2,
       DFT_ROWS           = 4,
       DFT_COMPLEX_OUTPUT = 16,
       DFT_REAL_OUTPUT    = 32,
       DCT_INVERSE        = DFT_INVERSE,
       DCT_ROWS           = DFT_ROWS
     };

//! Various border types, image boundaries are denoted with '|'
enum {
       BORDER_CONSTANT    = 0, // iiiiii|abcdefgh|iiiiiii  with some specified 'i'
       BORDER_REPLICATE   = 1, // aaaaaa|abcdefgh|hhhhhhh
       BORDER_REFLECT     = 2, // fedcba|abcdefgh|hgfedcb
       BORDER_WRAP        = 3, // cdefgh|abcdefgh|abcdefg
       BORDER_REFLECT_101 = 4, // gfedcb|abcdefgh|gfedcba
       BORDER_TRANSPARENT = 5, // uvwxyz|absdefgh|ijklmno

       BORDER_REFLECT101  = BORDER_REFLECT_101,
       BORDER_DEFAULT     = BORDER_REFLECT_101,
       BORDER_ISOLATED    = 16 // do not look outside of ROI
     };



//////////////// static assert /////////////////

#define CVAUX_CONCAT_EXP(a, b) a##b
#define CVAUX_CONCAT(a, b) CVAUX_CONCAT_EXP(a,b)

#if defined(__clang__)
#  ifndef __has_extension
#    define __has_extension __has_feature /* compatibility, for older versions of clang */
#  endif
#  if __has_extension(cxx_static_assert)
#    define CV_StaticAssert(condition, reason)    static_assert((condition), reason " " #condition)
#  endif
#elif defined(__GNUC__)
#  if (defined(__GXX_EXPERIMENTAL_CXX0X__) || __cplusplus >= 201103L)
#    define CV_StaticAssert(condition, reason)    static_assert((condition), reason " " #condition)
#  endif
#elif defined(_MSC_VER)
#  if _MSC_VER >= 1600 /* MSVC 10 */
#    define CV_StaticAssert(condition, reason)    static_assert((condition), reason " " #condition)
#  endif
#endif
#ifndef CV_StaticAssert
#  if defined(__GNUC__) && (__GNUC__ > 3) && (__GNUC_MINOR__ > 2)
#    define CV_StaticAssert(condition, reason) ({ extern int __attribute__((error("CV_StaticAssert: " reason " " #condition))) CV_StaticAssert(); ((condition) ? 0 : CV_StaticAssert()); })
#  else
     template <bool x> struct CV_StaticAssert_failed;
     template <> struct CV_StaticAssert_failed<true> { enum { val = 1 }; };
     template<int x> struct CV_StaticAssert_test {};
#    define CV_StaticAssert(condition, reason)\
       typedef cv::CV_StaticAssert_test< sizeof(cv::CV_StaticAssert_failed< static_cast<bool>(condition) >) > CVAUX_CONCAT(CV_StaticAssert_failed_at_, __LINE__)
#  endif
#endif

//! Suppress warning "-Wdeprecated-declarations" / C4996

#if defined(_MSC_VER)
    #define CV_DO_PRAGMA(x) __pragma(x)
#elif defined(__GNUC__)
    #define CV_DO_PRAGMA(x) _Pragma (#x)
#else
    #define CV_DO_PRAGMA(x)
#endif

#ifdef _MSC_VER
#define CV_SUPPRESS_DEPRECATED_START \
    CV_DO_PRAGMA(warning(push)) \
    CV_DO_PRAGMA(warning(disable: 4996))
#define CV_SUPPRESS_DEPRECATED_END CV_DO_PRAGMA(warning(pop))
#elif defined __GNUC__
#define CV_SUPPRESS_DEPRECATED_START \
    CV_DO_PRAGMA(GCC diagnostic push) \
    CV_DO_PRAGMA(GCC diagnostic ignored "-Wdeprecated-declarations")
#define CV_SUPPRESS_DEPRECATED_END CV_DO_PRAGMA(GCC diagnostic pop)
#else
#define CV_SUPPRESS_DEPRECATED_START
#define CV_SUPPRESS_DEPRECATED_END
#endif

//! Signals an error and raises the exception.
/*!
  By default the function prints information about the error to stderr,
  then it either stops if setBreakOnError() had been called before or raises the exception.
  It is possible to alternate error processing by using redirectError().

  \param exc the exception raisen.
 */
CV_EXPORTS void error(int _code, const String& _err, const char* _func, const char* _file, int _line);

#ifdef __GNUC__
# if defined __clang__ || defined __APPLE__
#   pragma GCC diagnostic push
#   pragma GCC diagnostic ignored "-Winvalid-noreturn"
# endif
#endif
CV_INLINE CV_NORETURN void errorNoReturn(int _code, const String& _err, const char* _func, const char* _file, int _line)
{
    error(_code, _err, _func, _file, _line);
#ifdef __GNUC__
# if !defined __clang__ && !defined __APPLE__
    // this suppresses this warning: "noreturn" function does return [enabled by default]
    __builtin_trap();
    // or use infinite loop: for (;;) {}
# endif
#endif
}
#ifdef __GNUC__
# if defined __clang__ || defined __APPLE__
#   pragma GCC diagnostic pop
# endif
#endif


#if defined __GNUC__
#define CV_Func __func__
#elif defined _MSC_VER
#define CV_Func __FUNCTION__
#else
#define CV_Func ""
#endif

#define CV_Error( code, msg ) cv::error( code, msg, CV_Func, __FILE__, __LINE__ )
#define CV_Error_( code, args ) cv::error( code, cv::format args, CV_Func, __FILE__, __LINE__ )
#define CV_Assert( expr ) if(!!(expr)) ; else cv::error( cv::Error::StsAssert, #expr, CV_Func, __FILE__, __LINE__ )

#define CV_ErrorNoReturn( code, msg ) cv::errorNoReturn( code, msg, CV_Func, __FILE__, __LINE__ )
#define CV_ErrorNoReturn_( code, args ) cv::errorNoReturn( code, cv::format args, CV_Func, __FILE__, __LINE__ )

#ifdef _DEBUG
#  define CV_DbgAssert(expr) CV_Assert(expr)
#else
#  define CV_DbgAssert(expr)
#endif



/////////////// saturate_cast (used in image & signal processing) ///////////////////

template<typename _Tp> static inline _Tp saturate_cast(uchar v)    { return _Tp(v); }
template<typename _Tp> static inline _Tp saturate_cast(schar v)    { return _Tp(v); }
template<typename _Tp> static inline _Tp saturate_cast(ushort v)   { return _Tp(v); }
template<typename _Tp> static inline _Tp saturate_cast(short v)    { return _Tp(v); }
template<typename _Tp> static inline _Tp saturate_cast(unsigned v) { return _Tp(v); }
template<typename _Tp> static inline _Tp saturate_cast(int v)      { return _Tp(v); }
template<typename _Tp> static inline _Tp saturate_cast(float v)    { return _Tp(v); }
template<typename _Tp> static inline _Tp saturate_cast(double v)   { return _Tp(v); }

template<> inline uchar saturate_cast<uchar>(schar v)        { return (uchar)std::max((int)v, 0); }
template<> inline uchar saturate_cast<uchar>(ushort v)       { return (uchar)std::min((unsigned)v, (unsigned)UCHAR_MAX); }
template<> inline uchar saturate_cast<uchar>(int v)          { return (uchar)((unsigned)v <= UCHAR_MAX ? v : v > 0 ? UCHAR_MAX : 0); }
template<> inline uchar saturate_cast<uchar>(short v)        { return saturate_cast<uchar>((int)v); }
template<> inline uchar saturate_cast<uchar>(unsigned v)     { return (uchar)std::min(v, (unsigned)UCHAR_MAX); }
template<> inline uchar saturate_cast<uchar>(float v)        { int iv = cvRound(v); return saturate_cast<uchar>(iv); }
template<> inline uchar saturate_cast<uchar>(double v)       { int iv = cvRound(v); return saturate_cast<uchar>(iv); }

template<> inline schar saturate_cast<schar>(uchar v)        { return (schar)std::min((int)v, SCHAR_MAX); }
template<> inline schar saturate_cast<schar>(ushort v)       { return (schar)std::min((unsigned)v, (unsigned)SCHAR_MAX); }
template<> inline schar saturate_cast<schar>(int v)          { return (schar)((unsigned)(v-SCHAR_MIN) <= (unsigned)UCHAR_MAX ? v : v > 0 ? SCHAR_MAX : SCHAR_MIN); }
template<> inline schar saturate_cast<schar>(short v)        { return saturate_cast<schar>((int)v); }
template<> inline schar saturate_cast<schar>(unsigned v)     { return (schar)std::min(v, (unsigned)SCHAR_MAX); }
template<> inline schar saturate_cast<schar>(float v)        { int iv = cvRound(v); return saturate_cast<schar>(iv); }
template<> inline schar saturate_cast<schar>(double v)       { int iv = cvRound(v); return saturate_cast<schar>(iv); }

template<> inline ushort saturate_cast<ushort>(schar v)      { return (ushort)std::max((int)v, 0); }
template<> inline ushort saturate_cast<ushort>(short v)      { return (ushort)std::max((int)v, 0); }
template<> inline ushort saturate_cast<ushort>(int v)        { return (ushort)((unsigned)v <= (unsigned)USHRT_MAX ? v : v > 0 ? USHRT_MAX : 0); }
template<> inline ushort saturate_cast<ushort>(unsigned v)   { return (ushort)std::min(v, (unsigned)USHRT_MAX); }
template<> inline ushort saturate_cast<ushort>(float v)      { int iv = cvRound(v); return saturate_cast<ushort>(iv); }
template<> inline ushort saturate_cast<ushort>(double v)     { int iv = cvRound(v); return saturate_cast<ushort>(iv); }

template<> inline short saturate_cast<short>(ushort v)       { return (short)std::min((int)v, SHRT_MAX); }
template<> inline short saturate_cast<short>(int v)          { return (short)((unsigned)(v - SHRT_MIN) <= (unsigned)USHRT_MAX ? v : v > 0 ? SHRT_MAX : SHRT_MIN); }
template<> inline short saturate_cast<short>(unsigned v)     { return (short)std::min(v, (unsigned)SHRT_MAX); }
template<> inline short saturate_cast<short>(float v)        { int iv = cvRound(v); return saturate_cast<short>(iv); }
template<> inline short saturate_cast<short>(double v)       { int iv = cvRound(v); return saturate_cast<short>(iv); }

template<> inline int saturate_cast<int>(float v)            { return cvRound(v); }
template<> inline int saturate_cast<int>(double v)           { return cvRound(v); }

// we intentionally do not clip negative numbers, to make -1 become 0xffffffff etc.
template<> inline unsigned saturate_cast<unsigned>(float v)  { return cvRound(v); }
template<> inline unsigned saturate_cast<unsigned>(double v) { return cvRound(v); }



//////////////////////////////// low-level functions ////////////////////////////////

CV_EXPORTS int LU(float* A, size_t astep, int m, float* b, size_t bstep, int n);
CV_EXPORTS int LU(double* A, size_t astep, int m, double* b, size_t bstep, int n);
CV_EXPORTS bool Cholesky(float* A, size_t astep, int m, float* b, size_t bstep, int n);
CV_EXPORTS bool Cholesky(double* A, size_t astep, int m, double* b, size_t bstep, int n);

CV_EXPORTS int normL1_(const uchar* a, const uchar* b, int n);
CV_EXPORTS int normHamming(const uchar* a, const uchar* b, int n);
CV_EXPORTS int normHamming(const uchar* a, const uchar* b, int n, int cellSize);
CV_EXPORTS float normL1_(const float* a, const float* b, int n);
CV_EXPORTS float normL2Sqr_(const float* a, const float* b, int n);

CV_EXPORTS void exp(const float* src, float* dst, int n);
CV_EXPORTS void log(const float* src, float* dst, int n);
CV_EXPORTS void fastAtan2(const float* y, const float* x, float* dst, int n, bool angleInDegrees);
CV_EXPORTS void magnitude(const float* x, const float* y, float* dst, int n);

//! computes cube root of the argument
CV_EXPORTS_W float cubeRoot(float val);
//! computes the angle in degrees (0..360) of the vector (x,y)
CV_EXPORTS_W float fastAtan2(float y, float x);



/////////////////////////////////// inline norms ////////////////////////////////////

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
        s += (_AccTp)std::abs(a[i]) + (_AccTp)std::abs(a[i+1]) +
            (_AccTp)std::abs(a[i+2]) + (_AccTp)std::abs(a[i+3]);
    }
#endif
    for( ; i < n; i++ )
        s += std::abs(a[i]);
    return s;
}

template<typename _Tp, typename _AccTp> static inline
_AccTp normInf(const _Tp* a, int n)
{
    _AccTp s = 0;
    for( int i = 0; i < n; i++ )
        s = std::max(s, (_AccTp)std::abs(a[i]));
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

template<> inline
float normL2Sqr(const float* a, const float* b, int n)
{
    if( n >= 8 )
        return normL2Sqr_(a, b, n);
    float s = 0;
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

template<> inline
float normL1(const float* a, const float* b, int n)
{
    if( n >= 8 )
        return normL1_(a, b, n);
    float s = 0;
    for( int i = 0; i < n; i++ )
    {
        float v = a[i] - b[i];
        s += std::abs(v);
    }
    return s;
}

template<> inline
int normL1(const uchar* a, const uchar* b, int n)
{
    return normL1_(a, b, n);
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



////////////////// forward declarations for important OpenCV types //////////////////

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
    class CV_EXPORTS CudaMem;
    class CV_EXPORTS Stream;
    class CV_EXPORTS Event;
}

namespace cudev
{
    template <typename _Tp> class GpuMat_;
}

} // cv

#endif //__OPENCV_CORE_BASE_HPP__
