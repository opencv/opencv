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

template<typename _Tp> inline _Tp cv_abs(_Tp x) { return (_Tp)std::abs(x); }
template<typename _Tp> inline _Tp cv_absdiff(_Tp x, _Tp y) { return (_Tp)std::abs(x - y); }
inline int cv_abs(uchar x) { return x; }
inline int cv_abs(schar x) { return std::abs(x); }
inline int cv_abs(ushort x) { return x; }
inline int cv_abs(short x) { return std::abs(x); }
inline unsigned cv_abs(int x) { return (unsigned)std::abs(x); }
inline unsigned cv_abs(unsigned x) { return x; }
inline uint64 cv_abs(uint64 x) { return x; }
inline uint64 cv_abs(int64 x) { return (uint64)std::abs(x); }
inline float cv_abs(hfloat x) { return std::abs((float)x); }
inline float cv_abs(bfloat x) { return std::abs((float)x); }
inline int cv_absdiff(uchar x, uchar y) { return (int)std::abs((int)x - (int)y); }
inline int cv_absdiff(schar x, schar y) { return (int)std::abs((int)x - (int)y); }
inline int cv_absdiff(ushort x, ushort y) { return (int)std::abs((int)x - (int)y); }
inline int cv_absdiff(short x, short y) { return (int)std::abs((int)x - (int)y); }
inline unsigned cv_absdiff(int x, int y) { return (unsigned)(std::max(x, y) - std::min(x, y)); }
inline unsigned cv_absdiff(unsigned x, unsigned y) { return std::max(x, y) - std::min(x, y); }
inline uint64 cv_absdiff(uint64 x, uint64 y) { return std::max(x, y) - std::min(x, y); }
inline float cv_absdiff(hfloat x, hfloat y) { return std::abs((float)x - (float)y); }
inline float cv_absdiff(bfloat x, bfloat y) { return std::abs((float)x - (float)y); }

template<typename _Tp, typename _AccTp> static inline
_AccTp normL2Sqr(const _Tp* a, int n)
{
    _AccTp s = 0;
    for( int i = 0; i < n; i++ )
    {
        _AccTp v = (_AccTp)a[i];
        s += v*v;
    }
    return s;
}

template<typename _Tp, typename _AccTp> static inline
_AccTp normL1(const _Tp* a, int n)
{
    _AccTp s = 0;
    for( int i = 0; i < n; i++ )
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
    for( int i = 0; i < n; i++ )
    {
        _AccTp v = (_AccTp)a[i] - (_AccTp)b[i];
        s += v*v;
    }
    return s;
}

template<typename _Tp, typename _AccTp> static inline
_AccTp normL1(const _Tp* a, const _Tp* b, int n)
{
    _AccTp s = 0;
    for( int i = 0; i < n; i++ )
        s += (_AccTp)cv_absdiff(a[i], b[i]);
    return s;
}

template<typename _Tp, typename _AccTp> static inline
_AccTp normInf(const _Tp* a, const _Tp* b, int n)
{
    _AccTp s = 0;
    for( int i = 0; i < n; i++ )
        s = std::max(s, (_AccTp)cv_absdiff(a[i], b[i]));
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

//! @} core_utils


//! @cond IGNORED

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

} // cv

#include "opencv2/core/fwddecl.hpp"
#include "opencv2/core/neon_utils.hpp"
#include "opencv2/core/vsx_utils.hpp"
#include "opencv2/core/exception.hpp"
#include "opencv2/core/check.hpp"

#endif //OPENCV_CORE_BASE_HPP
