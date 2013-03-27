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

#include "opencv2/core/cvdef.h"

namespace cv
{

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



////////////////// forward declarations for important OpenCV types //////////////////

template<typename _Tp, int cn> class CV_EXPORTS Vec;
template<typename _Tp, int m, int n> class CV_EXPORTS Matx;

template<typename _Tp> class CV_EXPORTS Complex;
template<typename _Tp> class CV_EXPORTS Point_;
template<typename _Tp> class CV_EXPORTS Point3_;
template<typename _Tp> class CV_EXPORTS Size_;
template<typename _Tp> class CV_EXPORTS Rect_;
template<typename _Tp> class CV_EXPORTS Scalar_;

class CV_EXPORTS RotatedRect;
class CV_EXPORTS Range;
class CV_EXPORTS TermCriteria;
class CV_EXPORTS KeyPoint;
class CV_EXPORTS DMatch;

class CV_EXPORTS Mat;
class CV_EXPORTS SparseMat;
typedef Mat MatND;

template<typename _Tp> class CV_EXPORTS Mat_;
template<typename _Tp> class CV_EXPORTS MatIterator_;
template<typename _Tp> class CV_EXPORTS MatConstIterator_;

namespace ogl
{
    class CV_EXPORTS Buffer;
    class CV_EXPORTS Texture2D;
    class CV_EXPORTS Arrays;
}

namespace gpu
{
    class CV_EXPORTS GpuMat;
}

} // cv

#endif //__OPENCV_CORE_BASE_HPP__
