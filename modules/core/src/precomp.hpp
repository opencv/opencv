/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

#ifndef __OPENCV_PRECOMP_H__
#define __OPENCV_PRECOMP_H__

#include "opencv2/opencv_modules.hpp"
#include "cvconfig.h"

#include "opencv2/core/utility.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/core/cuda.hpp"
#include "opencv2/core/opengl.hpp"

#include "opencv2/core/private.hpp"
#include "opencv2/core/private.cuda.hpp"
#include "opencv2/core/ocl.hpp"

#include <assert.h>
#include <ctype.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef HAVE_TEGRA_OPTIMIZATION
#include "opencv2/core/core_tegra.hpp"
#else
#define GET_OPTIMIZED(func) (func)
#endif

namespace cv
{

typedef void (*BinaryFunc)(const uchar* src1, size_t step1,
                       const uchar* src2, size_t step2,
                       uchar* dst, size_t step, Size sz,
                       void*);

BinaryFunc getConvertFunc(int sdepth, int ddepth);
BinaryFunc getCopyMaskFunc(size_t esz);

/* default memory block for sparse array elements */
#define  CV_SPARSE_MAT_BLOCK     (1<<12)

/* initial hash table size */
#define  CV_SPARSE_HASH_SIZE0    (1<<10)

/* maximal average node_count/hash_size ratio beyond which hash table is resized */
#define  CV_SPARSE_HASH_RATIO    3



// -128.f ... 255.f
extern const float g_8x32fTab[];
#define CV_8TO32F(x)  cv::g_8x32fTab[(x)+128]

extern const ushort g_8x16uSqrTab[];
#define CV_SQR_8U(x)  cv::g_8x16uSqrTab[(x)+255]

extern const char* g_HersheyGlyphs[];

extern const uchar g_Saturate8u[];
#define CV_FAST_CAST_8U(t)   (assert(-256 <= (t) && (t) <= 512), cv::g_Saturate8u[(t)+256])
#define CV_MIN_8U(a,b)       ((a) - CV_FAST_CAST_8U((a) - (b)))
#define CV_MAX_8U(a,b)       ((a) + CV_FAST_CAST_8U((b) - (a)))


#if defined WIN32 || defined _WIN32
void deleteThreadAllocData();
#endif

template<typename T1, typename T2=T1, typename T3=T1> struct OpAdd
{
    typedef T1 type1;
    typedef T2 type2;
    typedef T3 rtype;
    T3 operator ()(const T1 a, const T2 b) const { return saturate_cast<T3>(a + b); }
};

template<typename T1, typename T2=T1, typename T3=T1> struct OpSub
{
    typedef T1 type1;
    typedef T2 type2;
    typedef T3 rtype;
    T3 operator ()(const T1 a, const T2 b) const { return saturate_cast<T3>(a - b); }
};

template<typename T1, typename T2=T1, typename T3=T1> struct OpRSub
{
    typedef T1 type1;
    typedef T2 type2;
    typedef T3 rtype;
    T3 operator ()(const T1 a, const T2 b) const { return saturate_cast<T3>(b - a); }
};

template<typename T> struct OpMin
{
    typedef T type1;
    typedef T type2;
    typedef T rtype;
    T operator ()(const T a, const T b) const { return std::min(a, b); }
};

template<typename T> struct OpMax
{
    typedef T type1;
    typedef T type2;
    typedef T rtype;
    T operator ()(const T a, const T b) const { return std::max(a, b); }
};

inline Size getContinuousSize( const Mat& m1, int widthScale=1 )
{
    return m1.isContinuous() ? Size(m1.cols*m1.rows*widthScale, 1) :
        Size(m1.cols*widthScale, m1.rows);
}

inline Size getContinuousSize( const Mat& m1, const Mat& m2, int widthScale=1 )
{
    return (m1.flags & m2.flags & Mat::CONTINUOUS_FLAG) != 0 ?
        Size(m1.cols*m1.rows*widthScale, 1) : Size(m1.cols*widthScale, m1.rows);
}

inline Size getContinuousSize( const Mat& m1, const Mat& m2,
                               const Mat& m3, int widthScale=1 )
{
    return (m1.flags & m2.flags & m3.flags & Mat::CONTINUOUS_FLAG) != 0 ?
        Size(m1.cols*m1.rows*widthScale, 1) : Size(m1.cols*widthScale, m1.rows);
}

inline Size getContinuousSize( const Mat& m1, const Mat& m2,
                               const Mat& m3, const Mat& m4,
                               int widthScale=1 )
{
    return (m1.flags & m2.flags & m3.flags & m4.flags & Mat::CONTINUOUS_FLAG) != 0 ?
        Size(m1.cols*m1.rows*widthScale, 1) : Size(m1.cols*widthScale, m1.rows);
}

inline Size getContinuousSize( const Mat& m1, const Mat& m2,
                               const Mat& m3, const Mat& m4,
                               const Mat& m5, int widthScale=1 )
{
    return (m1.flags & m2.flags & m3.flags & m4.flags & m5.flags & Mat::CONTINUOUS_FLAG) != 0 ?
        Size(m1.cols*m1.rows*widthScale, 1) : Size(m1.cols*widthScale, m1.rows);
}

struct NoVec
{
    size_t operator()(const void*, const void*, void*, size_t) const { return 0; }
};

extern volatile bool USE_SSE2;
extern volatile bool USE_SSE4_2;
extern volatile bool USE_AVX;

enum { BLOCK_SIZE = 1024 };

#if defined HAVE_IPP && (IPP_VERSION_MAJOR >= 7)
#define ARITHM_USE_IPP 1
#define IF_IPP(then_call, else_call) then_call
#else
#define ARITHM_USE_IPP 0
#define IF_IPP(then_call, else_call) else_call
#endif

inline bool checkScalar(const Mat& sc, int atype, int sckind, int akind)
{
    if( sc.dims > 2 || !sc.isContinuous() )
        return false;
    Size sz = sc.size();
    if(sz.width != 1 && sz.height != 1)
        return false;
    int cn = CV_MAT_CN(atype);
    if( akind == _InputArray::MATX && sckind != _InputArray::MATX )
        return false;
    return sz == Size(1, 1) || sz == Size(1, cn) || sz == Size(cn, 1) ||
           (sz == Size(1, 4) && sc.type() == CV_64F && cn <= 4);
}

inline bool checkScalar(InputArray sc, int atype, int sckind, int akind)
{
    if( sc.dims() > 2 || !sc.isContinuous() )
        return false;
    Size sz = sc.size();
    if(sz.width != 1 && sz.height != 1)
        return false;
    int cn = CV_MAT_CN(atype);
    if( akind == _InputArray::MATX && sckind != _InputArray::MATX )
        return false;
    return sz == Size(1, 1) || sz == Size(1, cn) || sz == Size(cn, 1) ||
           (sz == Size(1, 4) && sc.type() == CV_64F && cn <= 4);
}

void convertAndUnrollScalar( const Mat& sc, int buftype, uchar* scbuf, size_t blocksize );

struct CoreTLSData
{
    CoreTLSData() : device(0), useOpenCL(-1)
    {}

    RNG rng;
    int device;
    ocl::Queue oclQueue;
    int useOpenCL; // 1 - use, 0 - do not use, -1 - auto/not initialized
};

extern TLSData<CoreTLSData> coreTlsData;

#if defined(BUILD_SHARED_LIBS)
#if defined WIN32 || defined _WIN32 || defined WINCE
#define CL_RUNTIME_EXPORT __declspec(dllexport)
#elif defined __GNUC__ && __GNUC__ >= 4
#define CL_RUNTIME_EXPORT __attribute__ ((visibility ("default")))
#else
#define CL_RUNTIME_EXPORT
#endif
#else
#define CL_RUNTIME_EXPORT
#endif

namespace ocl
{
    MatAllocator* getOpenCLAllocator();
}

extern bool __termination; // skip some cleanups, because process is terminating
                           // (for example, if ExitProcess() was already called)

}

#endif /*_CXCORE_INTERNAL_H_*/
