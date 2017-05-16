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
#include "opencv2/core/va_intel.hpp"

#include "opencv2/core/private.hpp"
#include "opencv2/core/private.cuda.hpp"
#ifdef HAVE_OPENCL
#include "opencv2/core/ocl.hpp"
#endif

#include <assert.h>
#include <ctype.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <float.h>
#include <cstring>
#include <cassert>

#define USE_SSE2  (cv::checkHardwareSupport(CV_CPU_SSE2))
#define USE_SSE4_2  (cv::checkHardwareSupport(CV_CPU_SSE4_2))
#define USE_AVX  (cv::checkHardwareSupport(CV_CPU_AVX))
#define USE_AVX2  (cv::checkHardwareSupport(CV_CPU_AVX2))

#include "opencv2/core/hal/hal.hpp"
#include "opencv2/core/hal/intrin.hpp"
#include "opencv2/core/sse_utils.hpp"
#include "opencv2/core/neon_utils.hpp"

#include "arithm_core.hpp"
#include "hal_replacement.hpp"

#ifdef HAVE_TEGRA_OPTIMIZATION
#include "opencv2/core/core_tegra.hpp"
#else
#define GET_OPTIMIZED(func) (func)
#endif

namespace cv
{

// -128.f ... 255.f
extern const float g_8x32fTab[];
#define CV_8TO32F(x)  cv::g_8x32fTab[(x)+128]

extern const ushort g_8x16uSqrTab[];
#define CV_SQR_8U(x)  cv::g_8x16uSqrTab[(x)+255]

extern const uchar g_Saturate8u[];
#define CV_FAST_CAST_8U(t)   (assert(-256 <= (t) && (t) <= 512), cv::g_Saturate8u[(t)+256])
#define CV_MIN_8U(a,b)       ((a) - CV_FAST_CAST_8U((a) - (b)))
#define CV_MAX_8U(a,b)       ((a) + CV_FAST_CAST_8U((b) - (a)))

template<> inline uchar OpAdd<uchar>::operator ()(uchar a, uchar b) const
{ return CV_FAST_CAST_8U(a + b); }

template<> inline uchar OpSub<uchar>::operator ()(uchar a, uchar b) const
{ return CV_FAST_CAST_8U(a - b); }

template<> inline short OpAbsDiff<short>::operator ()(short a, short b) const
{ return saturate_cast<short>(std::abs(a - b)); }

template<> inline schar OpAbsDiff<schar>::operator ()(schar a, schar b) const
{ return saturate_cast<schar>(std::abs(a - b)); }

template<> inline uchar OpMin<uchar>::operator ()(uchar a, uchar b) const { return CV_MIN_8U(a, b); }

template<> inline uchar OpMax<uchar>::operator ()(uchar a, uchar b) const { return CV_MAX_8U(a, b); }

typedef void (*BinaryFunc)(const uchar* src1, size_t step1,
                       const uchar* src2, size_t step2,
                       uchar* dst, size_t step, Size sz,
                       void*);

typedef void (*BinaryFuncC)(const uchar* src1, size_t step1,
                       const uchar* src2, size_t step2,
                       uchar* dst, size_t step, int width, int height,
                       void*);

BinaryFunc getConvertFuncFp16(int ddepth);
BinaryFunc getConvertFunc(int sdepth, int ddepth);
BinaryFunc getCopyMaskFunc(size_t esz);

/* default memory block for sparse array elements */
#define  CV_SPARSE_MAT_BLOCK     (1<<12)

/* initial hash table size */
#define  CV_SPARSE_HASH_SIZE0    (1<<10)

/* maximal average node_count/hash_size ratio beyond which hash table is resized */
#define  CV_SPARSE_HASH_RATIO    3

#if defined WIN32 || defined _WIN32
void deleteThreadAllocData();
#endif

inline Size getContinuousSize_( int flags, int cols, int rows, int widthScale )
{
    int64 sz = (int64)cols * rows * widthScale;
    return (flags & Mat::CONTINUOUS_FLAG) != 0 &&
        (int)sz == sz ? Size((int)sz, 1) : Size(cols * widthScale, rows);
}

inline Size getContinuousSize( const Mat& m1, int widthScale=1 )
{
    return getContinuousSize_(m1.flags,
                              m1.cols, m1.rows, widthScale);
}

inline Size getContinuousSize( const Mat& m1, const Mat& m2, int widthScale=1 )
{
    return getContinuousSize_(m1.flags & m2.flags,
                              m1.cols, m1.rows, widthScale);
}

inline Size getContinuousSize( const Mat& m1, const Mat& m2,
                               const Mat& m3, int widthScale=1 )
{
    return getContinuousSize_(m1.flags & m2.flags & m3.flags,
                              m1.cols, m1.rows, widthScale);
}

inline Size getContinuousSize( const Mat& m1, const Mat& m2,
                               const Mat& m3, const Mat& m4,
                               int widthScale=1 )
{
    return getContinuousSize_(m1.flags & m2.flags & m3.flags & m4.flags,
                              m1.cols, m1.rows, widthScale);
}

inline Size getContinuousSize( const Mat& m1, const Mat& m2,
                               const Mat& m3, const Mat& m4,
                               const Mat& m5, int widthScale=1 )
{
    return getContinuousSize_(m1.flags & m2.flags & m3.flags & m4.flags & m5.flags,
                              m1.cols, m1.rows, widthScale);
}

struct NoVec
{
    size_t operator()(const void*, const void*, void*, size_t) const { return 0; }
};

enum { BLOCK_SIZE = 1024 };

#if defined HAVE_IPP && (IPP_VERSION_X100 >= 700)
#define ARITHM_USE_IPP 1
#else
#define ARITHM_USE_IPP 0
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

#ifdef CV_COLLECT_IMPL_DATA
struct ImplCollector
{
    ImplCollector()
    {
        useCollection   = false;
        implFlags       = 0;
    }
    bool useCollection; // enable/disable impl data collection

    int implFlags;
    std::vector<int>    implCode;
    std::vector<String> implFun;

    cv::Mutex mutex;
};
#endif

struct CoreTLSData
{
    CoreTLSData() :
//#ifdef HAVE_OPENCL
        device(0), useOpenCL(-1),
//#endif
        useIPP(-1)
#ifdef HAVE_TEGRA_OPTIMIZATION
        ,useTegra(-1)
#endif
#ifdef HAVE_OPENVX
        ,useOpenVX(-1)
#endif
    {}

    RNG rng;
//#ifdef HAVE_OPENCL
    int device; // device index of an array of devices in a context, see also Device::getDefault
    ocl::Queue oclQueue; // the queue used for running a kernel, see also getQueue, Kernel::run
    int useOpenCL; // 1 - use, 0 - do not use, -1 - auto/not initialized
//#endif
    int useIPP; // 1 - use, 0 - do not use, -1 - auto/not initialized
#ifdef HAVE_TEGRA_OPTIMIZATION
    int useTegra; // 1 - use, 0 - do not use, -1 - auto/not initialized
#endif
#ifdef HAVE_OPENVX
    int useOpenVX; // 1 - use, 0 - do not use, -1 - auto/not initialized
#endif
};

TLSData<CoreTLSData>& getCoreTlsData();

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

extern bool __termination; // skip some cleanups, because process is terminating
                           // (for example, if ExitProcess() was already called)

cv::Mutex& getInitializationMutex();

// TODO Memory barriers?
#define CV_SINGLETON_LAZY_INIT_(TYPE, INITIALIZER, RET_VALUE) \
    static TYPE* volatile instance = NULL; \
    if (instance == NULL) \
    { \
        cv::AutoLock lock(cv::getInitializationMutex()); \
        if (instance == NULL) \
            instance = INITIALIZER; \
    } \
    return RET_VALUE;

#define CV_SINGLETON_LAZY_INIT(TYPE, INITIALIZER) CV_SINGLETON_LAZY_INIT_(TYPE, INITIALIZER, instance)
#define CV_SINGLETON_LAZY_INIT_REF(TYPE, INITIALIZER) CV_SINGLETON_LAZY_INIT_(TYPE, INITIALIZER, *instance)

}

#endif /*_CXCORE_INTERNAL_H_*/
