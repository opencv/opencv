// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.	

#ifndef OPENCV_NDSRVP_CVUTILS_HPP
#define OPENCV_NDSRVP_CVUTILS_HPP

#include <nds_intrinsic.h>

#include "opencv2/core/hal/interface.h"

#include <cstring>
#include <cmath>
#include <iostream>
#include <string>
#include <array>
#include <climits>
#include <algorithm>

// misc functions that not exposed to public interface

namespace cv {

namespace ndsrvp {

void* fastMalloc(size_t size);
void fastFree(void* ptr);
int borderInterpolate(int p, int len, int borderType);

#ifndef MAX
#  define MAX(a,b)  ((a) < (b) ? (b) : (a))
#endif

#define CV_MAT_CN_MASK          ((CV_CN_MAX - 1) << CV_CN_SHIFT)
#define CV_MAT_CN(flags)        ((((flags) & CV_MAT_CN_MASK) >> CV_CN_SHIFT) + 1)

#define CV_MALLOC_ALIGN 64

// error codes

enum Error{
    StsNoMem = -4,
    StsBadArg = -5,
    StsAssert = -215
};

// output error

#define ndsrvp_assert(expr) { if(!(expr)) ndsrvp_error(Error::StsAssert, std::string(#expr)); }

inline void ndsrvp_error(int code, std::string msg = "")
{
    std::cerr << "NDSRVP Error: code " << code << std::endl;
    if(!msg.empty())
        std::cerr << msg << std::endl;
    if(code < 0)
        throw code;
}

// clip & vclip

inline int clip(int x, int a, int b)
{
    return x >= a ? (x < b ? x : b - 1) : a;
}

inline int32x2_t vclip(int32x2_t x, int32x2_t a, int32x2_t b)
{
    return (int32x2_t)__nds__bpick((long)a, __nds__bpick((long)(b - 1), (long)x, (long)(x < b)), (long)(x >= a));
}

// saturate

template<typename _Tp> static inline _Tp saturate_cast(int v)    { return _Tp(v); }
template<typename _Tp> static inline _Tp saturate_cast(float v)    { return _Tp(v); }
template<typename _Tp> static inline _Tp saturate_cast(double v)   { return _Tp(v); }

template<> inline uchar saturate_cast<uchar>(int v)     { return __nds__uclip32(v, 8); }
template<> inline uchar saturate_cast<uchar>(float v)     { return saturate_cast<uchar>((int)lrintf(v)); }
template<> inline uchar saturate_cast<uchar>(double v)     { return saturate_cast<uchar>((int)lrint(v)); }

template<> inline char saturate_cast<char>(int v)     { return __nds__sclip32(v, 7); }
template<> inline char saturate_cast<char>(float v)     { return saturate_cast<char>((int)lrintf(v)); }
template<> inline char saturate_cast<char>(double v)     { return saturate_cast<char>((int)lrint(v)); }

template<> inline ushort saturate_cast<ushort>(int v)     { return __nds__uclip32(v, 16); }
template<> inline ushort saturate_cast<ushort>(float v)     { return saturate_cast<ushort>((int)lrintf(v)); }
template<> inline ushort saturate_cast<ushort>(double v)     { return saturate_cast<ushort>((int)lrint(v)); }

template<> inline short saturate_cast<short>(int v)     { return __nds__sclip32(v, 15); }
template<> inline short saturate_cast<short>(float v)     { return saturate_cast<short>((int)lrintf(v)); }
template<> inline short saturate_cast<short>(double v)     { return saturate_cast<short>((int)lrint(v)); }

template<> inline int saturate_cast<int>(float v)     { return (int)lrintf(v); }
template<> inline int saturate_cast<int>(double v)     { return (int)lrint(v); }

// align

inline long align(size_t v, int n)
{
    return (v + n - 1) & -n;
}

} // namespace ndsrvp

} // namespace cv

#endif
