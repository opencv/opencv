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
#include <vector>
#include <climits>
#include <algorithm>

// misc functions that not exposed to public interface

namespace cv {

namespace ndsrvp {

void* fastMalloc(size_t size);
void fastFree(void* ptr);
int borderInterpolate(int p, int len, int borderType);
int16x4_t borderInterpolate_vector(int16x4_t vp, short len, int borderType);

#ifndef MAX
#  define MAX(a,b)  ((a) < (b) ? (b) : (a))
#endif

#ifndef MIN
#  define MIN(a,b)  ((a) > (b) ? (b) : (a))
#endif

#define CV_MAT_CN_MASK          ((CV_CN_MAX - 1) << CV_CN_SHIFT)
#define CV_MAT_CN(flags)        ((((flags) & CV_MAT_CN_MASK) >> CV_CN_SHIFT) + 1)

#define CV_ELEM_SIZE1(type) ((0x28442211 >> CV_MAT_DEPTH(type)*4) & 15)
#define CV_ELEM_SIZE(type) (CV_MAT_CN(type)*CV_ELEM_SIZE1(type))

#define CV_MALLOC_ALIGN 64

inline size_t getElemSize(int type) { return (size_t)CV_ELEM_SIZE(type); }

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

// expand

/*
    [0] [1] [2] [3] [4] [5] [6] [7]
810 [  0  ] [  1  ] [  4  ] [  5  ]
832 [  2  ] [  3  ] [  6  ] [  7  ]
bb  [  0  ] [  1  ] [  2  ] [  3  ]
tt  [  4  ] [  5  ] [  6  ] [  7  ]
*/

inline void ndsrvp_u8_u16_expand8(const unsigned long vs, ushort* dst)
{
    unsigned long vs810 = __nds__zunpkd810(vs);
    unsigned long vs832 = __nds__zunpkd832(vs);
    *(unsigned long*)dst = __nds__pkbb32(vs832, vs810);
    *(unsigned long*)(dst + 4) = __nds__pktt32(vs832, vs810);
}

/*
    [0] [1] [2] [3] [4] [5] [6] [7]
820 [  0  ] [  2  ] [  4  ] [  6  ]
831 [  1  ] [  3  ] [  5  ] [  7  ]
bb  [  0  ] [  2  ] [  1  ] [  3  ]
tt  [  4  ] [  6  ] [  5  ] [  7  ]
*/

inline void ndsrvp_u8_u16_eswap8(const unsigned long vs, ushort* dst)
{
    unsigned long vs820 = __nds__zunpkd820(vs);
    unsigned long vs831 = __nds__zunpkd831(vs);
    *(unsigned long*)dst = __nds__pkbb32(vs831, vs820);
    *(unsigned long*)(dst + 4) = __nds__pktt32(vs831, vs820);
}

/*
    [0] [1] [2] [3] [4] [5] [6] [7]
820 [  0  ] [  2  ] [  4  ] [  6  ]
831 [  1  ] [  3  ] [  5  ] [  7  ]
bb  [  0  ] [  2  ] [  1  ] [  3  ]
tt  [  4  ] [  6  ] [  5  ] [  7  ]
bbbb[      0      ] [      1      ]
bbtt[      2      ] [      3      ]
ttbb[      4      ] [      5      ]
tttt[      6      ] [      7      ]
*/


inline void ndsrvp_u8_u32_expand8(const unsigned long vs, uint* dst)
{
    unsigned long vs820 = __nds__zunpkd820(vs);
    unsigned long vs831 = __nds__zunpkd831(vs);
    unsigned long vsbb = __nds__pkbb32(vs831, vs820);
    unsigned long vstt = __nds__pktt32(vs831, vs820);
    *(unsigned long*)dst = __nds__pkbb16(0, vsbb);
    *(unsigned long*)(dst + 2) = __nds__pktt16(0, vsbb);
    *(unsigned long*)(dst + 4) = __nds__pkbb16(0, vstt);
    *(unsigned long*)(dst + 6) = __nds__pktt16(0, vstt);
}

// float replacement

inline void ndsrvp_f32_add8(const float* a, const float* b, float* c)
{
    c[0] = a[0] + b[0];
    c[1] = a[1] + b[1];
    c[2] = a[2] + b[2];
    c[3] = a[3] + b[3];
    c[4] = a[4] + b[4];
    c[5] = a[5] + b[5];
    c[6] = a[6] + b[6];
    c[7] = a[7] + b[7];
}

/*
    [1] [8] [23]
    [24] [8]
*/

inline void ndsrvp_f32_u8_mul8(const float* a, const unsigned long b, float* c) // experimental, not bit exact
{
    const int mask_frac = 0x007FFFFF;
    const int mask_sign = 0x7FFFFFFF;
    const int mask_lead = 0x40000000;
    const int ofs_exp = 23;

    uint32x2_t va01 = *(uint32x2_t*)a;
    uint32x2_t va23 = *(uint32x2_t*)(a + 2);
    uint32x2_t va45 = *(uint32x2_t*)(a + 4);
    uint32x2_t va67 = *(uint32x2_t*)(a + 6);

    uint32x2_t vaexp01 = va01 >> ofs_exp;
    uint32x2_t vaexp23 = va23 >> ofs_exp;
    uint32x2_t vaexp45 = va45 >> ofs_exp;
    uint32x2_t vaexp67 = va67 >> ofs_exp;

    uint32x2_t vafrac01 = ((va01 << 7) & mask_sign) | mask_lead;
    uint32x2_t vafrac23 = ((va23 << 7) & mask_sign) | mask_lead;
    uint32x2_t vafrac45 = ((va45 << 7) & mask_sign) | mask_lead;
    uint32x2_t vafrac67 = ((va67 << 7) & mask_sign) | mask_lead;

    int16x4_t vb[2]; // fake signed for signed multiply
    ndsrvp_u8_u16_eswap8(b, (ushort*)vb);

    vafrac01 = (uint32x2_t)__nds__kmmwb2_u((long)vafrac01, (unsigned long)vb[0]);
    vafrac23 = (uint32x2_t)__nds__kmmwt2_u((long)vafrac23, (unsigned long)vb[0]);
    vafrac45 = (uint32x2_t)__nds__kmmwb2_u((long)vafrac45, (unsigned long)vb[1]);
    vafrac67 = (uint32x2_t)__nds__kmmwt2_u((long)vafrac67, (unsigned long)vb[1]);

    uint32x2_t vaclz01 = __nds__v_clz32(vafrac01) - 8;
    uint32x2_t vaclz23 = __nds__v_clz32(vafrac23) - 8;
    uint32x2_t vaclz45 = __nds__v_clz32(vafrac45) - 8;
    uint32x2_t vaclz67 = __nds__v_clz32(vafrac67) - 8;

    vaexp01 += 8 - vaclz01;
    vaexp23 += 8 - vaclz23;
    vaexp45 += 8 - vaclz45;
    vaexp67 += 8 - vaclz67;

    vafrac01 <<= vaclz01;
    vafrac23 <<= vaclz23;
    vafrac45 <<= vaclz45;
    vafrac67 <<= vaclz67;

    *(uint32x2_t*)c = (vaexp01 << ofs_exp) | (vafrac01 & mask_frac);
    *(uint32x2_t*)(c + 2) = (vaexp23 << ofs_exp) | (vafrac23 & mask_frac);
    *(uint32x2_t*)(c + 4) = (vaexp45 << ofs_exp) | (vafrac45 & mask_frac);
    *(uint32x2_t*)(c + 6) = (vaexp67 << ofs_exp) | (vafrac67 & mask_frac);
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

inline double cast_ptr_to_double(const uchar* v, int depth) {
    switch (depth) {
        case CV_8U: return (double)*(uchar*)v;
        case CV_8S: return (double)*(char*)v;
        case CV_16U: return (double)*(ushort*)v;
        case CV_16S: return (double)*(short*)v;
        case CV_32S: return (double)*(int*)v;
        case CV_32F: return (double)*(float*)v;
        case CV_64F: return (double)*(double*)v;
        case CV_16F: return (double)*(float*)v;
        default: return 0;
    }
}

template <typename _Tp>
inline _Tp data_at(const uchar* data, int step, int y, int x, int cn)
{
    return ((_Tp*)(data + y * step))[x * cn];
}

// align

inline long align(size_t v, int n)
{
    return (v + n - 1) & -n;
}

} // namespace ndsrvp

} // namespace cv

#endif
