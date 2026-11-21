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
// Copyright (C) 2000-2008, 2018, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2014-2015, Itseez Inc., all rights reserved.
// Copyright (C) 2025-2026, Advanced Micro Devices, all rights reserved.
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

#include "precomp.hpp"
#include "opencv2/core/hal/intrin.hpp"
#include <cstddef>
#include <type_traits>

namespace cv {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN
// forward declarations
Ptr<BaseRowFilter> getRowSumFilter(int srcType, int sumType, int ksize, int anchor);
Ptr<BaseColumnFilter> getColumnSumFilter(int sumType, int dstType, int ksize, int anchor, double scale);
Ptr<BaseRowColumnFilter> getRowColumnSumFilter(int srcType, int dstType, int ksize, double scale);
Ptr<FilterEngine> createBoxFilter(int srcType, int dstType, Size ksize,
                                  Point anchor, bool normalize, int borderType);

void blockSum(const Mat& src, Mat& dst, Size ksize, Point anchor,
              const Size &wsz, const Point &ofs, bool normalize, int borderType);

Ptr<BaseRowFilter> getSqrRowSumFilter(int srcType, int sumType, int ksize, int anchor);



#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY
/****************************************************************************************\
                                         Box Filter
\****************************************************************************************/

namespace {

// Kept outside the align-loops region below: forcing 64-byte alignment on the
// separable sliding window is the Zen 5 ksize 7/21 regression.
template<typename T, typename ST>
struct RowSumSeparableSIMD
{
    static bool apply(const T*, ST*, int, int, int) { return false; }
};

#if defined(__GNUC__)
#define CV_BOX_NOINLINE __attribute__((noinline))
#else
#define CV_BOX_NOINLINE
#endif

template<typename T, typename ST>
CV_BOX_NOINLINE void rowSlideScalarCn1(const T* s, ST* D, int len, int kcn)
{
    ST sum = 0;
    for( int i = 0; i < kcn; i++ )
        sum += (ST)s[i];
    D[0] = sum;
    for( int i = 0; i < len - 1; i++ )
    {
        sum += (ST)s[i + kcn] - (ST)s[i];
        D[i + 1] = sum;
    }
}

template<typename T, typename ST>
CV_BOX_NOINLINE void rowSlideScalarCn3(const T* s, ST* D, int len, int kcn)
{
    ST s0 = 0, s1 = 0, s2 = 0;
    for( int i = 0; i < kcn; i += 3 )
    {
        s0 += (ST)s[i];
        s1 += (ST)s[i + 1];
        s2 += (ST)s[i + 2];
    }
    D[0] = s0; D[1] = s1; D[2] = s2;
    for( int i = 0; i < len - 3; i += 3 )
    {
        s0 += (ST)s[i + kcn] - (ST)s[i];
        s1 += (ST)s[i + kcn + 1] - (ST)s[i + 1];
        s2 += (ST)s[i + kcn + 2] - (ST)s[i + 2];
        D[i + 3] = s0; D[i + 4] = s1; D[i + 5] = s2;
    }
}

#if CV_SIMD && !CV_SIMD_SCALABLE
// Separable integer row sum (every ksize other than 3 and 5, including 7, 21,
// and blur 16x16). 256-bit vectors are used even in the AVX-512 build: the
// 512-bit rotate expands through a mask and is slow.

#if CV_SIMD256
typedef v_int32x8 RowS32;
typedef v_uint32x8 RowU32;
typedef v_int16x16 RowS16;
typedef v_uint16x16 RowU16;
typedef v_uint8x32 RowU8;
enum { kRowS32 = 8, kRowS16 = 16, kRowU8 = 32 };
inline RowS32 rowBcastS32(int v) { return v256_setall_s32(v); }
inline RowU16 rowBcastU16(ushort v) { return v256_setall_u16(v); }
inline RowS16 rowLdS16(const short* p) { return v256_load(p); }
inline RowS32 rowLdS32(const int* p) { return v256_load(p); }
inline RowU8 rowLdU8(const uchar* p) { return v256_load(p); }
inline RowU16 rowLdExpandU8(const uchar* p) { return v256_load_expand(p); }
inline RowU16 rowBcastU64AsU16(uint64 v) { return v_reinterpret_as_u16(v256_setall_u64(v)); }
inline RowS32 rowPrefixS32(RowS32 v)
{
    v = v_add(v, v_rotate_left<1>(v));
    v = v_add(v, v_rotate_left<2>(v));
    v = v_add(v, v_rotate_left<4>(v));
    return v;
}
// Per-channel prefix over 4 interleaved channels (4 pixels per vector).
inline RowU16 rowPrefixCn4U16(RowU16 v)
{
    v = v_add_wrap(v, v_rotate_left<4>(v));
    v = v_add_wrap(v, v_rotate_left<8>(v));
    return v;
}
inline RowU16 rowPrefixU16(RowU16 v)
{
    v = v_add_wrap(v, v_rotate_left<1>(v));
    v = v_add_wrap(v, v_rotate_left<2>(v));
    v = v_add_wrap(v, v_rotate_left<4>(v));
    v = v_add_wrap(v, v_rotate_left<8>(v));
    return v;
}
#else
typedef v_int32x4 RowS32;
typedef v_uint32x4 RowU32;
typedef v_int16x8 RowS16;
typedef v_uint16x8 RowU16;
typedef v_uint8x16 RowU8;
enum { kRowS32 = 4, kRowS16 = 8, kRowU8 = 16 };
inline RowS32 rowBcastS32(int v) { return v_setall_s32(v); }
inline RowU16 rowBcastU16(ushort v) { return v_setall_u16(v); }
inline RowS16 rowLdS16(const short* p) { return v_load(p); }
inline RowS32 rowLdS32(const int* p) { return v_load(p); }
inline RowU8 rowLdU8(const uchar* p) { return v_load(p); }
inline RowU16 rowLdExpandU8(const uchar* p) { return v_load_expand(p); }
inline RowU16 rowBcastU64AsU16(uint64 v) { return v_reinterpret_as_u16(v_setall_u64(v)); }
inline RowS32 rowPrefixS32(RowS32 v)
{
    v = v_add(v, v_rotate_left<1>(v));
    v = v_add(v, v_rotate_left<2>(v));
    return v;
}
inline RowU16 rowPrefixCn4U16(RowU16 v)
{
    return v_add_wrap(v, v_rotate_left<4>(v));
}
inline RowU16 rowPrefixU16(RowU16 v)
{
    v = v_add_wrap(v, v_rotate_left<1>(v));
    v = v_add_wrap(v, v_rotate_left<2>(v));
    v = v_add_wrap(v, v_rotate_left<4>(v));
    return v;
}
#endif

// The running sum is kept broadcast in `carry`. Each block's prefix does not
// depend on it, so the only loop-carried operation is one vector add.
inline RowS32 rowFoldS32(RowS32 delta, RowS32& carry)
{
    RowS32 p = rowPrefixS32(delta);
    RowS32 d = v_add(p, carry);
    carry = v_add(carry, v_broadcast_element<kRowS32 - 1>(p));
    return d;
}

inline RowU16 rowFoldU16(RowU16 delta, RowU16& carry)
{
    RowU16 p = rowPrefixU16(delta);
    RowU16 d = v_add_wrap(p, carry);
    carry = v_add_wrap(carry, rowBcastU16(v_extract_highest(p)));
    return d;
}

inline void rowExpand16(const short* p, RowS32& lo, RowS32& hi)
{
    RowS16 v = rowLdS16(p);
    v_expand(v, lo, hi);
}

// 8-bit deltas: a 16-lane prefix stays within +-4080, so the scan runs in
// 16-bit lanes and is widened once.
inline void rowFoldU8S32(RowU16 neu, RowU16 old, RowS32& carry, RowS32& lo, RowS32& hi)
{
    RowS16 p = v_reinterpret_as_s16(rowPrefixU16(v_sub_wrap(neu, old)));
    RowS32 plo, phi;
    v_expand(p, plo, phi);
    lo = v_add(plo, carry);
    hi = v_add(phi, carry);
    carry = v_add(carry, v_broadcast_element<kRowS32 - 1>(phi));
}

inline void rowStoreU16AsS32(RowU16 neu, RowU16 old, RowS32& carry, int* dst)
{
    RowS32 lo, hi;
    rowFoldU8S32(neu, old, carry, lo, hi);
    v_store(dst, lo);
    v_store(dst + kRowS32, hi);
}

inline void rowWidenU8(const uchar* p, RowU16& lo, RowU16& hi)
{
    v_expand(rowLdU8(p), lo, hi);
}

// Per (T, ST) pair: the running-sum carry, the source elements consumed per
// vector block, and the block itself (load and widen, then fold into the carry).
template<typename T, typename ST> struct RowSlideCn1Ops;

template<> struct RowSlideCn1Ops<short, int>
{
    typedef RowS32 Carry;
    enum { step = kRowS16 };
    static Carry bcast(int v) { return rowBcastS32(v); }
    static void block(const short* sn, const short* so, Carry& c, int* D)
    {
        RowS32 nlo, nhi, olo, ohi;
        rowExpand16(sn, nlo, nhi);
        rowExpand16(so, olo, ohi);
        v_store(D, rowFoldS32(v_sub(nlo, olo), c));
        v_store(D + kRowS32, rowFoldS32(v_sub(nhi, ohi), c));
    }
};

template<> struct RowSlideCn1Ops<int, int>
{
    typedef RowS32 Carry;
    enum { step = kRowS32 };
    static Carry bcast(int v) { return rowBcastS32(v); }
    static void block(const int* sn, const int* so, Carry& c, int* D)
    {
        v_store(D, rowFoldS32(v_sub(rowLdS32(sn), rowLdS32(so)), c));
    }
};

template<> struct RowSlideCn1Ops<uchar, ushort>
{
    typedef RowU16 Carry;
    enum { step = kRowU8 };
    static Carry bcast(ushort v) { return rowBcastU16(v); }
    static void block(const uchar* sn, const uchar* so, Carry& c, ushort* D)
    {
        RowU16 nlo, nhi, olo, ohi;
        rowWidenU8(sn, nlo, nhi);
        rowWidenU8(so, olo, ohi);
        v_store(D, rowFoldU16(v_sub_wrap(nlo, olo), c));
        v_store(D + kRowS16, rowFoldU16(v_sub_wrap(nhi, ohi), c));
    }
};

template<> struct RowSlideCn1Ops<uchar, int>
{
    typedef RowS32 Carry;
    enum { step = kRowU8 };
    static Carry bcast(int v) { return rowBcastS32(v); }
    static void block(const uchar* sn, const uchar* so, Carry& c, int* D)
    {
        RowU16 nlo, nhi, olo, ohi;
        rowWidenU8(sn, nlo, nhi);
        rowWidenU8(so, olo, ohi);
        rowStoreU16AsS32(nlo, olo, c, D);
        rowStoreU16AsS32(nhi, ohi, c, D + kRowS16);
    }
};

template<typename T, typename ST>
inline void rowSlideCn1(const T* s, ST* D, int len, int kcn)
{
    typedef RowSlideCn1Ops<T, ST> Ops;
    ST sum = 0;
    for( int i = 0; i < kcn; i++ )
        sum += (ST)s[i];
    D[0] = sum;
    const int n = len - 1;
    int j = 0;
    typename Ops::Carry c = Ops::bcast(sum);
    for( ; j + Ops::step <= n; j += Ops::step )
        Ops::block(s + j + kcn, s + j, c, D + j + 1);
    sum = v_get0(c);
    for( ; j < n; j++ )
    {
        sum += (ST)s[j + kcn] - (ST)s[j];
        D[j + 1] = sum;
    }
}

// 3 interleaved 8-bit channels: one fold-and-store per half of the widened block.
template<typename ST> struct RowSlideCn3Ops;

template<> struct RowSlideCn3Ops<ushort>
{
    typedef RowU16 Carry;
    static Carry bcast(ushort v) { return rowBcastU16(v); }
    static void foldStore(RowU16 na, RowU16 oa, RowU16 nb, RowU16 ob, RowU16 nc, RowU16 oc,
                          Carry& c0, Carry& c1, Carry& c2, ushort* D)
    {
        RowU16 da = rowFoldU16(v_sub_wrap(na, oa), c0);
        RowU16 db = rowFoldU16(v_sub_wrap(nb, ob), c1);
        RowU16 dc = rowFoldU16(v_sub_wrap(nc, oc), c2);
        v_store_interleave(D, da, db, dc);
    }
};

template<> struct RowSlideCn3Ops<int>
{
    typedef RowS32 Carry;
    static Carry bcast(int v) { return rowBcastS32(v); }
    static void foldStore(RowU16 na, RowU16 oa, RowU16 nb, RowU16 ob, RowU16 nc, RowU16 oc,
                          Carry& c0, Carry& c1, Carry& c2, int* D)
    {
        RowS32 a0, a1, b0, b1, d0, d1;
        rowFoldU8S32(na, oa, c0, a0, a1);
        rowFoldU8S32(nb, ob, c1, b0, b1);
        rowFoldU8S32(nc, oc, c2, d0, d1);
        v_store_interleave(D, a0, b0, d0);
        v_store_interleave(D + kRowS32 * 3, a1, b1, d1);
    }
};

template<typename ST>
inline void rowSlideCn3U8(const uchar* s, ST* D, int len, int kcn)
{
    typedef RowSlideCn3Ops<ST> Ops;
    ST s0 = 0, s1 = 0, s2 = 0;
    for( int i = 0; i < kcn; i += 3 )
    {
        s0 += (ST)s[i]; s1 += (ST)s[i + 1]; s2 += (ST)s[i + 2];
    }
    D[0] = s0; D[1] = s1; D[2] = s2;
    const int n = len / 3 - 1;
    int j = 0;
    typename Ops::Carry c0 = Ops::bcast(s0), c1 = Ops::bcast(s1), c2 = Ops::bcast(s2);
    for( ; j + kRowU8 <= n; j += kRowU8 )
    {
        RowU8 a, b, c, oa, ob, oc;
        v_load_deinterleave(s + j * 3 + kcn, a, b, c);
        v_load_deinterleave(s + j * 3, oa, ob, oc);
        RowU16 alo, ahi, blo, bhi, clo, chi, oalo, oahi, oblo, obhi, oclo, ochi;
        v_expand(a, alo, ahi); v_expand(b, blo, bhi); v_expand(c, clo, chi);
        v_expand(oa, oalo, oahi); v_expand(ob, oblo, obhi); v_expand(oc, oclo, ochi);
        Ops::foldStore(alo, oalo, blo, oblo, clo, oclo, c0, c1, c2, D + (j + 1) * 3);
        Ops::foldStore(ahi, oahi, bhi, obhi, chi, ochi, c0, c1, c2, D + (j + 1 + kRowS16) * 3);
    }
    s0 = v_get0(c0); s1 = v_get0(c1); s2 = v_get0(c2);
    for( ; j < n; j++ )
    {
        int i = j * 3;
        s0 += (ST)s[i + kcn] - (ST)s[i];
        s1 += (ST)s[i + kcn + 1] - (ST)s[i + 1];
        s2 += (ST)s[i + kcn + 2] - (ST)s[i + 2];
        D[i + 3] = s0; D[i + 4] = s1; D[i + 5] = s2;
    }
}

// Channels stay interleaved: one pixel is 4 ushort lanes, so the per-channel
// prefix is a scan with stride 4 and the carry is the last pixel broadcast as
// one 64-bit lane.
inline void rowSlideCn4U8U16(const uchar* s, ushort* D, int len, int kcn)
{
    ushort s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for( int i = 0; i < kcn; i += 4 )
    {
        s0 += (ushort)s[i]; s1 += (ushort)s[i + 1];
        s2 += (ushort)s[i + 2]; s3 += (ushort)s[i + 3];
    }
    D[0] = s0; D[1] = s1; D[2] = s2; D[3] = s3;
    const int n = len - 4;
    int i = 0;
    const ushort initialCarry[4] = { s0, s1, s2, s3 };
    uint64 carryBits;
    memcpy(&carryBits, initialCarry, sizeof(carryBits));
    RowU16 c = rowBcastU64AsU16(carryBits);
    for( ; i + kRowS16 <= n; i += kRowS16 )
    {
        RowU16 p = rowPrefixCn4U16(v_sub_wrap(rowLdExpandU8(s + i + kcn), rowLdExpandU8(s + i)));
        v_store(D + i + 4, v_add_wrap(p, c));
        c = v_add_wrap(c, rowBcastU64AsU16(v_extract_highest(v_reinterpret_as_u64(p))));
    }
    ushort buf[kRowS16];
    v_store(buf, c);
    s0 = buf[0]; s1 = buf[1]; s2 = buf[2]; s3 = buf[3];
    for( ; i < n; i += 4 )
    {
        s0 += (ushort)s[i + kcn] - (ushort)s[i];
        s1 += (ushort)s[i + kcn + 1] - (ushort)s[i + 1];
        s2 += (ushort)s[i + kcn + 2] - (ushort)s[i + 2];
        s3 += (ushort)s[i + kcn + 3] - (ushort)s[i + 3];
        D[i + 4] = s0; D[i + 5] = s1; D[i + 6] = s2; D[i + 7] = s3;
    }
}

// The 16S/32S 1-channel scans lose to scalar on NEON (Apple M5).
template<>
CV_BOX_NOINLINE bool RowSumSeparableSIMD<short, int>::apply(const short* s, int* D, int len, int cn, int kcn)
{
#if CV_SSE2
    if( cn == 1 ) { rowSlideCn1(s, D, len, kcn); return true; }
#endif
    if( cn == 3 ) { rowSlideScalarCn3(s, D, len, kcn); return true; }
    return false;
}

template<>
CV_BOX_NOINLINE bool RowSumSeparableSIMD<ushort, int>::apply(const ushort* s, int* D, int len, int cn, int kcn)
{
    if( cn == 1 ) { rowSlideScalarCn1(s, D, len, kcn); return true; }
    if( cn == 3 ) { rowSlideScalarCn3(s, D, len, kcn); return true; }
    return false;
}

template<>
CV_BOX_NOINLINE bool RowSumSeparableSIMD<int, int>::apply(const int* s, int* D, int len, int cn, int kcn)
{
#if CV_SSE2
    if( cn == 1 ) { rowSlideCn1(s, D, len, kcn); return true; }
#endif
    if( cn == 3 ) { rowSlideScalarCn3(s, D, len, kcn); return true; }
    return false;
}

template<>
CV_BOX_NOINLINE bool RowSumSeparableSIMD<uchar, ushort>::apply(const uchar* s, ushort* D, int len, int cn, int kcn)
{
    if( cn == 1 ) { rowSlideCn1(s, D, len, kcn); return true; }
    if( cn == 3 ) { rowSlideCn3U8(s, D, len, kcn); return true; }
    if( cn == 4 ) { rowSlideCn4U8U16(s, D, len, kcn); return true; }
    return false;
}

template<>
CV_BOX_NOINLINE bool RowSumSeparableSIMD<uchar, int>::apply(const uchar* s, int* D, int len, int cn, int kcn)
{
    if( cn == 1 ) { rowSlideCn1(s, D, len, kcn); return true; }
    if( cn == 3 ) { rowSlideCn3U8(s, D, len, kcn); return true; }
    return false;
}

#if CV_SIMD_64F
inline v_float64 rowPrefixF64(const v_float64& a)
{
    v_float64 v = v_add(a, v_rotate_left<1>(a));
#if CV_SIMD_WIDTH >= 32
    v = v_add(v, v_rotate_left<2>(v));
#endif
#if CV_SIMD_WIDTH >= 64
    v = v_add(v, v_rotate_left<4>(v));
#endif
    return v;
}

// Native vector width: 256-bit doubles with AVX2, 512-bit with AVX-512.
inline void rowSlideCn1F32F64(const float* S, double* D, int width, int ksz_cn)
{
    const int VF32 = VTraits<v_float32>::vlanes();
    const int VF64 = VTraits<v_float64>::vlanes();

    int i = 0;
    v_float64 vsum0 = vx_setzero_f64(), vsum1 = vx_setzero_f64();
    for( ; i <= ksz_cn - 2*VF32; i += 2*VF32 )
    {
        v_float32 va = vx_load(S + i);
        v_float32 vb = vx_load(S + i + VF32);
        vsum0 = v_add(vsum0, v_add(v_cvt_f64(va), v_cvt_f64_high(va)));
        vsum1 = v_add(vsum1, v_add(v_cvt_f64(vb), v_cvt_f64_high(vb)));
    }
    for( ; i <= ksz_cn - VF32; i += VF32 )
    {
        v_float32 v = vx_load(S + i);
        vsum0 = v_add(vsum0, v_add(v_cvt_f64(v), v_cvt_f64_high(v)));
    }
    double s = v_reduce_sum(v_add(vsum0, vsum1));
    for( ; i < ksz_cn; i++ )
        s += (double)S[i];
    D[0] = s;

    // Both halves are scanned independently; only the final add per
    // block depends on the running sum, which keeps the carried chain short.
    int j = 0;
    for( ; j <= width - VF32; j += VF32 )
    {
        v_float32 vnew = vx_load(S + j + ksz_cn);
        v_float32 vold = vx_load(S + j);
        v_float64 vp0 = rowPrefixF64(v_sub(v_cvt_f64(vnew), v_cvt_f64(vold)));
        v_float64 vp1 = rowPrefixF64(v_sub(v_cvt_f64_high(vnew), v_cvt_f64_high(vold)));
        vp1 = v_add(vp1, vx_setall_f64(v_extract_highest(vp0)));
        v_float64 vc = vx_setall_f64(s);
        v_store(D + j + 1, v_add(vp0, vc));
        v_float64 vr1 = v_add(vp1, vc);
        v_store(D + j + 1 + VF64, vr1);
        s = v_extract_highest(vr1);
    }
    for( ; j < width; j++ )
    {
        s += (double)S[j + ksz_cn] - (double)S[j];
        D[j + 1] = s;
    }
}

template<>
CV_BOX_NOINLINE bool RowSumSeparableSIMD<float, double>::apply(const float* s, double* D, int len, int cn, int kcn)
{
    if( cn == 1 ) { rowSlideCn1F32F64(s, D, len - 1, kcn); return true; }
    return false;
}
#endif // CV_SIMD_64F
#endif // CV_SIMD && !CV_SIMD_SCALABLE

#undef CV_BOX_NOINLINE

// 64-byte loop alignment improves performance for the 3/5-tap rows, the fused
// kernels, and the column sum.
#if defined(__GNUC__) && !defined(__clang__) && defined(__x86_64__)
#pragma GCC push_options
#pragma GCC optimize("align-loops=64")
#endif
template<typename T, typename ST>
struct RowSum :
        public BaseRowFilter
{
    RowSum( int _ksize, int _anchor )
    {
        ksize = _ksize;
        anchor = _anchor;
    }

    bool isStateless() const CV_OVERRIDE { return true; }

    virtual void operator()(const uchar* src, uchar* dst, int width, int cn, bool processInnerRegion) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();

        const T* s = (const T*)src;
        ST* D = (ST*)dst;
        int i = 0, k, kcn = ksize*cn;

        int len = width*cn;
        if( ksize == 3 )
        {
            if(processInnerRegion)
            {
                for(i = 0; i < len ; i++ )
                {
                    D[i] = (ST)s[i] + (ST)s[i+cn] + (ST)s[i+cn*2];
                }
            }
            else
            {
                for(i = 0; i < cn ; i++ )
                {
                    D[i] = (ST)s[i] + (ST)s[i+cn] + (ST)s[i+cn*2];
                }
                for(i=len-cn; i < len ; i++ )
                {
                    D[i] = (ST)s[i] + (ST)s[i+cn] + (ST)s[i+cn*2];
                }
            }
        }
        else if( ksize == 5 )
        {
            if(processInnerRegion)
            {
                for(i = 0; i < len ; i++ )
                {
                    D[i] = (ST)s[i] + (ST)s[i+cn] + (ST)s[i+cn*2] + (ST)s[i + cn*3] + (ST)s[i + cn*4];
                }
            }
            else
            {
                for(i = 0; i < min(len,2*cn) ; i++ )
                {
                    D[i] = (ST)s[i] + (ST)s[i+cn] + (ST)s[i+cn*2] + (ST)s[i + cn*3] + (ST)s[i + cn*4];
                }
                for(i = max(len-(2*cn),0); i < len ; i++ )
                {
                    D[i] = (ST)s[i] + (ST)s[i+cn] + (ST)s[i+cn*2] + (ST)s[i + cn*3] + (ST)s[i + cn*4];
                }
            }
        }
        else if( (cn == 1 || cn == 3 || cn == 4) &&
                 RowSumSeparableSIMD<T, ST>::apply(s, D, len, cn, kcn) )
        {
        }
        else if( cn == 1 )
            rowSlideScalarCn1(s, D, len, kcn);
        else if( cn == 3 )
            rowSlideScalarCn3(s, D, len, kcn);
        else if( cn == 4 )
        {
            ST s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            for( i = 0; i < kcn; i += 4 )
            {
                s0 += (ST)s[i];
                s1 += (ST)s[i+1];
                s2 += (ST)s[i+2];
                s3 += (ST)s[i+3];
            }
            D[0] = s0;
            D[1] = s1;
            D[2] = s2;
            D[3] = s3;
            for( i = 0; i < len-cn; i += 4 )
            {
                s0 += (ST)s[i + kcn] - (ST)s[i];
                s1 += (ST)s[i + kcn + 1] - (ST)s[i + 1];
                s2 += (ST)s[i + kcn + 2] - (ST)s[i + 2];
                s3 += (ST)s[i + kcn + 3] - (ST)s[i + 3];
                D[i+4] = s0;
                D[i+5] = s1;
                D[i+6] = s2;
                D[i+7] = s3;
            }
        }
        else
        {
            for( k = 0; k < cn; k++, s++, D++ )
            {
                ST sum = 0;
                for( i = 0; i < kcn; i += cn )
                {
                    sum += (ST)s[i];
                }
                D[0] = sum;

                for( i = 0; i < len-cn; i += cn )
                {
                    sum += (ST)s[i + kcn] - (ST)s[i];
                    D[i+cn] = sum;
                }

            }

        }
    }
};

enum { SKIP_SCALING = 0, APPLY_SCALING = 1 };
#include "box_filter_2d.simd.hpp"

#if (CV_SIMD || CV_SIMD_SCALABLE)
struct ColumnSumVx
{
    typedef v_int32 vi;
    typedef v_float32 vf;
    typedef v_uint16 vu16;
    typedef v_uint32 vu32;
    static inline vi ld( const int* p ) { return vx_load(p); }
    static inline vu16 ld( const ushort* p ) { return vx_load(p); }
    static inline vf setf( float v ) { return vx_setall_f32(v); }
    static inline vu16 setu16( ushort v ) { return vx_setall_u16(v); }
    static inline vu32 setu32( unsigned v ) { return vx_setall_u32(v); }
};

#if !CV_SIMD_SCALABLE && CV_SIMD_WIDTH > 16
struct ColumnSumV4
{
    typedef v_int32x4 vi;
    typedef v_float32x4 vf;
    typedef v_uint16x8 vu16;
    typedef v_uint32x4 vu32;
    static inline vi ld( const int* p ) { return v_load(p); }
    static inline vu16 ld( const ushort* p ) { return v_load(p); }
    static inline vf setf( float v ) { return v_setall_f32(v); }
    static inline vu16 setu16( ushort v ) { return v_setall_u16(v); }
    static inline vu32 setu32( unsigned v ) { return v_setall_u32(v); }
};
#endif

// Converts a window sum (int32 lanes) into the value written to D.
struct ColumnSumNoScale
{
    template<typename VI, typename VF> static inline VI apply( const VI& s, const VF& ) { return s; }
};
struct ColumnSumMulRound
{
    template<typename VI, typename VF> static inline VI apply( const VI& s, const VF& vs ) { return v_round(v_mul(v_cvt_f32(s), vs)); }
};
struct ColumnSumCvtF
{
    template<typename VI, typename VF> static inline VF apply( const VI& s, const VF& ) { return v_cvt_f32(s); }
};
struct ColumnSumMulF
{
    template<typename VI, typename VF> static inline VF apply( const VI& s, const VF& vs ) { return v_mul(v_cvt_f32(s), vs); }
};

template<int SCALE_T, typename T> struct ColumnSumScale { typedef ColumnSumMulRound type; };
template<typename T> struct ColumnSumScale<SKIP_SCALING, T> { typedef ColumnSumNoScale type; };
template<> struct ColumnSumScale<APPLY_SCALING, float> { typedef ColumnSumMulF type; };
template<> struct ColumnSumScale<SKIP_SCALING, float> { typedef ColumnSumCvtF type; };

// How converted vectors are stored to D: narrowing types consume two int32 vectors per store.
template<typename T> struct ColumnSumPack
{
    enum { nvec = 1 };
    template<typename V> static inline void store( T* D, const V& a ) { v_store(D, a); }
};
template<> struct ColumnSumPack<uchar>
{
    enum { nvec = 2 };
    template<typename VI> static inline void store( uchar* D, const VI& a, const VI& b )
    { v_pack_store(D, v_pack(v_reinterpret_as_u32(a), v_reinterpret_as_u32(b))); }
};
template<> struct ColumnSumPack<ushort>
{
    enum { nvec = 2 };
    template<typename VI> static inline void store( ushort* D, const VI& a, const VI& b )
    { v_store(D, v_pack(v_reinterpret_as_u32(a), v_reinterpret_as_u32(b))); }
};
template<> struct ColumnSumPack<short>
{
    enum { nvec = 2 };
    template<typename VI> static inline void store( short* D, const VI& a, const VI& b )
    { v_store(D, v_pack(a, b)); }
};

template<class W>
inline int columnAccumulateLoop( int* SUM, const int* Sp, int i, int width )
{
    const int vl = VTraits<typename W::vi>::vlanes();
    for( ; i <= width - vl; i += vl )
        v_store(SUM + i, v_add(W::ld(SUM + i), W::ld(Sp + i)));
    return i;
}

template<class W>
inline int columnAccumulateLoop( ushort* SUM, const ushort* Sp, int i, int width )
{
    const int vl = VTraits<typename W::vu16>::vlanes();
    for( ; i <= width - vl; i += vl )
        v_store(SUM + i, v_add(W::ld(SUM + i), W::ld(Sp + i)));
    return i;
}

template<typename ST>
inline int columnAccumulate( ST* SUM, const ST* Sp, int width )
{
    int i = columnAccumulateLoop<ColumnSumVx>(SUM, Sp, 0, width);
#if !CV_SIMD_SCALABLE && CV_SIMD_WIDTH > 16
    i = columnAccumulateLoop<ColumnSumV4>(SUM, Sp, i, width);
#endif
    return i;
}

template<class W, class Scale, typename T>
inline int columnSumRowLoop( const int* Sp, const int* Sm, int* SUM, T* D, int i, int width,
                             float scale, std::integral_constant<int, 1> )
{
    typedef typename W::vi VI;
    const int vl = VTraits<VI>::vlanes();
    const typename W::vf vs = W::setf(scale);
    for( ; i <= width - vl; i += vl )
    {
        VI s0 = v_add(W::ld(SUM + i), W::ld(Sp + i));
        ColumnSumPack<T>::store(D + i, Scale::apply(s0, vs));
        v_store(SUM + i, v_sub(s0, W::ld(Sm + i)));
    }
    return i;
}

template<class W, class Scale, typename T>
inline int columnSumRowLoop( const int* Sp, const int* Sm, int* SUM, T* D, int i, int width,
                             float scale, std::integral_constant<int, 2> )
{
    typedef typename W::vi VI;
    const int vl = VTraits<VI>::vlanes();
    const typename W::vf vs = W::setf(scale);
    for( ; i <= width - 2*vl; i += 2*vl )
    {
        VI s0 = v_add(W::ld(SUM + i), W::ld(Sp + i));
        VI s1 = v_add(W::ld(SUM + i + vl), W::ld(Sp + i + vl));
        ColumnSumPack<T>::store(D + i, Scale::apply(s0, vs), Scale::apply(s1, vs));
        v_store(SUM + i, v_sub(s0, W::ld(Sm + i)));
        v_store(SUM + i + vl, v_sub(s1, W::ld(Sm + i + vl)));
    }
    return i;
}

template<class Scale, typename T>
inline int columnSumRow( const int* Sp, const int* Sm, int* SUM, T* D, int width, float scale )
{
    std::integral_constant<int, ColumnSumPack<T>::nvec> nvec;
    int i = columnSumRowLoop<ColumnSumVx, Scale>(Sp, Sm, SUM, D, 0, width, scale, nvec);
#if !CV_SIMD_SCALABLE && CV_SIMD_WIDTH > 16
    i = columnSumRowLoop<ColumnSumV4, Scale>(Sp, Sm, SUM, D, i, width, scale, nvec);
#endif
    return i;
}

// ushort sums of 8u rows, divided with the fixed-point (s + delta)*divScale >> SHIFT.
template<class W, int SHIFT>
inline int columnSumRowLoopU16( const ushort* Sp, const ushort* Sm, ushort* SUM, uchar* D, int i, int width,
                                int ds, int dd )
{
    typedef typename W::vu16 VU16;
    typedef typename W::vu32 VU32;
    const int vl = VTraits<VU16>::vlanes();
    const VU32 _ds4 = W::setu32((unsigned)ds);
    const VU16 _dd8 = W::setu16((ushort)dd);
    for( ; i <= width - 2*vl; i += 2*vl )
    {
        VU16 _sm0 = W::ld(Sm + i);
        VU16 _sm1 = W::ld(Sm + i + vl);

        VU16 _s0 = v_add_wrap(W::ld(SUM + i), W::ld(Sp + i));
        VU16 _s1 = v_add_wrap(W::ld(SUM + i + vl), W::ld(Sp + i + vl));

        VU32 _s00, _s01, _s10, _s11;

        v_expand(v_add(_s0, _dd8), _s00, _s01);
        v_expand(v_add(_s1, _dd8), _s10, _s11);

        _s00 = v_shr<SHIFT>(v_mul(_s00, _ds4));
        _s01 = v_shr<SHIFT>(v_mul(_s01, _ds4));
        _s10 = v_shr<SHIFT>(v_mul(_s10, _ds4));
        _s11 = v_shr<SHIFT>(v_mul(_s11, _ds4));

        _s0 = v_sub_wrap(_s0, _sm0);
        _s1 = v_sub_wrap(_s1, _sm1);

        v_store(D + i, v_pack_u(v_pack(v_reinterpret_as_s32(_s00), v_reinterpret_as_s32(_s01)),
                                v_pack(v_reinterpret_as_s32(_s10), v_reinterpret_as_s32(_s11))));
        v_store(SUM + i, _s0);
        v_store(SUM + i + vl, _s1);
    }
    return i;
}
#endif // CV_SIMD || CV_SIMD_SCALABLE

// Per (ST, T) pieces of ColumnSum: the vector part of the initial accumulation and of each
// output row (both return the first unprocessed index), and the scalar remainder of a row.
template<int SCALE_T, typename ST, typename T>
struct ColumnSumOps
{
    explicit ColumnSumOps( double _scale ) : scale(_scale) {}

    int accumulate( ST*, const ST*, int ) const { return 0; }
    int vecRow( const ST*, const ST*, ST*, T*, int ) const { return 0; }

    void scalarRow( const ST* Sp, const ST* Sm, ST* SUM, T* D, int i, int width ) const
    {
        double _scale = scale;
        if( SCALE_T == APPLY_SCALING )
        {
            for( ; i <= width - 2; i += 2 )
            {
                ST s0 = SUM[i] + Sp[i], s1 = SUM[i+1] + Sp[i+1];
                D[i] = saturate_cast<T>(s0*_scale);
                D[i+1] = saturate_cast<T>(s1*_scale);
                s0 -= Sm[i]; s1 -= Sm[i+1];
                SUM[i] = s0; SUM[i+1] = s1;
            }
        }
        else
        {
            for( ; i <= width - 2; i += 2 )
            {
                ST s0 = SUM[i] + Sp[i], s1 = SUM[i+1] + Sp[i+1];
                D[i] = saturate_cast<T>(s0);
                D[i+1] = saturate_cast<T>(s1);
                s0 -= Sm[i]; s1 -= Sm[i+1];
                SUM[i] = s0; SUM[i+1] = s1;
            }
        }
        for( ; i < width; i++ )
        {
            ST s0 = SUM[i] + Sp[i];
            D[i] = SCALE_T == APPLY_SCALING ? saturate_cast<T>(s0*_scale) : saturate_cast<T>(s0);
            SUM[i] = s0 - Sm[i];
        }
    }

    double scale;
};

template<int SCALE_T, typename T>
struct ColumnSumOpsS32
{
    explicit ColumnSumOpsS32( double _scale ) : scale(_scale) {}

    T cast( int v ) const
    {
        return SCALE_T == APPLY_SCALING ? saturate_cast<T>(v*scale) : saturate_cast<T>(v);
    }

    int accumulate( int* SUM, const int* Sp, int width ) const
    {
#if (CV_SIMD || CV_SIMD_SCALABLE)
        return columnAccumulate(SUM, Sp, width);
#else
        CV_UNUSED(SUM); CV_UNUSED(Sp); CV_UNUSED(width);
        return 0;
#endif
    }

    int vecRow( const int* Sp, const int* Sm, int* SUM, T* D, int width ) const
    {
#if (CV_SIMD || CV_SIMD_SCALABLE)
        return columnSumRow<typename ColumnSumScale<SCALE_T, T>::type>(Sp, Sm, SUM, D, width, (float)scale);
#else
        CV_UNUSED(Sp); CV_UNUSED(Sm); CV_UNUSED(SUM); CV_UNUSED(D); CV_UNUSED(width);
        return 0;
#endif
    }

    void scalarRow( const int* Sp, const int* Sm, int* SUM, T* D, int i, int width ) const
    {
        for( ; i < width; i++ )
        {
            int s0 = SUM[i] + Sp[i];
            D[i] = cast(s0);
            SUM[i] = s0 - Sm[i];
        }
    }

    double scale;
};

template<int SCALE_T> struct ColumnSumOps<SCALE_T, int, uchar> : ColumnSumOpsS32<SCALE_T, uchar>
{ explicit ColumnSumOps( double _scale ) : ColumnSumOpsS32<SCALE_T, uchar>(_scale) {} };
template<int SCALE_T> struct ColumnSumOps<SCALE_T, int, ushort> : ColumnSumOpsS32<SCALE_T, ushort>
{ explicit ColumnSumOps( double _scale ) : ColumnSumOpsS32<SCALE_T, ushort>(_scale) {} };
template<int SCALE_T> struct ColumnSumOps<SCALE_T, int, short> : ColumnSumOpsS32<SCALE_T, short>
{ explicit ColumnSumOps( double _scale ) : ColumnSumOpsS32<SCALE_T, short>(_scale) {} };
#if CV_SIMD_SCALABLE
// The unscaled int -> short vector loop crashes GCC 13 on RISC-V (see #24166).
template<> struct ColumnSumOps<SKIP_SCALING, int, short> : ColumnSumOpsS32<SKIP_SCALING, short>
{
    explicit ColumnSumOps( double _scale ) : ColumnSumOpsS32<SKIP_SCALING, short>(_scale) {}
    int vecRow( const int*, const int*, int*, short*, int ) const { return 0; }
};
#endif
template<int SCALE_T> struct ColumnSumOps<SCALE_T, int, int> : ColumnSumOpsS32<SCALE_T, int>
{ explicit ColumnSumOps( double _scale ) : ColumnSumOpsS32<SCALE_T, int>(_scale) {} };
template<int SCALE_T> struct ColumnSumOps<SCALE_T, int, float> : ColumnSumOpsS32<SCALE_T, float>
{ explicit ColumnSumOps( double _scale ) : ColumnSumOpsS32<SCALE_T, float>(_scale) {} };

template<>
struct ColumnSumOps<APPLY_SCALING, ushort, uchar>
{
    enum { SHIFT = 23 };

    explicit ColumnSumOps( double scale ) : divDelta(0), divScale(1)
    {
        if( scale != 1 )
        {
            int d = cvRound(1./scale);
            double scalef = ((double)(1 << SHIFT))/d;
            divScale = cvFloor(scalef);
            scalef -= divScale;
            divDelta = d/2;
            if( scalef < 0.5 )
                divDelta++;
            else
                divScale++;
        }
    }

    int accumulate( ushort* SUM, const ushort* Sp, int width ) const
    {
#if (CV_SIMD || CV_SIMD_SCALABLE)
        return columnAccumulate(SUM, Sp, width);
#else
        CV_UNUSED(SUM); CV_UNUSED(Sp); CV_UNUSED(width);
        return 0;
#endif
    }

    int vecRow( const ushort* Sp, const ushort* Sm, ushort* SUM, uchar* D, int width ) const
    {
#if (CV_SIMD || CV_SIMD_SCALABLE)
        int i = columnSumRowLoopU16<ColumnSumVx, SHIFT>(Sp, Sm, SUM, D, 0, width, divScale, divDelta);
#if !CV_SIMD_SCALABLE && CV_SIMD_WIDTH > 16
        i = columnSumRowLoopU16<ColumnSumV4, SHIFT>(Sp, Sm, SUM, D, i, width, divScale, divDelta);
#endif
        return i;
#else
        CV_UNUSED(Sp); CV_UNUSED(Sm); CV_UNUSED(SUM); CV_UNUSED(D); CV_UNUSED(width);
        return 0;
#endif
    }

    void scalarRow( const ushort* Sp, const ushort* Sm, ushort* SUM, uchar* D, int i, int width ) const
    {
        const int ds = divScale;
        const int dd = divDelta;
        for( ; i < width; i++ )
        {
            int s0 = SUM[i] + Sp[i];
            D[i] = (uchar)((s0 + dd)*ds >> SHIFT);
            SUM[i] = (ushort)(s0 - Sm[i]);
        }
    }

    int divDelta;
    int divScale;
};

template<int SCALE_T, typename ST, typename T>
struct ColumnSum :
        public BaseColumnFilter
{
    ColumnSum( int _ksize, int _anchor, double _scale ) :
        BaseColumnFilter(), ops(_scale)
    {
        ksize = _ksize;
        anchor = _anchor;
        sumCount = 0;
    }

    virtual void reset() CV_OVERRIDE { sumCount = 0; }

    virtual void operator()(const uchar** src, uchar* dst, int dststep, int count, int width) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();

        if( width != (int)sum.size() )
        {
            sum.resize(width);
            sumCount = 0;
        }

        ST* SUM = &sum[0];
        if( sumCount == 0 )
        {
            memset((void*)SUM, 0, width*sizeof(ST));
            for( ; sumCount < ksize - 1; sumCount++, src++ )
            {
                const ST* Sp = (const ST*)src[0];
                int i = ops.accumulate(SUM, Sp, width);
                for( ; i < width; i++ )
                    SUM[i] += Sp[i];
            }
        }
        else
        {
            CV_Assert( sumCount == ksize-1 );
            src += ksize-1;
        }

        for( ; count--; src++ )
        {
            const ST* Sp = (const ST*)src[0];
            const ST* Sm = (const ST*)src[1-ksize];
            T* D = (T*)dst;

            int i = ops.vecRow(Sp, Sm, SUM, D, width);
            ops.scalarRow(Sp, Sm, SUM, D, i, width);

            dst += dststep;
        }
    }

    ColumnSumOps<SCALE_T, ST, T> ops;
    int sumCount;
    std::vector<ST> sum;
};

#if defined(__GNUC__) && !defined(__clang__) && defined(__x86_64__)
#pragma GCC pop_options
#endif

#define VEC_ALIGN CV_MALLOC_ALIGN

template<typename ST, typename T>
inline void BlockSumBorderInplace(const T* ref, const T* S, T* R,
    const int *btab, int width, int cn, int dx1, int dx2, int borderType)
{
    int j=0, k;
    for( k=0; k < dx1*cn; k++, j++ )
    {
        if( borderType != BORDER_CONSTANT )
            R[j] = ref[btab[k]];
        else
            R[j] = 0;
    }
    for( k=0; k < width*cn; k++, j++ )
    {
        R[j] = S[k];
    }
    for( k=0; k < dx2*cn; k++, j++ )
    {
        if( borderType != BORDER_CONSTANT )
            R[j] = ref[btab[dx1*cn+k]];
        else
            R[j] = 0;
    }
}

template<typename ST, typename T>
inline int BlockSumCoreRow(const T* S, ST* SUM, T* D, ST** buf_ptr, double scale,
     int widthcn, int i, int cn, int kheight, int kwidth)
{
    bool haveScale = scale != 1;
    if( haveScale )
    {
        for( ; i < widthcn; i++ )
        {
            ST Ss = 0;
            for( int k=0; k < kwidth; k++ )
                Ss += (ST)S[i+k*cn];
            buf_ptr[kheight-1][i] = Ss;
            ST s0 = SUM[i] + Ss;
            D[i] = saturate_cast<T>(s0*scale);
            SUM[i] = s0 - buf_ptr[0][i];
        }
    }
    else
    {
        for( ; i < widthcn; i++ )
        {
            ST Ss = 0;
            for( int k=0; k < kwidth; k++ )
                Ss += (ST)S[i+k*cn];
            buf_ptr[kheight-1][i] = Ss;
            ST s0 = SUM[i] + Ss;
            D[i] = saturate_cast<T>(s0);
            SUM[i] = s0 - buf_ptr[0][i];
        }
    }
    return i;
}

template<typename ST, typename T>
inline int BlockSumCore(const T* S, ST* SUM, T* D, ST** buf_ptr, double scale,
     int widthcn, int i, int cn, int kheight, int kwidth)
{
    bool haveScale = scale != 1;
    switch(kwidth)
    {
        case 3:
            if( haveScale )
            {
                for( ; i < widthcn; i++ )
                {
                    ST Sp = (ST)S[i] + (ST)S[i+1*cn] + (ST)S[i+2*cn];
                    buf_ptr[kheight-1][i] = Sp;
                    ST s0 = SUM[i] + Sp;
                    D[i] = saturate_cast<T>(s0*scale);
                    SUM[i] = s0 - buf_ptr[0][i];
                }
            }
            else
            {
                for( ; i < widthcn; i++ )
                {
                    ST Sp = (ST)S[i] + (ST)S[i+1*cn] + (ST)S[i+2*cn];
                    buf_ptr[kheight-1][i] = Sp;
                    ST s0 = SUM[i] + Sp;
                    D[i] = saturate_cast<T>(s0);
                    SUM[i] = s0 - buf_ptr[0][i];
                }
            }
            break;
        case 5:
            if( haveScale )
            {
                for( ; i < widthcn; i++ )
                {
                    ST Sp = (ST)S[i] + (ST)S[i+1*cn] + (ST)S[i+2*cn] + (ST)S[i+3*cn] + (ST)S[i+4*cn];
                    buf_ptr[kheight-1][i] = Sp;
                    ST s0 = SUM[i] + Sp;
                    D[i] = saturate_cast<T>(s0*scale);
                    SUM[i] = s0 - buf_ptr[0][i];
                }
            }
            else
            {
                for( ; i < widthcn; i++ )
                {
                    ST Sp = (ST)S[i] + (ST)S[i+1*cn] + (ST)S[i+2*cn] + (ST)S[i+3*cn] + (ST)S[i+4*cn];
                    buf_ptr[kheight-1][i] = Sp;
                    ST s0 = SUM[i] + Sp;
                    D[i] = saturate_cast<T>(s0);
                    SUM[i] = s0 - buf_ptr[0][i];
                }
            }
            break;
        case 1:
            i = BlockSumCoreRow(S, SUM, D, buf_ptr, scale, widthcn, i, cn, kheight, 1);
            break;
        case 2:
            i = BlockSumCoreRow(S, SUM, D, buf_ptr, scale, widthcn, i, cn, kheight, 2);
            break;
        case 4:
            i = BlockSumCoreRow(S, SUM, D, buf_ptr, scale, widthcn, i, cn, kheight, 4);
            break;
        default:
            CV_Error_( cv::Error::StsNotImplemented,
                ("Unsupported kernel width (=%d)", kwidth));
            break;
    }
    return i;
}

uchar* alignCore(uchar* ptr) // align core region to VEC_ALIGN
{
    return alignPtr((uchar*)ptr, VEC_ALIGN);
}

template<typename ST, typename T>
void blockSum(const Mat& _src, Mat& _dst, Size ksize, Point anchor, const Size &wsz,
     const Point &ofs, double scale, int borderType, int sumType)
{
    int i, j;
    cv::utils::BufferArea area, area2;
    int* borderTab = 0;
    uchar* buf = 0, *constBorder = 0;
    uchar* border = 0, *inplaceSrc = 0, *inplaceDst = 0;
    std::vector<ST*> bufPtr;
    const uchar* src = _src.ptr();
    uchar* dst = _dst.ptr();
    Size wholeSize = wsz;
    Rect roi = Rect(ofs, _src.size());
    CV_Assert( roi.x >= 0 && roi.y >= 0 && roi.width >= 0 && roi.height >= 0 &&
        roi.x + roi.width <= wholeSize.width &&
        roi.y + roi.height <= wholeSize.height );

    size_t srcStep = _src.step;
    size_t dstStep = _dst.step;
    int srcType = _src.type();
    int cn = CV_MAT_CN(srcType);
    int sesz = (int)getElemSize(srcType);
    int besz = (int)getElemSize(sumType);
    int kwidth = ksize.width;
    int kheight = ksize.height;
    int borderLength = std::max(kwidth - 1, 1);
    int width = roi.width;
    int height = roi.height;
    int width1 = width + kwidth - 1;
    int height1 = height + kheight - 1;
    int xofs1 = std::min(roi.x, anchor.x);
    int dx1 = std::max(anchor.x - roi.x, 0);
    int dx2 = std::max(kwidth - anchor.x - 1 + roi.x + roi.width - wholeSize.width, 0);
    int dy2 = std::max(kheight - anchor.y - 1 + roi.y + roi.height - wholeSize.height, 0);
    int borderLeft = min(dx1, width)*cn;
    int bufWidth = (width1+borderLeft+VEC_ALIGN);
    int bufStep = bufWidth*besz;

    src -= xofs1*sesz;
    area.allocate(borderTab, borderLength*cn);
    area.allocate(buf, bufWidth*(kheight+1)*besz);
    area.allocate(constBorder, bufWidth*sesz);
    area.commit();
    area.zeroFill();

    // compute border tables
    if( dx1 > 0 || dx2 > 0 )
    {
        if( borderType != BORDER_CONSTANT )
        {
            int xofs1w = std::min(roi.x, anchor.x) - roi.x;
            int wholeWidth = wholeSize.width;
            int* btabx = borderTab;

            for( i = 0; i < dx1; i++ )
            {
                int p0 = (borderInterpolate(i-dx1, wholeWidth, borderType) + xofs1w)*cn;
                for( j = 0; j < cn; j++ )
                    btabx[i*cn + j] = p0 + j;
            }

            for( i = 0; i < dx2; i++ )
            {
                int p0 = (borderInterpolate(wholeWidth+i, wholeWidth, borderType) + xofs1w)*cn;
                for( j = 0; j < cn; j++ )
                    btabx[(i + dx1)*cn + j] = p0 + j;
            }
        }
    }


    // In place, the bottom border rows are read after dst has overwritten them, so they come from
    // a copy taken up front. The border table may point anywhere in the parent row, so copy whole rows.
    const bool inplace = _src.data == _dst.data;
    const int srcLeft = (roi.x - xofs1)*sesz;
    const int snapStep = wholeSize.width*sesz;
    area2.allocate(border, bufWidth*sesz);
    area2.allocate(inplaceDst, bufWidth*sesz);
    if( inplace && dy2 )
        area2.allocate(inplaceSrc, dy2*snapStep);
    area2.commit();
    if( inplace && dy2 && borderType != BORDER_CONSTANT )
    {
        for( int idx=0; idx < dy2; ++idx )
        {
            int rc = height1-(dy2-idx);
            int srcY = borderInterpolate(rc + roi.y - anchor.y, wholeSize.height, borderType);
            memcpy(inplaceSrc + idx*snapStep, src + (srcY-roi.y)*(ptrdiff_t)srcStep - srcLeft, snapStep);
        }
    }

    bufPtr.resize(kheight);

    const T *S;
    const T* ref;
    T* D;
    T* R;
    ST** buf_ptr = &bufPtr[0];
    const int *btab = borderTab;
    T* C = (T*)alignCore(constBorder);
    ST* SUM = (ST*)alignCore(&buf[bufStep*kheight]);
    for( int k=0;k<kheight;k++ )
        buf_ptr[k] = (ST*)alignCore(&buf[bufStep*k]);

    for( int rc=0, bbi=0; rc < height1; rc++ )
    {
        int srcY = borderInterpolate(rc + roi.y - anchor.y, wholeSize.height, borderType);
        if( srcY < 0 ) // can happen only with constant border type
            ref = (const T*)(C);
        else if( inplace && rc >= height1-dy2 )
            ref = (const T*)(inplaceSrc + (rc - (height1-dy2))*snapStep + srcLeft);
        else
            ref = (const T*)(src+(srcY-roi.y)*(ptrdiff_t)srcStep);

        S = ref;
        R = (T*)alignPtr(border, VEC_ALIGN);
        if ( rc < (kheight-1) )
            D = (T*)alignCore(inplaceDst);
        else
            D = (T*)dst;
        for( int k=0; k < kheight; k++ )
            buf_ptr[k] = (ST*)alignCore(&buf[((bbi+k)%kheight)*bufStep]);

        int widthcn = width*cn;
        i=0;
        BlockSumBorderInplace<ST,T>(ref, S, R, btab, width + kwidth - 1 - dx1 - dx2, cn, dx1, dx2, borderType);
        BlockSumCore<ST,T>(R, SUM, D, buf_ptr, scale, widthcn, i, cn, kheight, kwidth);


        bbi++; bbi %= kheight;
        if( rc >= (kheight-1) )
            dst += dstStep;
    }
}

}  // namespace anon

Ptr<BaseRowFilter> getRowSumFilter(int srcType, int sumType, int ksize, int anchor)
{
    CV_INSTRUMENT_REGION();

    int sdepth = CV_MAT_DEPTH(srcType), ddepth = CV_MAT_DEPTH(sumType);
    CV_Assert( CV_MAT_CN(sumType) == CV_MAT_CN(srcType) );

    if( anchor < 0 )
        anchor = ksize/2;

    if( sdepth == CV_8U && ddepth == CV_32S )
        return makePtr<RowSum<uchar, int> >(ksize, anchor);
    if( sdepth == CV_8U && ddepth == CV_16U )
        return makePtr<RowSum<uchar, ushort> >(ksize, anchor);
    if( sdepth == CV_8U && ddepth == CV_64F )
        return makePtr<RowSum<uchar, double> >(ksize, anchor);
    if( sdepth == CV_16U && ddepth == CV_32S )
        return makePtr<RowSum<ushort, int> >(ksize, anchor);
    if( sdepth == CV_16U && ddepth == CV_64F )
        return makePtr<RowSum<ushort, double> >(ksize, anchor);
    if( sdepth == CV_16S && ddepth == CV_32S )
        return makePtr<RowSum<short, int> >(ksize, anchor);
    if( sdepth == CV_32S && ddepth == CV_32S )
        return makePtr<RowSum<int, int> >(ksize, anchor);
    if( sdepth == CV_16S && ddepth == CV_64F )
        return makePtr<RowSum<short, double> >(ksize, anchor);
    if( sdepth == CV_32F && ddepth == CV_64F )
        return makePtr<RowSum<float, double> >(ksize, anchor);
    if( sdepth == CV_64F && ddepth == CV_64F )
        return makePtr<RowSum<double, double> >(ksize, anchor);

    CV_Error_( cv::Error::StsNotImplemented,
        ("Unsupported combination of source format (=%d), and buffer format (=%d)",
        srcType, sumType));
}


Ptr<BaseColumnFilter> getColumnSumFilter(int sumType, int dstType, int ksize, int anchor, double scale)
{
    CV_INSTRUMENT_REGION();

    int sdepth = CV_MAT_DEPTH(sumType), ddepth = CV_MAT_DEPTH(dstType);
    CV_Assert( CV_MAT_CN(sumType) == CV_MAT_CN(dstType) );

    if( anchor < 0 )
        anchor = ksize/2;

    if(scale==1)//box
    {
        if( ddepth == CV_8U && sdepth == CV_32S )
            return makePtr<ColumnSum<SKIP_SCALING, int, uchar> >(ksize, anchor, scale);
        if( ddepth == CV_8U && sdepth == CV_16U )
            return makePtr<ColumnSum<SKIP_SCALING, ushort, uchar> >(ksize, anchor, scale);
        if( ddepth == CV_8U && sdepth == CV_64F )
            return makePtr<ColumnSum<SKIP_SCALING, double, uchar> >(ksize, anchor, scale);
        if( ddepth == CV_16U && sdepth == CV_32S )
            return makePtr<ColumnSum<SKIP_SCALING, int, ushort> >(ksize, anchor, scale);
        if( ddepth == CV_16U && sdepth == CV_64F )
            return makePtr<ColumnSum<SKIP_SCALING, double, ushort> >(ksize, anchor, scale);
        if( ddepth == CV_16S && sdepth == CV_32S )
            return makePtr<ColumnSum<SKIP_SCALING, int, short> >(ksize, anchor, scale);
        if( ddepth == CV_16S && sdepth == CV_64F )
            return makePtr<ColumnSum<SKIP_SCALING, double, short> >(ksize, anchor, scale);
        if( ddepth == CV_32S && sdepth == CV_32S )
            return makePtr<ColumnSum<SKIP_SCALING, int, int> >(ksize, anchor, scale);
        if( ddepth == CV_32F && sdepth == CV_32S )
            return makePtr<ColumnSum<SKIP_SCALING, int, float> >(ksize, anchor, scale);
        if( ddepth == CV_32F && sdepth == CV_64F )
            return makePtr<ColumnSum<SKIP_SCALING, double, float> >(ksize, anchor, scale);
        if( ddepth == CV_64F && sdepth == CV_32S )
            return makePtr<ColumnSum<SKIP_SCALING, int, double> >(ksize, anchor, scale);
        if( ddepth == CV_64F && sdepth == CV_64F )
            return makePtr<ColumnSum<SKIP_SCALING, double, double> >(ksize, anchor, scale);
    }
    else
    {
        if( ddepth == CV_8U && sdepth == CV_32S )
            return makePtr<ColumnSum<APPLY_SCALING, int, uchar> >(ksize, anchor, scale);
        if( ddepth == CV_8U && sdepth == CV_16U )
            return makePtr<ColumnSum<APPLY_SCALING, ushort, uchar> >(ksize, anchor, scale);
        if( ddepth == CV_8U && sdepth == CV_64F )
            return makePtr<ColumnSum<APPLY_SCALING, double, uchar> >(ksize, anchor, scale);
        if( ddepth == CV_16U && sdepth == CV_32S )
            return makePtr<ColumnSum<APPLY_SCALING, int, ushort> >(ksize, anchor, scale);
        if( ddepth == CV_16U && sdepth == CV_64F )
            return makePtr<ColumnSum<APPLY_SCALING, double, ushort> >(ksize, anchor, scale);
        if( ddepth == CV_16S && sdepth == CV_32S )
            return makePtr<ColumnSum<APPLY_SCALING, int, short> >(ksize, anchor, scale);
        if( ddepth == CV_16S && sdepth == CV_64F )
            return makePtr<ColumnSum<APPLY_SCALING, double, short> >(ksize, anchor, scale);
        if( ddepth == CV_32S && sdepth == CV_32S )
            return makePtr<ColumnSum<APPLY_SCALING, int, int> >(ksize, anchor, scale);
        if( ddepth == CV_32F && sdepth == CV_32S )
            return makePtr<ColumnSum<APPLY_SCALING, int, float> >(ksize, anchor, scale);
        if( ddepth == CV_32F && sdepth == CV_64F )
            return makePtr<ColumnSum<APPLY_SCALING, double, float> >(ksize, anchor, scale);
        if( ddepth == CV_64F && sdepth == CV_32S )
            return makePtr<ColumnSum<APPLY_SCALING, int, double> >(ksize, anchor, scale);
        if( ddepth == CV_64F && sdepth == CV_64F )
            return makePtr<ColumnSum<APPLY_SCALING, double, double> >(ksize, anchor, scale);
    }

    CV_Error_( cv::Error::StsNotImplemented,
        ("Unsupported combination of sum format (=%d), and destination format (=%d)",
        sumType, dstType));
}

Ptr<BaseRowColumnFilter> getRowColumnSumFilter(int srcType, int dstType, int ksize, double scale)
{
    CV_INSTRUMENT_REGION();

    int sdepth = CV_MAT_DEPTH(srcType), ddepth = CV_MAT_DEPTH(dstType);
    CV_Assert( CV_MAT_CN(srcType) == CV_MAT_CN(dstType) );

    if(sdepth == ddepth)
    {
        if( ksize == 3 )
        {
            if(scale==1)
            {
                // the separable path is faster for 8- and 16-bit sums on NEON (Apple M5)
#if defined(__x86_64__) || defined(_M_X64)
                if( sdepth == CV_8U  )
                    return makePtr<Sum3x3<SKIP_SCALING, uchar, ushort, v_uint8, v_uint16> >(ksize, scale);
                if( sdepth == CV_16U  )
                    return makePtr<Sum3x3<SKIP_SCALING, ushort, uint32_t, v_uint16, v_uint32> >(ksize, scale);
                if( sdepth == CV_16S )
                    return makePtr<Sum3x3<SKIP_SCALING, short, int32_t, v_int16, v_int32> >(ksize, scale);
#endif
                if( sdepth == CV_32S )
                    return makePtr<Sum3x3sameType<SKIP_SCALING,  int, v_int32> >(ksize, scale);
                if( sdepth == CV_32F )
                    return makePtr<Sum3x3sameType<SKIP_SCALING, float, v_float32> >(ksize, scale);
#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
                if( sdepth == CV_64F )
                    return makePtr<Sum3x3sameType<SKIP_SCALING, double,v_float64> >(ksize, scale);
#endif
            }
            else
            {
#if defined(__x86_64__) || defined(_M_X64)
                if( sdepth == CV_8U  )
                    return makePtr<Sum3x3_8u>(ksize, scale);
                if( sdepth == CV_16U  )
                    return makePtr<Sum3x3<APPLY_SCALING, ushort, uint32_t, v_uint16, v_uint32> >(ksize, scale);
                if( sdepth == CV_16S )
                    return makePtr<Sum3x3<APPLY_SCALING, short, int32_t, v_int16, v_int32> >(ksize, scale);
#endif
                if( sdepth == CV_32S )
                    return makePtr<Sum3x3sameType<APPLY_SCALING, int, v_int32> >(ksize, scale);
                if( sdepth == CV_32F )
                    return makePtr<Sum3x3sameType<APPLY_SCALING, float, v_float32> >(ksize, scale);
#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
                if( sdepth == CV_64F )
                    return makePtr<Sum3x3sameType<APPLY_SCALING, double, v_float64> >(ksize, scale);
#endif
            }
        }
        else if (ksize == 5)
        {
            if(scale==1)
            {
#if defined(__x86_64__) || defined(_M_X64)
                if( sdepth == CV_8U  )
                    return makePtr<Sum5x5<SKIP_SCALING, uchar, ushort, v_uint8, v_uint16> >(ksize, scale);
                if( sdepth == CV_16U  )
                    return makePtr<Sum5x5<SKIP_SCALING, ushort, uint32_t, v_uint16, v_uint32> >(ksize, scale);
                if( sdepth == CV_16S )
                    return makePtr<Sum5x5<SKIP_SCALING, short, int32_t, v_int16, v_int32> >(ksize, scale);
#endif
                if( sdepth == CV_32S )
                    return makePtr<Sum5x5sameType<SKIP_SCALING, int, v_int32> >(ksize, scale); //intermediate stored can be stored in int64_t, but matching reference code to avoid output mismatch.
                if( sdepth == CV_32F )
                    return makePtr<Sum5x5sameType<SKIP_SCALING, float, v_float32> >(ksize, scale);
#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
                if( sdepth == CV_64F )
                    return makePtr<Sum5x5sameType<SKIP_SCALING, double, v_float64> >(ksize, scale);
#endif
            }
            else
            {
#if defined(__x86_64__) || defined(_M_X64)
                if( sdepth == CV_8U  )
                    return makePtr<Sum5x5_8u>(ksize, scale);
                if( sdepth == CV_16U  )
                    return makePtr<Sum5x5<APPLY_SCALING, ushort, uint32_t, v_uint16, v_uint32> >(ksize, scale);
                if( sdepth == CV_16S )
                    return makePtr<Sum5x5<APPLY_SCALING, short, int32_t, v_int16, v_int32> >(ksize, scale);
#endif
                if( sdepth == CV_32S )
                    return makePtr<Sum5x5sameType<APPLY_SCALING, int, v_int32> >(ksize, scale);
                if( sdepth == CV_32F )
                    return makePtr<Sum5x5sameType<APPLY_SCALING, float, v_float32> >(ksize, scale);
#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
                if( sdepth == CV_64F )
                    return makePtr<Sum5x5sameType<APPLY_SCALING, double, v_float64> >(ksize, scale);
#endif
            }
        }
    }
    return NULL;
}

Ptr<FilterEngine> createBoxFilter(int srcType, int dstType, Size ksize,
                                  Point anchor, bool normalize, int borderType)
{
    CV_INSTRUMENT_REGION();

    int sdepth = CV_MAT_DEPTH(srcType);
    int cn = CV_MAT_CN(srcType), sumType = CV_64F;
    if( sdepth == CV_8U && CV_MAT_DEPTH(dstType) == CV_8U &&
        ksize.width*ksize.height <= 256 )
        sumType = CV_16U;
    else if( sdepth <= CV_32S && (!normalize ||
        ksize.width*ksize.height <= (sdepth == CV_8U ? (1<<23) :
            sdepth == CV_16U ? (1 << 15) : (1 << 16))) )
        sumType = CV_32S;
    sumType = CV_MAKETYPE( sumType, cn );

    Ptr<BaseRowFilter> rowFilter = getRowSumFilter(srcType, sumType, ksize.width, anchor.x );

    Ptr<BaseColumnFilter> columnFilter = getColumnSumFilter(sumType,
        dstType, ksize.height, anchor.y, normalize ? 1./(ksize.width*ksize.height) : 1);

    Ptr<BaseRowColumnFilter> rowColumnFilter = nullptr;
    if(ksize.width == ksize.height &&
       (ksize.width ==3 || ksize.width ==5))
    {
        rowColumnFilter = getRowColumnSumFilter(srcType, dstType, ksize.width, normalize ? 1./(ksize.width*ksize.height) : 1);
    }
    return makePtr<FilterEngine>(Ptr<BaseFilter>(), rowFilter, columnFilter, rowColumnFilter,
           srcType, dstType, sumType, borderType );
}

void blockSum(const Mat& src, Mat& dst, Size ksize, Point anchor,
    const Size &wsz, const Point &ofs, bool normalize, int borderType)
{
    CV_INSTRUMENT_REGION();

    int cn = CV_MAT_CN(src.type()), sumType = CV_64F;
    sumType = CV_MAKETYPE( sumType, cn );

    CV_Assert( CV_MAT_CN(src.type()) == CV_MAT_CN(dst.type()) );
    int sdepth = CV_MAT_DEPTH(sumType), ddepth = CV_MAT_DEPTH(dst.type());

    if( anchor.x < 0 )
        anchor.x = ksize.width/2;

    if( anchor.y < 0 )
        anchor.y = ksize.height/2;

    double scale = normalize ? 1./(ksize.width*ksize.height) : 1;

    if( ddepth == CV_32F && sdepth == CV_64F )
        blockSum<double, float> (src, dst, ksize, anchor, wsz, ofs, scale, borderType, sumType);
    else if( ddepth == CV_64F && sdepth == CV_64F )
        blockSum<double, double> (src, dst, ksize, anchor, wsz, ofs, scale, borderType, sumType);
    else
        CV_Error_( cv::Error::StsNotImplemented,
            ("Unsupported combination of sum format (=%d), and destination format (=%d)",
            sumType, dst.type()));
}

/****************************************************************************************\
                                    Squared Box Filter
\****************************************************************************************/
namespace {

template<typename T, typename ST>
struct SqrRowSum :
        public BaseRowFilter
{
    SqrRowSum( int _ksize, int _anchor )
    {
        ksize = _ksize;
        anchor = _anchor;
    }

    bool isStateless() const CV_OVERRIDE { return true; }

    virtual void operator()(const uchar* src, uchar* dst, int width, int cn, bool) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();

        const T* S = (const T*)src;
        ST* D = (ST*)dst;
        int i = 0, k, ksz_cn = ksize*cn;

        width = (width - 1)*cn;
        for( k = 0; k < cn; k++, S++, D++ )
        {
            ST s = 0;
            for( i = 0; i < ksz_cn; i += cn )
            {
                ST val = (ST)S[i];
                s += val*val;
            }
            D[0] = s;
            for( i = 0; i < width; i += cn )
            {
                ST val0 = (ST)S[i], val1 = (ST)S[i + ksz_cn];
                s += val1*val1 - val0*val0;
                D[i+cn] = s;
            }
        }
    }
};

} // namespace anon

Ptr<BaseRowFilter> getSqrRowSumFilter(int srcType, int sumType, int ksize, int anchor)
{
    int sdepth = CV_MAT_DEPTH(srcType), ddepth = CV_MAT_DEPTH(sumType);
    CV_Assert( CV_MAT_CN(sumType) == CV_MAT_CN(srcType) );

    if( anchor < 0 )
        anchor = ksize/2;

    if( sdepth == CV_8U && ddepth == CV_32S )
        return makePtr<SqrRowSum<uchar, int> >(ksize, anchor);
    if( sdepth == CV_8U && ddepth == CV_64F )
        return makePtr<SqrRowSum<uchar, double> >(ksize, anchor);
    if( sdepth == CV_16U && ddepth == CV_64F )
        return makePtr<SqrRowSum<ushort, double> >(ksize, anchor);
    if( sdepth == CV_16S && ddepth == CV_64F )
        return makePtr<SqrRowSum<short, double> >(ksize, anchor);
    if( sdepth == CV_32F && ddepth == CV_64F )
        return makePtr<SqrRowSum<float, double> >(ksize, anchor);
    if( sdepth == CV_64F && ddepth == CV_64F )
        return makePtr<SqrRowSum<double, double> >(ksize, anchor);

    CV_Error_( cv::Error::StsNotImplemented,
              ("Unsupported combination of source format (=%d), and buffer format (=%d)",
               srcType, sumType));
}

#endif
CV_CPU_OPTIMIZATION_NAMESPACE_END
} // namespace
