// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// Copyright (C) 2026, Advanced Micro Devices, Inc., all rights reserved.

#include "precomp.hpp"
#include "stat.hpp"

#include <algorithm>

namespace cv {

typedef void (*MinMaxIdxFunc)(const void*, const uchar*, void*, void*, size_t*, size_t*, int, size_t);

CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

MinMaxIdxFunc getMinmaxTab(int depth);

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

template<typename T, typename WT> static void
minMaxIdx_( const T* src, const uchar* mask, WT* _minVal, WT* _maxVal,
            size_t* _minIdx, size_t* _maxIdx, int len, size_t startIdx )
{
    WT minVal = *_minVal, maxVal = *_maxVal;
    size_t minIdx = *_minIdx, maxIdx = *_maxIdx;

    if( !mask )
    {
        for( int i = 0; i < len; i++ )
        {
            T val = src[i];
            if( val < minVal )
            {
                minVal = val;
                minIdx = startIdx + i;
            }
            if( val > maxVal )
            {
                maxVal = val;
                maxIdx = startIdx + i;
            }
        }
    }
    else
    {
        for( int i = 0; i < len; i++ )
        {
            T val = src[i];
            if( mask[i] && val < minVal )
            {
                minVal = val;
                minIdx = startIdx + i;
            }
            if( mask[i] && val > maxVal )
            {
                maxVal = val;
                maxIdx = startIdx + i;
            }
        }
    }

    *_minIdx = minIdx;
    *_maxIdx = maxIdx;
    *_minVal = minVal;
    *_maxVal = maxVal;
}

#if (CV_SIMD || CV_SIMD_SCALABLE)
template<typename T, typename WT> CV_ALWAYS_INLINE void
minMaxIdx_init( const T* src, const uchar* mask, WT* minval, WT* maxval,
                size_t* minidx, size_t* maxidx, WT &minVal, WT &maxVal,
                size_t &minIdx, size_t &maxIdx, const WT minInit, const WT maxInit,
                const int nlanes, int len, size_t startidx, int &j, int &len0 )
{
    len0 = len & -nlanes;
    j = 0;

    minVal = *minval, maxVal = *maxval;
    minIdx = *minidx, maxIdx = *maxidx;

    // To handle start values out of range
    if ( minVal < minInit || maxVal < minInit || minVal > maxInit || maxVal > maxInit )
    {
        uchar done = 0x00;

        for ( ; (j < len) && (done != 0x03); j++ )
        {
            if ( !mask || mask[j] ) {
                T val = src[j];
                if ( val < minVal )
                {
                    minVal = val;
                    minIdx = startidx + j;
                    done |= 0x01;
                }
                if ( val > maxVal )
                {
                    maxVal = val;
                    maxIdx = startidx + j;
                    done |= 0x02;
                }
            }
        }

        len0 = j + ((len - j) & -nlanes);
    }
}

template<typename T, typename WT> CV_ALWAYS_INLINE void
minMaxIdx_finish( const T* src, const uchar* mask, WT* minval, WT* maxval,
                  size_t* minidx, size_t* maxidx, WT minVal, WT maxVal,
                  size_t minIdx, size_t maxIdx, int len, size_t startidx,
                  int j )
{
    for ( ; j < len ; j++ )
    {
        if ( !mask || mask[j] )
        {
            T val = src[j];
            if ( val < minVal )
            {
                minVal = val;
                minIdx = startidx + j;
            }
            if ( val > maxVal )
            {
                maxVal = val;
                maxIdx = startidx + j;
            }
        }
    }

    *minidx = minIdx;
    *maxidx = maxIdx;
    *minval = minVal;
    *maxval = maxVal;
}

//============================================================================
// Templated SIMD core (universal intrinsics) shared by every depth. The
// per-depth wrappers below only select the value/index vector + result types.
//============================================================================

// Fast single-instruction broadcast into any universal vector (value or index).
template<typename V> static inline V mm_set(typename VTraits<V>::lane_type v);
template<> inline v_uint8   mm_set<v_uint8  >(uchar v)    { return vx_setall_u8(v); }
template<> inline v_int8    mm_set<v_int8   >(schar v)    { return vx_setall_s8(v); }
template<> inline v_uint16  mm_set<v_uint16 >(ushort v)   { return vx_setall_u16(v); }
template<> inline v_int16   mm_set<v_int16  >(short v)    { return vx_setall_s16(v); }
template<> inline v_uint32  mm_set<v_uint32 >(uint v)     { return vx_setall_u32(v); }
template<> inline v_int32   mm_set<v_int32  >(int v)      { return vx_setall_s32(v); }
template<> inline v_float32 mm_set<v_float32>(float v)    { return vx_setall_f32(v); }
#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
template<> inline v_uint64  mm_set<v_uint64 >(uint64_t v) { return vx_setall_u64(v); }
template<> inline v_float64 mm_set<v_float64>(double v)   { return vx_setall_f64(v); }
#endif

// Reinterpret a value-domain comparison mask to the matching unsigned index vector.
static inline v_uint8  mm_idxmask(const v_uint8&   m) { return m; }
static inline v_uint8  mm_idxmask(const v_int8&    m) { return v_reinterpret_as_u8(m); }
static inline v_uint16 mm_idxmask(const v_uint16&  m) { return m; }
static inline v_uint16 mm_idxmask(const v_int16&   m) { return v_reinterpret_as_u16(m); }
static inline v_uint32 mm_idxmask(const v_int32&   m) { return v_reinterpret_as_u32(m); }
static inline v_uint32 mm_idxmask(const v_float32& m) { return v_reinterpret_as_u32(m); }
#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
static inline v_uint64 mm_idxmask(const v_float64& m) { return v_reinterpret_as_u64(m); }
#endif

// Blend for index vectors (universal API has no v_select for u64).
template<typename IT> static inline IT mm_sel(const IT& mask, const IT& a, const IT& b)
{ return v_xor(b, v_and(v_xor(a, b), mask)); }

// Value reduce-min / reduce-max (f64 has no universal v_reduce_*).
static inline int    mm_vmin(const v_uint8&  v){ return v_reduce_min(v); }
static inline int    mm_vmin(const v_int8&   v){ return v_reduce_min(v); }
static inline int    mm_vmin(const v_uint16& v){ return v_reduce_min(v); }
static inline int    mm_vmin(const v_int16&  v){ return v_reduce_min(v); }
static inline int    mm_vmin(const v_int32&  v){ return v_reduce_min(v); }
static inline float  mm_vmin(const v_float32&v){ return v_reduce_min(v); }
static inline int    mm_vmax(const v_uint8&  v){ return v_reduce_max(v); }
static inline int    mm_vmax(const v_int8&   v){ return v_reduce_max(v); }
static inline int    mm_vmax(const v_uint16& v){ return v_reduce_max(v); }
static inline int    mm_vmax(const v_int16&  v){ return v_reduce_max(v); }
static inline int    mm_vmax(const v_int32&  v){ return v_reduce_max(v); }
static inline float  mm_vmax(const v_float32&v){ return v_reduce_max(v); }
#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
static inline double mm_vmin(const v_float64& v)
{ double b[VTraits<v_float64>::max_nlanes]; v_store(b, v); double r=b[0]; const int n=VTraits<v_float64>::vlanes(); for(int i=1;i<n;i++) if(b[i]<r) r=b[i]; return r; }
static inline double mm_vmax(const v_float64& v)
{ double b[VTraits<v_float64>::max_nlanes]; v_store(b, v); double r=b[0]; const int n=VTraits<v_float64>::vlanes(); for(int i=1;i<n;i++) if(b[i]>r) r=b[i]; return r; }
#endif

// Index reduce-min (u64 has no universal v_reduce_min).
static inline unsigned mm_imin(const v_uint8&  v){ return v_reduce_min(v); }
static inline unsigned mm_imin(const v_uint16& v){ return v_reduce_min(v); }
static inline unsigned mm_imin(const v_uint32& v){ return v_reduce_min(v); }
#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
static inline uint64_t mm_imin(const v_uint64& v)
{ uint64_t b[VTraits<v_uint64>::max_nlanes]; v_store(b, v); uint64_t r=b[0]; const int n=VTraits<v_uint64>::vlanes(); for(int i=1;i<n;i++) if(b[i]<r) r=b[i]; return r; }
#endif

// Active-lane mask in the *value* vector domain, from a uchar* mask.
template<typename VT> static inline VT mm_active(const uchar* mask, int k, int nlanes);
template<> inline v_uint8  mm_active<v_uint8 >(const uchar* mask, int k, int)
{ return v_ne(vx_load(mask + k), vx_setzero_u8()); }
template<> inline v_int8   mm_active<v_int8  >(const uchar* mask, int k, int)
{ return v_reinterpret_as_s8(v_ne(vx_load(mask + k), vx_setzero_u8())); }
template<> inline v_uint16 mm_active<v_uint16>(const uchar* mask, int k, int)
{ return v_ne(vx_load_expand(mask + k), vx_setzero_u16()); }
template<> inline v_int16  mm_active<v_int16 >(const uchar* mask, int k, int)
{ return v_reinterpret_as_s16(v_ne(vx_load_expand(mask + k), vx_setzero_u16())); }
template<> inline v_int32  mm_active<v_int32 >(const uchar* mask, int k, int nlanes)
{ uint32_t b[VTraits<v_uint32>::max_nlanes]; for(int t=0;t<nlanes;t++) b[t]=mask[k+t]?~0u:0u; return v_reinterpret_as_s32(vx_load(b)); }
template<> inline v_float32 mm_active<v_float32>(const uchar* mask, int k, int nlanes)
{ uint32_t b[VTraits<v_uint32>::max_nlanes]; for(int t=0;t<nlanes;t++) b[t]=mask[k+t]?~0u:0u; return v_reinterpret_as_f32(vx_load(b)); }
#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
template<> inline v_float64 mm_active<v_float64>(const uchar* mask, int k, int nlanes)
{ uint64_t b[VTraits<v_uint64>::max_nlanes]; for(int t=0;t<nlanes;t++) b[t]=mask[k+t]?~0ull:0ull; return v_reinterpret_as_f64(vx_load(b)); }
#endif

// Fold one accumulator stream's (value,index) extremum into the running result.
// Scalar tie-break (smaller index wins) lets several independent streams combine
// correctly without any vector index comparison (works for u64 indices too).
template<typename VT, typename IT, typename WT>
static inline void mm_fold_min(const VT& valMin, const IT& idxMin, const IT& none,
                               size_t delta, WT& minVal, size_t& minIdx)
{
    if ( v_check_any(v_ne(idxMin, none)) )
    {
        WT cv = (WT)mm_vmin(valMin);
        IT sel = mm_sel(mm_idxmask(v_eq(mm_set<VT>((typename VTraits<VT>::lane_type)cv), valMin)), idxMin, none);
        size_t ci = (size_t)mm_imin(sel) + delta;
        if ( cv < minVal || (cv == minVal && ci < minIdx) ) { minVal = cv; minIdx = ci; }
    }
}
template<typename VT, typename IT, typename WT>
static inline void mm_fold_max(const VT& valMax, const IT& idxMax, const IT& none,
                               size_t delta, WT& maxVal, size_t& maxIdx)
{
    if ( v_check_any(v_ne(idxMax, none)) )
    {
        WT cv = (WT)mm_vmax(valMax);
        IT sel = mm_sel(mm_idxmask(v_eq(mm_set<VT>((typename VTraits<VT>::lane_type)cv), valMax)), idxMax, none);
        size_t ci = (size_t)mm_imin(sel) + delta;
        if ( cv > maxVal || (cv == maxVal && ci < maxIdx) ) { maxVal = cv; maxIdx = ci; }
    }
}

// IST = unsigned index lane type (uchar / ushort / uint / uint64_t).
template<typename T, typename VT, typename IT, typename IST, typename WT>
static void minMaxIdx_simd_(const T* src, const uchar* mask, WT* minval, WT* maxval,
                            size_t* minidx, size_t* maxidx, int len, size_t startidx,
                            WT minInit, WT maxInit)
{
    const int nlanes = VTraits<VT>::vlanes();
    if ( len >= nlanes )
    {
        int j, len0;
        WT minVal, maxVal;
        size_t minIdx, maxIdx;

        minMaxIdx_init( src, mask, minval, maxval, minidx, maxidx, minVal, maxVal, minIdx, maxIdx,
                        minInit, maxInit, nlanes, len, startidx, j, len0 );

        if ( j <= len0 - nlanes )
        {
            IST idxbuf[VTraits<IT>::max_nlanes];
            for ( int t = 0; t < nlanes; t++ ) idxbuf[t] = (IST)t;
            const IT idxStart = vx_load(idxbuf);
            const IT inc  = mm_set<IT>((IST)nlanes);
            const IT none = mm_set<IT>((IST)~(IST)0);
            // Reduce before the per-lane index could reach the 'none' sentinel.
            // For >= 32-bit indices (len <= INT_MAX) one block covers everything.
            const int64_t idxcap = (sizeof(IST) <= 2) ? (int64_t)std::numeric_limits<IST>::max() : (int64_t)INT_MAX;
            const int blockStep = (int)((idxcap / nlanes) * nlanes);

            const IT inc2 = mm_set<IT>((IST)(nlanes * 2));

            do
            {
                // Two independent accumulator streams break the serial
                // min/max + index-select dependency chain so the CPU can
                // overlap iterations (latency-bound loop). Stream 0 covers the
                // even vector slots, stream 1 the odd ones; results are merged
                // by mm_fold_*. (2 streams is the sweet spot: 4 spills the 16
                // YMM regs on AVX2 and gives no gain on AVX-512.) The masked
                // path keeps a single stream (stream 0).
                VT vMin0 = mm_set<VT>((T)minVal), vMin1 = vMin0;
                VT vMax0 = mm_set<VT>((T)maxVal), vMax1 = vMax0;
                IT idx0 = idxStart, idx1 = v_add(idxStart, inc);
                IT iMin0 = none, iMin1 = none, iMax0 = none, iMax1 = none;

                int k = j;
                size_t delta = startidx + j;
                // 64-bit math: blockStep can be ~INT_MAX (32/64-bit indices), so
                // j + blockStep would overflow int once j advances past the start.
                const int limit = (int)std::min<int64_t>((int64_t)len0, (int64_t)j + (int64_t)blockStep);

                if ( !mask )
                {
                    // Dual stream only for >= 16-bit indices: 8-bit forces tiny
                    // reduce-blocks (u8 index < 256), where extra per-block folds
                    // outweigh the dependency-break benefit.
                    for ( ; (sizeof(IST) > 1) && k <= limit - 2 * nlanes; k += 2 * nlanes )
                    {
                        VT d0 = vx_load(src + k), d1 = vx_load(src + k + nlanes);
                        iMin0 = mm_sel(mm_idxmask(v_lt(d0, vMin0)), idx0, iMin0);
                        iMin1 = mm_sel(mm_idxmask(v_lt(d1, vMin1)), idx1, iMin1);
                        iMax0 = mm_sel(mm_idxmask(v_gt(d0, vMax0)), idx0, iMax0);
                        iMax1 = mm_sel(mm_idxmask(v_gt(d1, vMax1)), idx1, iMax1);
                        vMin0 = v_min(d0, vMin0); vMin1 = v_min(d1, vMin1);
                        vMax0 = v_max(d0, vMax0); vMax1 = v_max(d1, vMax1);
                        idx0 = v_add(idx0, inc2); idx1 = v_add(idx1, inc2);
                    }
                    for ( ; k < limit; k += nlanes ) // odd trailing vector
                    {
                        VT d0 = vx_load(src + k);
                        iMin0 = mm_sel(mm_idxmask(v_lt(d0, vMin0)), idx0, iMin0);
                        iMax0 = mm_sel(mm_idxmask(v_gt(d0, vMax0)), idx0, iMax0);
                        vMin0 = v_min(d0, vMin0); vMax0 = v_max(d0, vMax0);
                        idx0 = v_add(idx0, inc);
                    }
                }
                else
                {
                    for ( ; k < limit; k += nlanes )
                    {
                        VT data = vx_load(src + k);
                        VT active = mm_active<VT>(mask, k, nlanes);
                        VT cmpMin = v_and(v_lt(data, vMin0), active);
                        VT cmpMax = v_and(v_gt(data, vMax0), active);
                        iMin0 = mm_sel(mm_idxmask(cmpMin), idx0, iMin0);
                        iMax0 = mm_sel(mm_idxmask(cmpMax), idx0, iMax0);
                        vMin0 = v_select(cmpMin, data, vMin0);
                        vMax0 = v_select(cmpMax, data, vMax0);
                        idx0 = v_add(idx0, inc);
                    }
                }

                j = k;

                mm_fold_min(vMin0, iMin0, none, delta, minVal, minIdx);
                mm_fold_min(vMin1, iMin1, none, delta, minVal, minIdx);
                mm_fold_max(vMax0, iMax0, none, delta, maxVal, maxIdx);
                mm_fold_max(vMax1, iMax1, none, delta, maxVal, maxIdx);
            }
            while ( j < len0 );
        }

        minMaxIdx_finish( src, mask, minval, maxval, minidx, maxidx, minVal, maxVal,
                          minIdx, maxIdx, len, startidx, j );
        vx_cleanup();
    }
    else
    {
        minMaxIdx_(src, mask, minval, maxval, minidx, maxidx, len, startidx);
    }
}
#endif

static void minMaxIdx_8u(const void* src_, const uchar* mask, void* minval_, void* maxval_,
                         size_t* minidx, size_t* maxidx, int len, size_t startidx )
{
#if (CV_SIMD || CV_SIMD_SCALABLE)
    minMaxIdx_simd_<uchar, v_uint8, v_uint8, uchar, int>(
        static_cast<const uchar*>(src_), mask, static_cast<int*>(minval_), static_cast<int*>(maxval_),
        minidx, maxidx, len, startidx, (int)0, (int)UCHAR_MAX);
#else
    minMaxIdx_(static_cast<const uchar*>(src_), mask, static_cast<int*>(minval_),
               static_cast<int*>(maxval_), minidx, maxidx, len, startidx);
#endif
}

static void minMaxIdx_8s(const void* src_, const uchar* mask, void* minval_, void* maxval_,
                         size_t* minidx, size_t* maxidx, int len, size_t startidx )
{
#if (CV_SIMD || CV_SIMD_SCALABLE)
    minMaxIdx_simd_<schar, v_int8, v_uint8, uchar, int>(
        static_cast<const schar*>(src_), mask, static_cast<int*>(minval_), static_cast<int*>(maxval_),
        minidx, maxidx, len, startidx, (int)SCHAR_MIN, (int)SCHAR_MAX);
#else
    minMaxIdx_(static_cast<const schar*>(src_), mask, static_cast<int*>(minval_),
               static_cast<int*>(maxval_), minidx, maxidx, len, startidx);
#endif
}

static void minMaxIdx_16u(const void* src_, const uchar* mask, void* minval_, void* maxval_,
                          size_t* minidx, size_t* maxidx, int len, size_t startidx )
{
#if (CV_SIMD || CV_SIMD_SCALABLE)
    minMaxIdx_simd_<ushort, v_uint16, v_uint16, ushort, int>(
        static_cast<const ushort*>(src_), mask, static_cast<int*>(minval_), static_cast<int*>(maxval_),
        minidx, maxidx, len, startidx, (int)0, (int)USHRT_MAX);
#else
    minMaxIdx_(static_cast<const ushort*>(src_), mask, static_cast<int*>(minval_),
               static_cast<int*>(maxval_), minidx, maxidx, len, startidx);
#endif
}

static void minMaxIdx_16s(const void* src_, const uchar* mask, void* minval_, void* maxval_,
                          size_t* minidx, size_t* maxidx, int len, size_t startidx )
{
#if (CV_SIMD || CV_SIMD_SCALABLE)
    minMaxIdx_simd_<short, v_int16, v_uint16, ushort, int>(
        static_cast<const short*>(src_), mask, static_cast<int*>(minval_), static_cast<int*>(maxval_),
        minidx, maxidx, len, startidx, (int)SHRT_MIN, (int)SHRT_MAX);
#else
    minMaxIdx_(static_cast<const short*>(src_), mask, static_cast<int*>(minval_),
               static_cast<int*>(maxval_), minidx, maxidx, len, startidx);
#endif
}

static void minMaxIdx_32s(const void* src_, const uchar* mask, void* minval_, void* maxval_,
                          size_t* minidx, size_t* maxidx, int len, size_t startidx )
{
#if (CV_SIMD || CV_SIMD_SCALABLE)
    minMaxIdx_simd_<int, v_int32, v_uint32, uint, int>(
        static_cast<const int*>(src_), mask, static_cast<int*>(minval_), static_cast<int*>(maxval_),
        minidx, maxidx, len, startidx, INT_MIN, INT_MAX);
#else
    minMaxIdx_(static_cast<const int*>(src_), mask, static_cast<int*>(minval_),
               static_cast<int*>(maxval_), minidx, maxidx, len, startidx);
#endif
}

static void minMaxIdx_32f(const void* src_, const uchar* mask, void* minval_, void* maxval_,
                          size_t* minidx, size_t* maxidx, int len, size_t startidx )
{
#if (CV_SIMD || CV_SIMD_SCALABLE)
    minMaxIdx_simd_<float, v_float32, v_uint32, uint, float>(
        static_cast<const float*>(src_), mask, static_cast<float*>(minval_), static_cast<float*>(maxval_),
        minidx, maxidx, len, startidx, FLT_MIN, FLT_MAX);
#else
    minMaxIdx_(static_cast<const float*>(src_), mask, static_cast<float*>(minval_),
               static_cast<float*>(maxval_), minidx, maxidx, len, startidx);
#endif
}

static void minMaxIdx_64f(const void* src_, const uchar* mask, void* minval_, void* maxval_,
                          size_t* minidx, size_t* maxidx, int len, size_t startidx )
{
#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
    minMaxIdx_simd_<double, v_float64, v_uint64, uint64_t, double>(
        static_cast<const double*>(src_), mask, static_cast<double*>(minval_), static_cast<double*>(maxval_),
        minidx, maxidx, len, startidx, DBL_MIN, DBL_MAX);
#else
    minMaxIdx_(static_cast<const double*>(src_), mask, static_cast<double*>(minval_),
               static_cast<double*>(maxval_), minidx, maxidx, len, startidx);
#endif
}

MinMaxIdxFunc getMinmaxTab(int depth)
{
    static MinMaxIdxFunc minmaxTab[CV_DEPTH_MAX] =
    {
        GET_OPTIMIZED(minMaxIdx_8u), GET_OPTIMIZED(minMaxIdx_8s),
        GET_OPTIMIZED(minMaxIdx_16u), GET_OPTIMIZED(minMaxIdx_16s),
        GET_OPTIMIZED(minMaxIdx_32s),
        GET_OPTIMIZED(minMaxIdx_32f), GET_OPTIMIZED(minMaxIdx_64f),
        0
    };

    return minmaxTab[depth];
}

#endif // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

CV_CPU_OPTIMIZATION_NAMESPACE_END
} // namespace cv
