// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
// Copyright (C) 2026, Advanced Micro Devices, Inc., all rights reserved.

#include "precomp.hpp"
#include "opencv2/core/hal/intrin.hpp"

namespace cv {

typedef void (*MomentsInTileFunc)(const Mat& img, double* moments);

template<typename T, typename WT, typename MT>
void momentsInTileAccumulateRow(const T* ptr, int x0, int len, WT& s0, WT& s1, WT& s2, MT& s3);

CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

MomentsInTileFunc getMomentsInTileFunc(int depth);

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

template<typename T, typename WT, typename MT>
struct MomentsInTile_SIMD
{
    int operator() (const T *, int, WT &, WT &, WT &, MT &) const
    {
        return 0;
    }
};

#if CV_SIMD128

template <>
struct MomentsInTile_SIMD<uchar, int, int>
{
    int operator() (const uchar * ptr, int len, int & x0, int & x1, int & x2, int & x3) const
    {
        int x = 0;

        v_int16x8 dx = v_setall_s16(8), qx = v_int16x8(0, 1, 2, 3, 4, 5, 6, 7);
        v_uint32x4 z = v_setzero_u32(), qx0 = z, qx1 = z, qx2 = z, qx3 = z;

        for( ; x <= len - 8; x += 8 )
        {
            v_int16x8 p = v_reinterpret_as_s16(v_load_expand(ptr + x));
            v_int16x8 sx = v_mul_wrap(qx, qx);

            qx0 = v_add(qx0, v_reinterpret_as_u32(p));
            qx1 = v_reinterpret_as_u32(v_dotprod(p, qx, v_reinterpret_as_s32(qx1)));
            qx2 = v_reinterpret_as_u32(v_dotprod(p, sx, v_reinterpret_as_s32(qx2)));
            qx3 = v_reinterpret_as_u32(v_dotprod(v_mul_wrap(p, qx), sx, v_reinterpret_as_s32(qx3)));

            qx = v_add(qx, dx);
        }

        x0 = v_reduce_sum(qx0);
        x0 = (x0 & 0xffff) + (x0 >> 16);
        x1 = v_reduce_sum(qx1);
        x2 = v_reduce_sum(qx2);
        x3 = v_reduce_sum(qx3);

        return x;
    }
};

#endif // CV_SIMD128

#if (CV_SIMD || CV_SIMD_SCALABLE)

static inline v_int32 vx_load_iota_s32()
{
    static int CV_DECL_ALIGNED(CV_SIMD_WIDTH) data[VTraits<v_int32>::max_nlanes];
    static bool initialized = false;
    if (!initialized)
    {
        for (int i = 0; i < VTraits<v_int32>::max_nlanes; ++i)
            data[i] = i;
        initialized = true;
    }
    return vx_load(data);
}

#if !CV_SIMD128

template <>
struct MomentsInTile_SIMD<uchar, int, int>
{
    int operator() (const uchar * ptr, int len, int & x0, int & x1, int & x2, int & x3) const
    {
        int x = 0;
        const int vlanes32 = VTraits<v_int32>::vlanes();

        v_int32 v_delta = vx_setall_s32(vlanes32);
        v_int32 v_ix0 = vx_load_iota_s32();
        v_int32 v_x0 = vx_setzero_s32(), v_x1 = vx_setzero_s32();
        v_int32 v_x2 = vx_setzero_s32(), v_x3 = vx_setzero_s32();

        for ( ; x <= len - vlanes32; x += vlanes32 )
        {
            v_int32 v_src = v_reinterpret_as_s32(vx_load_expand(ptr + x));

            v_x0 = v_add(v_x0, v_src);
            v_x1 = v_add(v_x1, v_mul(v_src, v_ix0));

            v_int32 v_ix1 = v_mul(v_ix0, v_ix0);
            v_x2 = v_add(v_x2, v_mul(v_src, v_ix1));

            v_ix1 = v_mul(v_ix0, v_ix1);
            v_x3 = v_add(v_x3, v_mul(v_src, v_ix1));

            v_ix0 = v_add(v_ix0, v_delta);
        }

        x0 = v_reduce_sum(v_x0);
        x1 = v_reduce_sum(v_x1);
        x2 = v_reduce_sum(v_x2);
        x3 = v_reduce_sum(v_x3);
        vx_cleanup();
        return x;
    }
};

#endif // !CV_SIMD128

template<typename ST>
struct moments_vx_load_expand16
{
    static inline v_int32 fn(const ST* ptr, int x)
    { return vx_load_expand(ptr + x); }
};

template<>
struct moments_vx_load_expand16<ushort>
{
    static inline v_int32 fn(const ushort* ptr, int x)
    { return v_reinterpret_as_s32(vx_load_expand(ptr + x)); }
};

template<typename ST>
struct MomentsInTile_SIMD_16u
{
    int operator() (const ST * ptr, int len, int & x0, int & x1, int & x2, int64 & x3) const
    {
        int x = 0;
        const int vlanes32 = VTraits<v_int32>::vlanes();

        v_int32 v_delta = vx_setall_s32(vlanes32);
        v_int32 v_ix0 = vx_load_iota_s32();
        v_int32 v_x0 = vx_setzero_s32(), v_x1 = vx_setzero_s32(), v_x2 = vx_setzero_s32();
        v_int64 v_x3 = vx_setzero_s64();

        for ( ; x <= len - vlanes32; x += vlanes32 )
        {
            v_int32 v_src = moments_vx_load_expand16<ST>::fn(ptr, x);

            v_x0 = v_add(v_x0, v_src);
            v_x1 = v_add(v_x1, v_mul(v_src, v_ix0));

            v_int32 v_ix1 = v_mul(v_ix0, v_ix0);
            v_x2 = v_add(v_x2, v_mul(v_src, v_ix1));

            v_ix1 = v_mul(v_ix0, v_ix1);
            v_int32 v_src3 = v_mul(v_src, v_ix1);
            v_int64 v_lo, v_hi;
            v_expand(v_src3, v_lo, v_hi);
            v_x3 = v_add(v_x3, v_add(v_lo, v_hi));

            v_ix0 = v_add(v_ix0, v_delta);
        }

        x0 = v_reduce_sum(v_x0);
        x1 = v_reduce_sum(v_x1);
        x2 = v_reduce_sum(v_x2);
        x3 = (int64)v_reduce_sum(v_x3);
        vx_cleanup();
        return x;
    }
};

template <>
struct MomentsInTile_SIMD<ushort, int, int64> : MomentsInTile_SIMD_16u<ushort> {};

#endif // CV_SIMD || CV_SIMD_SCALABLE

#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)

template <>
struct MomentsInTile_SIMD<float, double, double>
{
    int operator() (const float * ptr, int len, double & x0, double & x1, double & x2, double & x3) const
    {
        int x = 0;
        const int vlanesf = VTraits<v_float32>::vlanes();
        const int vlanes64 = VTraits<v_float64>::vlanes();

        if (vlanesf % vlanes64 != 0 || vlanesf != 2 * vlanes64)
            return 0;

        const int nhalfs = 2;

        v_float64 v_block_delta = vx_setall_f64((double)vlanesf);
        v_float64 v_ix0;
        {
            double CV_DECL_ALIGNED(CV_SIMD_WIDTH) buf[VTraits<v_float64>::max_nlanes];
            for (int i = 0; i < vlanes64; ++i)
                buf[i] = (double)i;
            v_ix0 = vx_load(buf);
        }
        v_float64 v_x0 = vx_setzero_f64(), v_x1 = vx_setzero_f64();
        v_float64 v_x2 = vx_setzero_f64(), v_x3 = vx_setzero_f64();

        for ( ; x <= len - vlanesf; x += vlanesf )
        {
            v_float32 v_srcf = vx_load(ptr + x);

            for (int h = 0; h < nhalfs; ++h)
            {
                v_float64 v_src = h == 0 ? v_cvt_f64(v_srcf) : v_cvt_f64_high(v_srcf);
                v_float64 v_cur_ix = v_add(v_ix0, vx_setall_f64((double)(h * vlanes64)));

                v_x0 = v_add(v_x0, v_src);
                v_x1 = v_add(v_x1, v_mul(v_src, v_cur_ix));

                v_float64 v_ix1 = v_mul(v_cur_ix, v_cur_ix);
                v_x2 = v_add(v_x2, v_mul(v_src, v_ix1));

                v_ix1 = v_mul(v_cur_ix, v_ix1);
                v_x3 = v_add(v_x3, v_mul(v_src, v_ix1));
            }

            v_ix0 = v_add(v_ix0, v_block_delta);
        }

        x0 = v_reduce_sum(v_x0);
        x1 = v_reduce_sum(v_x1);
        x2 = v_reduce_sum(v_x2);
        x3 = v_reduce_sum(v_x3);
        vx_cleanup();
        return x;
    }
};

template <>
struct MomentsInTile_SIMD<double, double, double>
{
    int operator() (const double * ptr, int len, double & x0, double & x1, double & x2, double & x3) const
    {
        int x = 0;
        const int vlanes64 = VTraits<v_float64>::vlanes();

        v_float64 v_delta = vx_setall_f64((double)VTraits<v_float64>::vlanes());
        v_float64 v_ix0;
        {
            double CV_DECL_ALIGNED(CV_SIMD_WIDTH) buf[VTraits<v_float64>::max_nlanes];
            for (int i = 0; i < vlanes64; ++i)
                buf[i] = (double)i;
            v_ix0 = vx_load(buf);
        }
        v_float64 v_x0 = vx_setzero_f64(), v_x1 = vx_setzero_f64();
        v_float64 v_x2 = vx_setzero_f64(), v_x3 = vx_setzero_f64();

        for ( ; x <= len - vlanes64; x += vlanes64 )
        {
            v_float64 v_src = vx_load(ptr + x);

            v_x0 = v_add(v_x0, v_src);
            v_x1 = v_add(v_x1, v_mul(v_src, v_ix0));

            v_float64 v_ix1 = v_mul(v_ix0, v_ix0);
            v_x2 = v_add(v_x2, v_mul(v_src, v_ix1));

            v_ix1 = v_mul(v_ix0, v_ix1);
            v_x3 = v_add(v_x3, v_mul(v_src, v_ix1));

            v_ix0 = v_add(v_ix0, v_delta);
        }

        x0 = v_reduce_sum(v_x0);
        x1 = v_reduce_sum(v_x1);
        x2 = v_reduce_sum(v_x2);
        x3 = v_reduce_sum(v_x3);
        vx_cleanup();
        return x;
    }
};

#endif // CV_SIMD_64F || CV_SIMD_SCALABLE_64F

template<typename T, typename WT, typename MT>
#if defined(__GNUC__) && __GNUC__ >= 4 && (__GNUC__ > 4 || __GNUC_MINOR__ >= 5)
__attribute__((optimize("no-tree-vectorize")))
#endif
static void momentsInTile( const Mat& img, double* moments )
{
    Size size = img.size();
    int x, y;
    MT mom[10] = {0,0,0,0,0,0,0,0,0,0};
    MomentsInTile_SIMD<T, WT, MT> vop;

    for( y = 0; y < size.height; y++ )
    {
        const T* ptr = img.ptr<T>(y);
        WT x0 = 0, x1 = 0, x2 = 0;
        MT x3 = 0;
        x = vop(ptr, size.width, x0, x1, x2, x3);

        if (x < size.width)
            cv::momentsInTileAccumulateRow<T, WT, MT>(ptr, x, size.width, x0, x1, x2, x3);

        WT py = y * x0, sy = y*y;

        mom[9] += ((MT)py) * sy;  // m03
        mom[8] += ((MT)x1) * sy;  // m12
        mom[7] += ((MT)x2) * y;  // m21
        mom[6] += x3;             // m30
        mom[5] += x0 * sy;        // m02
        mom[4] += x1 * y;         // m11
        mom[3] += x2;             // m20
        mom[2] += py;             // m01
        mom[1] += x1;             // m10
        mom[0] += x0;             // m00
    }

    for( x = 0; x < 10; x++ )
        moments[x] = (double)mom[x];
}

static void momentsInTile8u( const Mat& img, double* moments )
{ CV_INSTRUMENT_REGION(); momentsInTile<uchar, int, int>(img, moments); }

static void momentsInTile16u( const Mat& img, double* moments )
{ CV_INSTRUMENT_REGION(); momentsInTile<ushort, int, int64>(img, moments); }

static void momentsInTile32f( const Mat& img, double* moments )
{ CV_INSTRUMENT_REGION(); momentsInTile<float, double, double>(img, moments); }

static void momentsInTile64f( const Mat& img, double* moments )
{ CV_INSTRUMENT_REGION(); momentsInTile<double, double, double>(img, moments); }

MomentsInTileFunc getMomentsInTileFunc(int depth)
{
    static MomentsInTileFunc momentsInTileTab[CV_DEPTH_MAX] =
    {
        (MomentsInTileFunc)GET_OPTIMIZED(momentsInTile8u),
        (MomentsInTileFunc)GET_OPTIMIZED(momentsInTile8u),
        (MomentsInTileFunc)GET_OPTIMIZED(momentsInTile16u),
        0,
        0,
        (MomentsInTileFunc)GET_OPTIMIZED(momentsInTile32f),
        (MomentsInTileFunc)GET_OPTIMIZED(momentsInTile64f),
        0
    };

    return momentsInTileTab[depth];
}

#endif // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

CV_CPU_OPTIMIZATION_NAMESPACE_END
} // namespace cv
