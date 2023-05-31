// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


#include "precomp.hpp"
#include "stat.hpp"

namespace cv {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

SumFunc getSumFunc(int depth);

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

template <typename T, typename ST>
struct Sum_SIMD
{
    int operator () (const T *, const uchar *, ST *, int, int) const
    {
        return 0;
    }
};

#if CV_SIMD

template <>
struct Sum_SIMD<uchar, int>
{
    int operator () (const uchar * src0, const uchar * mask, int * dst, int len, int cn) const
    {
        if (mask || (cn != 1 && cn != 2 && cn != 4))
            return 0;
        len *= cn;

        int x = 0;
        v_uint32 v_sum = vx_setzero_u32();

        int len0 = len & -v_uint8::nlanes;
        while (x < len0)
        {
            const int len_tmp = min(x + 256*v_uint16::nlanes, len0);
            v_uint16 v_sum16 = vx_setzero_u16();
            for (; x < len_tmp; x += v_uint8::nlanes)
            {
                v_uint16 v_src0, v_src1;
                v_expand(vx_load(src0 + x), v_src0, v_src1);
                v_sum16 += v_src0 + v_src1;
            }
            v_uint32 v_half0, v_half1;
            v_expand(v_sum16, v_half0, v_half1);
            v_sum += v_half0 + v_half1;
        }
        if (x <= len - v_uint16::nlanes)
        {
            v_uint32 v_half0, v_half1;
            v_expand(vx_load_expand(src0 + x), v_half0, v_half1);
            v_sum += v_half0 + v_half1;
            x += v_uint16::nlanes;
        }
        if (x <= len - v_uint32::nlanes)
        {
            v_sum += vx_load_expand_q(src0 + x);
            x += v_uint32::nlanes;
        }

        if (cn == 1)
            *dst += v_reduce_sum(v_sum);
        else
        {
            uint32_t CV_DECL_ALIGNED(CV_SIMD_WIDTH) ar[v_uint32::nlanes];
            v_store_aligned(ar, v_sum);
            for (int i = 0; i < v_uint32::nlanes; ++i)
                dst[i % cn] += ar[i];
        }
        v_cleanup();

        return x / cn;
    }
};

template <>
struct Sum_SIMD<schar, int>
{
    int operator () (const schar * src0, const uchar * mask, int * dst, int len, int cn) const
    {
        if (mask || (cn != 1 && cn != 2 && cn != 4))
            return 0;
        len *= cn;

        int x = 0;
        v_int32 v_sum = vx_setzero_s32();

        int len0 = len & -v_int8::nlanes;
        while (x < len0)
        {
            const int len_tmp = min(x + 256*v_int16::nlanes, len0);
            v_int16 v_sum16 = vx_setzero_s16();
            for (; x < len_tmp; x += v_int8::nlanes)
            {
                v_int16 v_src0, v_src1;
                v_expand(vx_load(src0 + x), v_src0, v_src1);
                v_sum16 += v_src0 + v_src1;
            }
            v_int32 v_half0, v_half1;
            v_expand(v_sum16, v_half0, v_half1);
            v_sum += v_half0 + v_half1;
        }
        if (x <= len - v_int16::nlanes)
        {
            v_int32 v_half0, v_half1;
            v_expand(vx_load_expand(src0 + x), v_half0, v_half1);
            v_sum += v_half0 + v_half1;
            x += v_int16::nlanes;
        }
        if (x <= len - v_int32::nlanes)
        {
            v_sum += vx_load_expand_q(src0 + x);
            x += v_int32::nlanes;
        }

        if (cn == 1)
            *dst += v_reduce_sum(v_sum);
        else
        {
            int32_t CV_DECL_ALIGNED(CV_SIMD_WIDTH) ar[v_int32::nlanes];
            v_store_aligned(ar, v_sum);
            for (int i = 0; i < v_int32::nlanes; ++i)
                dst[i % cn] += ar[i];
        }
        v_cleanup();

        return x / cn;
    }
};

template <>
struct Sum_SIMD<ushort, int>
{
    int operator () (const ushort * src0, const uchar * mask, int * dst, int len, int cn) const
    {
        if (mask || (cn != 1 && cn != 2 && cn != 4))
            return 0;
        len *= cn;

        int x = 0;
        v_uint32 v_sum = vx_setzero_u32();

        for (; x <= len - v_uint16::nlanes; x += v_uint16::nlanes)
        {
            v_uint32 v_src0, v_src1;
            v_expand(vx_load(src0 + x), v_src0, v_src1);
            v_sum += v_src0 + v_src1;
        }
        if (x <= len - v_uint32::nlanes)
        {
            v_sum += vx_load_expand(src0 + x);
            x += v_uint32::nlanes;
        }

        if (cn == 1)
            *dst += v_reduce_sum(v_sum);
        else
        {
            uint32_t CV_DECL_ALIGNED(CV_SIMD_WIDTH) ar[v_uint32::nlanes];
            v_store_aligned(ar, v_sum);
            for (int i = 0; i < v_uint32::nlanes; ++i)
                dst[i % cn] += ar[i];
        }
        v_cleanup();

        return x / cn;
    }
};

template <>
struct Sum_SIMD<short, int>
{
    int operator () (const short * src0, const uchar * mask, int * dst, int len, int cn) const
    {
        if (mask || (cn != 1 && cn != 2 && cn != 4))
            return 0;
        len *= cn;

        int x = 0;
        v_int32 v_sum = vx_setzero_s32();

        for (; x <= len - v_int16::nlanes; x += v_int16::nlanes)
        {
            v_int32 v_src0, v_src1;
            v_expand(vx_load(src0 + x), v_src0, v_src1);
            v_sum += v_src0 + v_src1;
        }
        if (x <= len - v_int32::nlanes)
        {
            v_sum += vx_load_expand(src0 + x);
            x += v_int32::nlanes;
        }

        if (cn == 1)
            *dst += v_reduce_sum(v_sum);
        else
        {
            int32_t CV_DECL_ALIGNED(CV_SIMD_WIDTH) ar[v_int32::nlanes];
            v_store_aligned(ar, v_sum);
            for (int i = 0; i < v_int32::nlanes; ++i)
                dst[i % cn] += ar[i];
        }
        v_cleanup();

        return x / cn;
    }
};

#if CV_SIMD_64F
template <>
struct Sum_SIMD<int, double>
{
    int operator () (const int * src0, const uchar * mask, double * dst, int len, int cn) const
    {
        if (mask || (cn != 1 && cn != 2 && cn != 4))
            return 0;
        len *= cn;

        int x = 0;
        v_float64 v_sum0 = vx_setzero_f64();
        v_float64 v_sum1 = vx_setzero_f64();

        for (; x <= len - 2 * v_int32::nlanes; x += 2 * v_int32::nlanes)
        {
            v_int32 v_src0 = vx_load(src0 + x);
            v_int32 v_src1 = vx_load(src0 + x + v_int32::nlanes);
            v_sum0 += v_cvt_f64(v_src0) + v_cvt_f64(v_src1);
            v_sum1 += v_cvt_f64_high(v_src0) + v_cvt_f64_high(v_src1);
        }

#if CV_SIMD256 || CV_SIMD512
        double CV_DECL_ALIGNED(CV_SIMD_WIDTH) ar[v_float64::nlanes];
        v_store_aligned(ar, v_sum0 + v_sum1);
        for (int i = 0; i < v_float64::nlanes; ++i)
            dst[i % cn] += ar[i];
#else
        double CV_DECL_ALIGNED(CV_SIMD_WIDTH) ar[2 * v_float64::nlanes];
        v_store_aligned(ar, v_sum0);
        v_store_aligned(ar + v_float64::nlanes, v_sum1);
        for (int i = 0; i < 2 * v_float64::nlanes; ++i)
            dst[i % cn] += ar[i];
#endif
        v_cleanup();

        return x / cn;
    }
};

template <>
struct Sum_SIMD<float, double>
{
    int operator () (const float * src0, const uchar * mask, double * dst, int len, int cn) const
    {
        if (mask || (cn != 1 && cn != 2 && cn != 4))
            return 0;
        len *= cn;

        int x = 0;
        v_float64 v_sum0 = vx_setzero_f64();
        v_float64 v_sum1 = vx_setzero_f64();

        for (; x <= len - 2 * v_float32::nlanes; x += 2 * v_float32::nlanes)
        {
            v_float32 v_src0 = vx_load(src0 + x);
            v_float32 v_src1 = vx_load(src0 + x + v_float32::nlanes);
            v_sum0 += v_cvt_f64(v_src0) + v_cvt_f64(v_src1);
            v_sum1 += v_cvt_f64_high(v_src0) + v_cvt_f64_high(v_src1);
        }

#if CV_SIMD256 || CV_SIMD512
        double CV_DECL_ALIGNED(CV_SIMD_WIDTH) ar[v_float64::nlanes];
        v_store_aligned(ar, v_sum0 + v_sum1);
        for (int i = 0; i < v_float64::nlanes; ++i)
            dst[i % cn] += ar[i];
#else
        double CV_DECL_ALIGNED(CV_SIMD_WIDTH) ar[2 * v_float64::nlanes];
        v_store_aligned(ar, v_sum0);
        v_store_aligned(ar + v_float64::nlanes, v_sum1);
        for (int i = 0; i < 2 * v_float64::nlanes; ++i)
            dst[i % cn] += ar[i];
#endif
        v_cleanup();

        return x / cn;
    }
};
#endif
#endif

template<typename T, typename ST>
static int sum_(const T* src0, const uchar* mask, ST* dst, int len, int cn )
{
    const T* src = src0;
    if( !mask )
    {
        Sum_SIMD<T, ST> vop;
        int i = vop(src0, mask, dst, len, cn), k = cn % 4;
        src += i * cn;

        if( k == 1 )
        {
            ST s0 = dst[0];

            #if CV_ENABLE_UNROLLED
            for(; i <= len - 4; i += 4, src += cn*4 )
                s0 += src[0] + src[cn] + src[cn*2] + src[cn*3];
            #endif
            for( ; i < len; i++, src += cn )
                s0 += src[0];
            dst[0] = s0;
        }
        else if( k == 2 )
        {
            ST s0 = dst[0], s1 = dst[1];
            for( ; i < len; i++, src += cn )
            {
                s0 += src[0];
                s1 += src[1];
            }
            dst[0] = s0;
            dst[1] = s1;
        }
        else if( k == 3 )
        {
            ST s0 = dst[0], s1 = dst[1], s2 = dst[2];
            for( ; i < len; i++, src += cn )
            {
                s0 += src[0];
                s1 += src[1];
                s2 += src[2];
            }
            dst[0] = s0;
            dst[1] = s1;
            dst[2] = s2;
        }

        for( ; k < cn; k += 4 )
        {
            src = src0 + i*cn + k;
            ST s0 = dst[k], s1 = dst[k+1], s2 = dst[k+2], s3 = dst[k+3];
            for( ; i < len; i++, src += cn )
            {
                s0 += src[0]; s1 += src[1];
                s2 += src[2]; s3 += src[3];
            }
            dst[k] = s0;
            dst[k+1] = s1;
            dst[k+2] = s2;
            dst[k+3] = s3;
        }
        return len;
    }

    int i, nzm = 0;
    if( cn == 1 )
    {
        ST s = dst[0];
        for( i = 0; i < len; i++ )
            if( mask[i] )
            {
                s += src[i];
                nzm++;
            }
        dst[0] = s;
    }
    else if( cn == 3 )
    {
        ST s0 = dst[0], s1 = dst[1], s2 = dst[2];
        for( i = 0; i < len; i++, src += 3 )
            if( mask[i] )
            {
                s0 += src[0];
                s1 += src[1];
                s2 += src[2];
                nzm++;
            }
        dst[0] = s0;
        dst[1] = s1;
        dst[2] = s2;
    }
    else
    {
        for( i = 0; i < len; i++, src += cn )
            if( mask[i] )
            {
                int k = 0;
                #if CV_ENABLE_UNROLLED
                for( ; k <= cn - 4; k += 4 )
                {
                    ST s0, s1;
                    s0 = dst[k] + src[k];
                    s1 = dst[k+1] + src[k+1];
                    dst[k] = s0; dst[k+1] = s1;
                    s0 = dst[k+2] + src[k+2];
                    s1 = dst[k+3] + src[k+3];
                    dst[k+2] = s0; dst[k+3] = s1;
                }
                #endif
                for( ; k < cn; k++ )
                    dst[k] += src[k];
                nzm++;
            }
    }
    return nzm;
}


static int sum8u( const uchar* src, const uchar* mask, int* dst, int len, int cn )
{ CV_INSTRUMENT_REGION(); return sum_(src, mask, dst, len, cn); }

static int sum8s( const schar* src, const uchar* mask, int* dst, int len, int cn )
{ CV_INSTRUMENT_REGION(); return sum_(src, mask, dst, len, cn); }

static int sum16u( const ushort* src, const uchar* mask, int* dst, int len, int cn )
{ CV_INSTRUMENT_REGION(); return sum_(src, mask, dst, len, cn); }

static int sum16s( const short* src, const uchar* mask, int* dst, int len, int cn )
{ CV_INSTRUMENT_REGION(); return sum_(src, mask, dst, len, cn); }

static int sum32s( const int* src, const uchar* mask, double* dst, int len, int cn )
{ CV_INSTRUMENT_REGION(); return sum_(src, mask, dst, len, cn); }

static int sum32f( const float* src, const uchar* mask, double* dst, int len, int cn )
{ CV_INSTRUMENT_REGION(); return sum_(src, mask, dst, len, cn); }

static int sum64f( const double* src, const uchar* mask, double* dst, int len, int cn )
{ CV_INSTRUMENT_REGION(); return sum_(src, mask, dst, len, cn); }

SumFunc getSumFunc(int depth)
{
    static SumFunc sumTab[] =
    {
        (SumFunc)GET_OPTIMIZED(sum8u), (SumFunc)sum8s,
        (SumFunc)sum16u, (SumFunc)sum16s,
        (SumFunc)sum32s,
        (SumFunc)GET_OPTIMIZED(sum32f), (SumFunc)sum64f,
        0
    };

    return sumTab[depth];
}

#endif

CV_CPU_OPTIMIZATION_NAMESPACE_END
} // namespace
