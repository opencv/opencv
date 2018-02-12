// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


#include "precomp.hpp"
#include "opencl_kernels_core.hpp"
#include "stat.hpp"

namespace cv
{

template <typename T, typename ST>
struct Sum_SIMD
{
    int operator () (const T *, const uchar *, ST *, int, int) const
    {
        return 0;
    }
};

template <typename ST, typename DT>
inline void addChannels(DT * dst, ST * buf, int cn)
{
    for (int i = 0; i < 4; ++i)
        dst[i % cn] += buf[i];
}

#if CV_SSE2

template <>
struct Sum_SIMD<schar, int>
{
    int operator () (const schar * src0, const uchar * mask, int * dst, int len, int cn) const
    {
        if (mask || (cn != 1 && cn != 2 && cn != 4) || !USE_SSE2)
            return 0;

        int x = 0;
        __m128i v_zero = _mm_setzero_si128(), v_sum = v_zero;

        for ( ; x <= len - 16; x += 16)
        {
            __m128i v_src = _mm_loadu_si128((const __m128i *)(src0 + x));
            __m128i v_half = _mm_srai_epi16(_mm_unpacklo_epi8(v_zero, v_src), 8);

            v_sum = _mm_add_epi32(v_sum, _mm_srai_epi32(_mm_unpacklo_epi16(v_zero, v_half), 16));
            v_sum = _mm_add_epi32(v_sum, _mm_srai_epi32(_mm_unpackhi_epi16(v_zero, v_half), 16));

            v_half = _mm_srai_epi16(_mm_unpackhi_epi8(v_zero, v_src), 8);
            v_sum = _mm_add_epi32(v_sum, _mm_srai_epi32(_mm_unpacklo_epi16(v_zero, v_half), 16));
            v_sum = _mm_add_epi32(v_sum, _mm_srai_epi32(_mm_unpackhi_epi16(v_zero, v_half), 16));
        }

        for ( ; x <= len - 8; x += 8)
        {
            __m128i v_src = _mm_srai_epi16(_mm_unpacklo_epi8(v_zero, _mm_loadl_epi64((__m128i const *)(src0 + x))), 8);

            v_sum = _mm_add_epi32(v_sum, _mm_srai_epi32(_mm_unpacklo_epi16(v_zero, v_src), 16));
            v_sum = _mm_add_epi32(v_sum, _mm_srai_epi32(_mm_unpackhi_epi16(v_zero, v_src), 16));
        }

        int CV_DECL_ALIGNED(16) ar[4];
        _mm_store_si128((__m128i*)ar, v_sum);

        addChannels(dst, ar, cn);

        return x / cn;
    }
};

template <>
struct Sum_SIMD<int, double>
{
    int operator () (const int * src0, const uchar * mask, double * dst, int len, int cn) const
    {
        if (mask || (cn != 1 && cn != 2 && cn != 4) || !USE_SSE2)
            return 0;

        int x = 0;
        __m128d v_zero = _mm_setzero_pd(), v_sum0 = v_zero, v_sum1 = v_zero;

        for ( ; x <= len - 4; x += 4)
        {
            __m128i v_src = _mm_loadu_si128((__m128i const *)(src0 + x));
            v_sum0 = _mm_add_pd(v_sum0, _mm_cvtepi32_pd(v_src));
            v_sum1 = _mm_add_pd(v_sum1, _mm_cvtepi32_pd(_mm_srli_si128(v_src, 8)));
        }

        double CV_DECL_ALIGNED(16) ar[4];
        _mm_store_pd(ar, v_sum0);
        _mm_store_pd(ar + 2, v_sum1);

        addChannels(dst, ar, cn);

        return x / cn;
    }
};

template <>
struct Sum_SIMD<float, double>
{
    int operator () (const float * src0, const uchar * mask, double * dst, int len, int cn) const
    {
        if (mask || (cn != 1 && cn != 2 && cn != 4) || !USE_SSE2)
            return 0;

        int x = 0;
        __m128d v_zero = _mm_setzero_pd(), v_sum0 = v_zero, v_sum1 = v_zero;

        for ( ; x <= len - 4; x += 4)
        {
            __m128 v_src = _mm_loadu_ps(src0 + x);
            v_sum0 = _mm_add_pd(v_sum0, _mm_cvtps_pd(v_src));
            v_src = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(v_src), 8));
            v_sum1 = _mm_add_pd(v_sum1, _mm_cvtps_pd(v_src));
        }

        double CV_DECL_ALIGNED(16) ar[4];
        _mm_store_pd(ar, v_sum0);
        _mm_store_pd(ar + 2, v_sum1);

        addChannels(dst, ar, cn);

        return x / cn;
    }
};


#elif CV_NEON

template <>
struct Sum_SIMD<uchar, int>
{
    int operator () (const uchar * src0, const uchar * mask, int * dst, int len, int cn) const
    {
        if (mask || (cn != 1 && cn != 2 && cn != 4))
            return 0;

        int x = 0;
        uint32x4_t v_sum = vdupq_n_u32(0u);

        for ( ; x <= len - 16; x += 16)
        {
            uint8x16_t v_src = vld1q_u8(src0 + x);
            uint16x8_t v_half = vmovl_u8(vget_low_u8(v_src));

            v_sum = vaddw_u16(v_sum, vget_low_u16(v_half));
            v_sum = vaddw_u16(v_sum, vget_high_u16(v_half));

            v_half = vmovl_u8(vget_high_u8(v_src));
            v_sum = vaddw_u16(v_sum, vget_low_u16(v_half));
            v_sum = vaddw_u16(v_sum, vget_high_u16(v_half));
        }

        for ( ; x <= len - 8; x += 8)
        {
            uint16x8_t v_src = vmovl_u8(vld1_u8(src0 + x));

            v_sum = vaddw_u16(v_sum, vget_low_u16(v_src));
            v_sum = vaddw_u16(v_sum, vget_high_u16(v_src));
        }

        unsigned int CV_DECL_ALIGNED(16) ar[4];
        vst1q_u32(ar, v_sum);

        addChannels(dst, ar, cn);

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

        int x = 0;
        int32x4_t v_sum = vdupq_n_s32(0);

        for ( ; x <= len - 16; x += 16)
        {
            int8x16_t v_src = vld1q_s8(src0 + x);
            int16x8_t v_half = vmovl_s8(vget_low_s8(v_src));

            v_sum = vaddw_s16(v_sum, vget_low_s16(v_half));
            v_sum = vaddw_s16(v_sum, vget_high_s16(v_half));

            v_half = vmovl_s8(vget_high_s8(v_src));
            v_sum = vaddw_s16(v_sum, vget_low_s16(v_half));
            v_sum = vaddw_s16(v_sum, vget_high_s16(v_half));
        }

        for ( ; x <= len - 8; x += 8)
        {
            int16x8_t v_src = vmovl_s8(vld1_s8(src0 + x));

            v_sum = vaddw_s16(v_sum, vget_low_s16(v_src));
            v_sum = vaddw_s16(v_sum, vget_high_s16(v_src));
        }

        int CV_DECL_ALIGNED(16) ar[4];
        vst1q_s32(ar, v_sum);

        addChannels(dst, ar, cn);

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

        int x = 0;
        uint32x4_t v_sum = vdupq_n_u32(0u);

        for ( ; x <= len - 8; x += 8)
        {
            uint16x8_t v_src = vld1q_u16(src0 + x);

            v_sum = vaddw_u16(v_sum, vget_low_u16(v_src));
            v_sum = vaddw_u16(v_sum, vget_high_u16(v_src));
        }

        for ( ; x <= len - 4; x += 4)
            v_sum = vaddw_u16(v_sum, vld1_u16(src0 + x));

        unsigned int CV_DECL_ALIGNED(16) ar[4];
        vst1q_u32(ar, v_sum);

        addChannels(dst, ar, cn);

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

        int x = 0;
        int32x4_t v_sum = vdupq_n_s32(0u);

        for ( ; x <= len - 8; x += 8)
        {
            int16x8_t v_src = vld1q_s16(src0 + x);

            v_sum = vaddw_s16(v_sum, vget_low_s16(v_src));
            v_sum = vaddw_s16(v_sum, vget_high_s16(v_src));
        }

        for ( ; x <= len - 4; x += 4)
            v_sum = vaddw_s16(v_sum, vld1_s16(src0 + x));

        int CV_DECL_ALIGNED(16) ar[4];
        vst1q_s32(ar, v_sum);

        addChannels(dst, ar, cn);

        return x / cn;
    }
};

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
{ return sum_(src, mask, dst, len, cn); }

static int sum8s( const schar* src, const uchar* mask, int* dst, int len, int cn )
{ return sum_(src, mask, dst, len, cn); }

static int sum16u( const ushort* src, const uchar* mask, int* dst, int len, int cn )
{ return sum_(src, mask, dst, len, cn); }

static int sum16s( const short* src, const uchar* mask, int* dst, int len, int cn )
{ return sum_(src, mask, dst, len, cn); }

static int sum32s( const int* src, const uchar* mask, double* dst, int len, int cn )
{ return sum_(src, mask, dst, len, cn); }

static int sum32f( const float* src, const uchar* mask, double* dst, int len, int cn )
{ return sum_(src, mask, dst, len, cn); }

static int sum64f( const double* src, const uchar* mask, double* dst, int len, int cn )
{ return sum_(src, mask, dst, len, cn); }

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

#ifdef HAVE_OPENCL

bool ocl_sum( InputArray _src, Scalar & res, int sum_op, InputArray _mask,
                     InputArray _src2, bool calc2, const Scalar & res2 )
{
    CV_Assert(sum_op == OCL_OP_SUM || sum_op == OCL_OP_SUM_ABS || sum_op == OCL_OP_SUM_SQR);

    const ocl::Device & dev = ocl::Device::getDefault();
    bool doubleSupport = dev.doubleFPConfig() > 0,
        haveMask = _mask.kind() != _InputArray::NONE,
        haveSrc2 = _src2.kind() != _InputArray::NONE;
    int type = _src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type),
            kercn = cn == 1 && !haveMask ? ocl::predictOptimalVectorWidth(_src, _src2) : 1,
            mcn = std::max(cn, kercn);
    CV_Assert(!haveSrc2 || _src2.type() == type);
    int convert_cn = haveSrc2 ? mcn : cn;

    if ( (!doubleSupport && depth == CV_64F) || cn > 4 )
        return false;

    int ngroups = dev.maxComputeUnits(), dbsize = ngroups * (calc2 ? 2 : 1);
    size_t wgs = dev.maxWorkGroupSize();

    int ddepth = std::max(sum_op == OCL_OP_SUM_SQR ? CV_32F : CV_32S, depth),
            dtype = CV_MAKE_TYPE(ddepth, cn);
    CV_Assert(!haveMask || _mask.type() == CV_8UC1);

    int wgs2_aligned = 1;
    while (wgs2_aligned < (int)wgs)
        wgs2_aligned <<= 1;
    wgs2_aligned >>= 1;

    static const char * const opMap[3] = { "OP_SUM", "OP_SUM_ABS", "OP_SUM_SQR" };
    char cvt[2][40];
    String opts = format("-D srcT=%s -D srcT1=%s -D dstT=%s -D dstTK=%s -D dstT1=%s -D ddepth=%d -D cn=%d"
                         " -D convertToDT=%s -D %s -D WGS=%d -D WGS2_ALIGNED=%d%s%s%s%s -D kercn=%d%s%s%s -D convertFromU=%s",
                         ocl::typeToStr(CV_MAKE_TYPE(depth, mcn)), ocl::typeToStr(depth),
                         ocl::typeToStr(dtype), ocl::typeToStr(CV_MAKE_TYPE(ddepth, mcn)),
                         ocl::typeToStr(ddepth), ddepth, cn,
                         ocl::convertTypeStr(depth, ddepth, mcn, cvt[0]),
                         opMap[sum_op], (int)wgs, wgs2_aligned,
                         doubleSupport ? " -D DOUBLE_SUPPORT" : "",
                         haveMask ? " -D HAVE_MASK" : "",
                         _src.isContinuous() ? " -D HAVE_SRC_CONT" : "",
                         haveMask && _mask.isContinuous() ? " -D HAVE_MASK_CONT" : "", kercn,
                         haveSrc2 ? " -D HAVE_SRC2" : "", calc2 ? " -D OP_CALC2" : "",
                         haveSrc2 && _src2.isContinuous() ? " -D HAVE_SRC2_CONT" : "",
                         depth <= CV_32S && ddepth == CV_32S ? ocl::convertTypeStr(CV_8U, ddepth, convert_cn, cvt[1]) : "noconvert");

    ocl::Kernel k("reduce", ocl::core::reduce_oclsrc, opts);
    if (k.empty())
        return false;

    UMat src = _src.getUMat(), src2 = _src2.getUMat(),
        db(1, dbsize, dtype), mask = _mask.getUMat();

    ocl::KernelArg srcarg = ocl::KernelArg::ReadOnlyNoSize(src),
            dbarg = ocl::KernelArg::PtrWriteOnly(db),
            maskarg = ocl::KernelArg::ReadOnlyNoSize(mask),
            src2arg = ocl::KernelArg::ReadOnlyNoSize(src2);

    if (haveMask)
    {
        if (haveSrc2)
            k.args(srcarg, src.cols, (int)src.total(), ngroups, dbarg, maskarg, src2arg);
        else
            k.args(srcarg, src.cols, (int)src.total(), ngroups, dbarg, maskarg);
    }
    else
    {
        if (haveSrc2)
            k.args(srcarg, src.cols, (int)src.total(), ngroups, dbarg, src2arg);
        else
            k.args(srcarg, src.cols, (int)src.total(), ngroups, dbarg);
    }

    size_t globalsize = ngroups * wgs;
    if (k.run(1, &globalsize, &wgs, false))
    {
        typedef Scalar (*part_sum)(Mat m);
        part_sum funcs[3] = { ocl_part_sum<int>, ocl_part_sum<float>, ocl_part_sum<double> },
                func = funcs[ddepth - CV_32S];

        Mat mres = db.getMat(ACCESS_READ);
        if (calc2)
            const_cast<Scalar &>(res2) = func(mres.colRange(ngroups, dbsize));

        res = func(mres.colRange(0, ngroups));
        return true;
    }
    return false;
}

#endif

#ifdef HAVE_IPP
static bool ipp_sum(Mat &src, Scalar &_res)
{
    CV_INSTRUMENT_REGION_IPP()

#if IPP_VERSION_X100 >= 700
    int cn = src.channels();
    if (cn > 4)
        return false;
    size_t total_size = src.total();
    int rows = src.size[0], cols = rows ? (int)(total_size/rows) : 0;
    if( src.dims == 2 || (src.isContinuous() && cols > 0 && (size_t)rows*cols == total_size) )
    {
        IppiSize sz = { cols, rows };
        int type = src.type();
        typedef IppStatus (CV_STDCALL* ippiSumFuncHint)(const void*, int, IppiSize, double *, IppHintAlgorithm);
        typedef IppStatus (CV_STDCALL* ippiSumFuncNoHint)(const void*, int, IppiSize, double *);
        ippiSumFuncHint ippiSumHint =
            type == CV_32FC1 ? (ippiSumFuncHint)ippiSum_32f_C1R :
            type == CV_32FC3 ? (ippiSumFuncHint)ippiSum_32f_C3R :
            type == CV_32FC4 ? (ippiSumFuncHint)ippiSum_32f_C4R :
            0;
        ippiSumFuncNoHint ippiSum =
            type == CV_8UC1 ? (ippiSumFuncNoHint)ippiSum_8u_C1R :
            type == CV_8UC3 ? (ippiSumFuncNoHint)ippiSum_8u_C3R :
            type == CV_8UC4 ? (ippiSumFuncNoHint)ippiSum_8u_C4R :
            type == CV_16UC1 ? (ippiSumFuncNoHint)ippiSum_16u_C1R :
            type == CV_16UC3 ? (ippiSumFuncNoHint)ippiSum_16u_C3R :
            type == CV_16UC4 ? (ippiSumFuncNoHint)ippiSum_16u_C4R :
            type == CV_16SC1 ? (ippiSumFuncNoHint)ippiSum_16s_C1R :
            type == CV_16SC3 ? (ippiSumFuncNoHint)ippiSum_16s_C3R :
            type == CV_16SC4 ? (ippiSumFuncNoHint)ippiSum_16s_C4R :
            0;
        CV_Assert(!ippiSumHint || !ippiSum);
        if( ippiSumHint || ippiSum )
        {
            Ipp64f res[4];
            IppStatus ret = ippiSumHint ?
                            CV_INSTRUMENT_FUN_IPP(ippiSumHint, src.ptr(), (int)src.step[0], sz, res, ippAlgHintAccurate) :
                            CV_INSTRUMENT_FUN_IPP(ippiSum, src.ptr(), (int)src.step[0], sz, res);
            if( ret >= 0 )
            {
                for( int i = 0; i < cn; i++ )
                    _res[i] = res[i];
                return true;
            }
        }
    }
#else
    CV_UNUSED(src); CV_UNUSED(_res);
#endif
    return false;
}
#endif

} // cv::

cv::Scalar cv::sum( InputArray _src )
{
    CV_INSTRUMENT_REGION()

#if defined HAVE_OPENCL || defined HAVE_IPP
    Scalar _res;
#endif

#ifdef HAVE_OPENCL
    CV_OCL_RUN_(OCL_PERFORMANCE_CHECK(_src.isUMat()) && _src.dims() <= 2,
                ocl_sum(_src, _res, OCL_OP_SUM),
                _res)
#endif

    Mat src = _src.getMat();
    CV_IPP_RUN(IPP_VERSION_X100 >= 700, ipp_sum(src, _res), _res);

    int k, cn = src.channels(), depth = src.depth();
    SumFunc func = getSumFunc(depth);
    CV_Assert( cn <= 4 && func != 0 );

    const Mat* arrays[] = {&src, 0};
    uchar* ptrs[1];
    NAryMatIterator it(arrays, ptrs);
    Scalar s;
    int total = (int)it.size, blockSize = total, intSumBlockSize = 0;
    int j, count = 0;
    AutoBuffer<int> _buf;
    int* buf = (int*)&s[0];
    size_t esz = 0;
    bool blockSum = depth < CV_32S;

    if( blockSum )
    {
        intSumBlockSize = depth <= CV_8S ? (1 << 23) : (1 << 15);
        blockSize = std::min(blockSize, intSumBlockSize);
        _buf.allocate(cn);
        buf = _buf;

        for( k = 0; k < cn; k++ )
            buf[k] = 0;
        esz = src.elemSize();
    }

    for( size_t i = 0; i < it.nplanes; i++, ++it )
    {
        for( j = 0; j < total; j += blockSize )
        {
            int bsz = std::min(total - j, blockSize);
            func( ptrs[0], 0, (uchar*)buf, bsz, cn );
            count += bsz;
            if( blockSum && (count + blockSize >= intSumBlockSize || (i+1 >= it.nplanes && j+bsz >= total)) )
            {
                for( k = 0; k < cn; k++ )
                {
                    s[k] += buf[k];
                    buf[k] = 0;
                }
                count = 0;
            }
            ptrs[0] += bsz*esz;
        }
    }
    return s;
}
