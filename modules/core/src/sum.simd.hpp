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
    CV_INSTRUMENT_REGION_IPP();

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
    CV_INSTRUMENT_REGION();

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
    uchar* ptrs[1] = {};
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
        buf = _buf.data();

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
