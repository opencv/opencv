// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


#include "precomp.hpp"
#include "opencl_kernels_core.hpp"
#include "stat.hpp"

namespace cv {

template<typename T>
static int countNonZero_(const T* src, int len )
{
    int i=0, nz = 0;
    #if CV_ENABLE_UNROLLED
    for(; i <= len - 4; i += 4 )
        nz += (src[i] != 0) + (src[i+1] != 0) + (src[i+2] != 0) + (src[i+3] != 0);
    #endif
    for( ; i < len; i++ )
        nz += src[i] != 0;
    return nz;
}

static int countNonZero8u( const uchar* src, int len )
{
    int i=0, nz = 0;
#if CV_SSE2
    if(USE_SSE2)//5x-6x
    {
        __m128i v_zero = _mm_setzero_si128();
        __m128i sum = _mm_setzero_si128();

        for (; i<=len-16; i+=16)
        {
            __m128i r0 = _mm_loadu_si128((const __m128i*)(src+i));
            sum = _mm_add_epi32(sum, _mm_sad_epu8(_mm_sub_epi8(v_zero, _mm_cmpeq_epi8(r0, v_zero)), v_zero));
        }
        nz = i - _mm_cvtsi128_si32(_mm_add_epi32(sum, _mm_unpackhi_epi64(sum, sum)));
    }
#elif CV_NEON
    int len0 = len & -16, blockSize1 = (1 << 8) - 16, blockSize0 = blockSize1 << 6;
    uint32x4_t v_nz = vdupq_n_u32(0u);
    uint8x16_t v_zero = vdupq_n_u8(0), v_1 = vdupq_n_u8(1);
    const uchar * src0 = src;

    while( i < len0 )
    {
        int blockSizei = std::min(len0 - i, blockSize0), j = 0;

        while (j < blockSizei)
        {
            int blockSizej = std::min(blockSizei - j, blockSize1), k = 0;
            uint8x16_t v_pz = v_zero;

            for( ; k <= blockSizej - 16; k += 16 )
                v_pz = vaddq_u8(v_pz, vandq_u8(vceqq_u8(vld1q_u8(src0 + k), v_zero), v_1));

            uint16x8_t v_p1 = vmovl_u8(vget_low_u8(v_pz)), v_p2 = vmovl_u8(vget_high_u8(v_pz));
            v_nz = vaddq_u32(vaddl_u16(vget_low_u16(v_p1), vget_high_u16(v_p1)), v_nz);
            v_nz = vaddq_u32(vaddl_u16(vget_low_u16(v_p2), vget_high_u16(v_p2)), v_nz);

            src0 += blockSizej;
            j += blockSizej;
        }

        i += blockSizei;
    }

    CV_DECL_ALIGNED(16) unsigned int buf[4];
    vst1q_u32(buf, v_nz);
    nz += i - saturate_cast<int>(buf[0] + buf[1] + buf[2] + buf[3]);
#endif
    for( ; i < len; i++ )
        nz += src[i] != 0;
    return nz;
}

static int countNonZero16u( const ushort* src, int len )
{
    int i = 0, nz = 0;
#if CV_SSE2
    if (USE_SSE2)
    {
        __m128i v_zero = _mm_setzero_si128 ();
        __m128i sum = _mm_setzero_si128();

        for ( ; i <= len - 8; i += 8)
        {
            __m128i r0 = _mm_loadu_si128((const __m128i*)(src + i));
            sum = _mm_add_epi32(sum, _mm_sad_epu8(_mm_sub_epi8(v_zero, _mm_cmpeq_epi16(r0, v_zero)), v_zero));
        }

        nz = i - (_mm_cvtsi128_si32(_mm_add_epi32(sum, _mm_unpackhi_epi64(sum, sum))) >> 1);
        src += i;
    }
#elif CV_NEON
    int len0 = len & -8, blockSize1 = (1 << 15), blockSize0 = blockSize1 << 6;
    uint32x4_t v_nz = vdupq_n_u32(0u);
    uint16x8_t v_zero = vdupq_n_u16(0), v_1 = vdupq_n_u16(1);

    while( i < len0 )
    {
        int blockSizei = std::min(len0 - i, blockSize0), j = 0;

        while (j < blockSizei)
        {
            int blockSizej = std::min(blockSizei - j, blockSize1), k = 0;
            uint16x8_t v_pz = v_zero;

            for( ; k <= blockSizej - 8; k += 8 )
                v_pz = vaddq_u16(v_pz, vandq_u16(vceqq_u16(vld1q_u16(src + k), v_zero), v_1));

            v_nz = vaddq_u32(vaddl_u16(vget_low_u16(v_pz), vget_high_u16(v_pz)), v_nz);

            src += blockSizej;
            j += blockSizej;
        }

        i += blockSizei;
    }

    CV_DECL_ALIGNED(16) unsigned int buf[4];
    vst1q_u32(buf, v_nz);
    nz += i - saturate_cast<int>(buf[0] + buf[1] + buf[2] + buf[3]);
#endif
    return nz + countNonZero_(src, len - i);
}

static int countNonZero32s( const int* src, int len )
{
    int i = 0, nz = 0;
#if CV_SSE2
    if (USE_SSE2)
    {
        __m128i v_zero = _mm_setzero_si128 ();
        __m128i sum = _mm_setzero_si128();

        for ( ; i <= len - 4; i += 4)
        {
            __m128i r0 = _mm_loadu_si128((const __m128i*)(src + i));
            sum = _mm_add_epi32(sum, _mm_sad_epu8(_mm_sub_epi8(v_zero, _mm_cmpeq_epi32(r0, v_zero)), v_zero));
        }

        nz = i - (_mm_cvtsi128_si32(_mm_add_epi32(sum, _mm_unpackhi_epi64(sum, sum))) >> 2);
        src += i;
    }
#elif CV_NEON
    int len0 = len & -8, blockSize1 = (1 << 15), blockSize0 = blockSize1 << 6;
    uint32x4_t v_nz = vdupq_n_u32(0u);
    int32x4_t v_zero = vdupq_n_s32(0.0f);
    uint16x8_t v_1 = vdupq_n_u16(1u), v_zerou = vdupq_n_u16(0u);

    while( i < len0 )
    {
        int blockSizei = std::min(len0 - i, blockSize0), j = 0;

        while (j < blockSizei)
        {
            int blockSizej = std::min(blockSizei - j, blockSize1), k = 0;
            uint16x8_t v_pz = v_zerou;

            for( ; k <= blockSizej - 8; k += 8 )
                v_pz = vaddq_u16(v_pz, vandq_u16(vcombine_u16(vmovn_u32(vceqq_s32(vld1q_s32(src + k), v_zero)),
                                                              vmovn_u32(vceqq_s32(vld1q_s32(src + k + 4), v_zero))), v_1));

            v_nz = vaddq_u32(vaddl_u16(vget_low_u16(v_pz), vget_high_u16(v_pz)), v_nz);

            src += blockSizej;
            j += blockSizej;
        }

        i += blockSizei;
    }

    CV_DECL_ALIGNED(16) unsigned int buf[4];
    vst1q_u32(buf, v_nz);
    nz += i - saturate_cast<int>(buf[0] + buf[1] + buf[2] + buf[3]);
#endif
    return nz + countNonZero_(src, len - i);
}

static int countNonZero32f( const float* src, int len )
{
    int i = 0, nz = 0;
#if CV_SSE2
    if (USE_SSE2)
    {
        __m128 v_zero_f = _mm_setzero_ps();
        __m128i v_zero = _mm_setzero_si128 ();
        __m128i sum = _mm_setzero_si128();

        for ( ; i <= len - 4; i += 4)
        {
            __m128 r0 = _mm_loadu_ps(src + i);
            sum = _mm_add_epi32(sum, _mm_sad_epu8(_mm_sub_epi8(v_zero, _mm_castps_si128(_mm_cmpeq_ps(r0, v_zero_f))), v_zero));
        }

        nz = i - (_mm_cvtsi128_si32(_mm_add_epi32(sum, _mm_unpackhi_epi64(sum, sum))) >> 2);
        src += i;
    }
#elif CV_NEON
    int len0 = len & -8, blockSize1 = (1 << 15), blockSize0 = blockSize1 << 6;
    uint32x4_t v_nz = vdupq_n_u32(0u);
    float32x4_t v_zero = vdupq_n_f32(0.0f);
    uint16x8_t v_1 = vdupq_n_u16(1u), v_zerou = vdupq_n_u16(0u);

    while( i < len0 )
    {
        int blockSizei = std::min(len0 - i, blockSize0), j = 0;

        while (j < blockSizei)
        {
            int blockSizej = std::min(blockSizei - j, blockSize1), k = 0;
            uint16x8_t v_pz = v_zerou;

            for( ; k <= blockSizej - 8; k += 8 )
                v_pz = vaddq_u16(v_pz, vandq_u16(vcombine_u16(vmovn_u32(vceqq_f32(vld1q_f32(src + k), v_zero)),
                                                              vmovn_u32(vceqq_f32(vld1q_f32(src + k + 4), v_zero))), v_1));

            v_nz = vaddq_u32(vaddl_u16(vget_low_u16(v_pz), vget_high_u16(v_pz)), v_nz);

            src += blockSizej;
            j += blockSizej;
        }

        i += blockSizei;
    }

    CV_DECL_ALIGNED(16) unsigned int buf[4];
    vst1q_u32(buf, v_nz);
    nz += i - saturate_cast<int>(buf[0] + buf[1] + buf[2] + buf[3]);
#endif
    return nz + countNonZero_(src, len - i);
}

static int countNonZero64f( const double* src, int len )
{
    return countNonZero_(src, len);
}

typedef int (*CountNonZeroFunc)(const uchar*, int);

static CountNonZeroFunc getCountNonZeroTab(int depth)
{
    static CountNonZeroFunc countNonZeroTab[] =
    {
        (CountNonZeroFunc)GET_OPTIMIZED(countNonZero8u), (CountNonZeroFunc)GET_OPTIMIZED(countNonZero8u),
        (CountNonZeroFunc)GET_OPTIMIZED(countNonZero16u), (CountNonZeroFunc)GET_OPTIMIZED(countNonZero16u),
        (CountNonZeroFunc)GET_OPTIMIZED(countNonZero32s), (CountNonZeroFunc)GET_OPTIMIZED(countNonZero32f),
        (CountNonZeroFunc)GET_OPTIMIZED(countNonZero64f), 0
    };

    return countNonZeroTab[depth];
}


#ifdef HAVE_OPENCL
static bool ocl_countNonZero( InputArray _src, int & res )
{
    int type = _src.type(), depth = CV_MAT_DEPTH(type), kercn = ocl::predictOptimalVectorWidth(_src);
    bool doubleSupport = ocl::Device::getDefault().doubleFPConfig() > 0;

    if (depth == CV_64F && !doubleSupport)
        return false;

    int dbsize = ocl::Device::getDefault().maxComputeUnits();
    size_t wgs = ocl::Device::getDefault().maxWorkGroupSize();

    int wgs2_aligned = 1;
    while (wgs2_aligned < (int)wgs)
        wgs2_aligned <<= 1;
    wgs2_aligned >>= 1;

    ocl::Kernel k("reduce", ocl::core::reduce_oclsrc,
                  format("-D srcT=%s -D srcT1=%s -D cn=1 -D OP_COUNT_NON_ZERO"
                         " -D WGS=%d -D kercn=%d -D WGS2_ALIGNED=%d%s%s",
                         ocl::typeToStr(CV_MAKE_TYPE(depth, kercn)),
                         ocl::typeToStr(depth), (int)wgs, kercn,
                         wgs2_aligned, doubleSupport ? " -D DOUBLE_SUPPORT" : "",
                         _src.isContinuous() ? " -D HAVE_SRC_CONT" : ""));
    if (k.empty())
        return false;

    UMat src = _src.getUMat(), db(1, dbsize, CV_32SC1);
    k.args(ocl::KernelArg::ReadOnlyNoSize(src), src.cols, (int)src.total(),
           dbsize, ocl::KernelArg::PtrWriteOnly(db));

    size_t globalsize = dbsize * wgs;
    if (k.run(1, &globalsize, &wgs, true))
        return res = saturate_cast<int>(cv::sum(db.getMat(ACCESS_READ))[0]), true;
    return false;
}
#endif

#if defined HAVE_IPP
static bool ipp_countNonZero( Mat &src, int &res )
{
    CV_INSTRUMENT_REGION_IPP()

#if IPP_VERSION_X100 < 201801
    // Poor performance of SSE42
    if(cv::ipp::getIppTopFeatures() == ippCPUID_SSE42)
        return false;
#endif

    Ipp32s  count = 0;
    int     depth = src.depth();

    if(src.dims <= 2)
    {
        IppStatus status;
        IppiSize  size = {src.cols*src.channels(), src.rows};

        if(depth == CV_8U)
            status = CV_INSTRUMENT_FUN_IPP(ippiCountInRange_8u_C1R, (const Ipp8u *)src.ptr(), (int)src.step, size, &count, 0, 0);
        else if(depth == CV_32F)
            status = CV_INSTRUMENT_FUN_IPP(ippiCountInRange_32f_C1R, (const Ipp32f *)src.ptr(), (int)src.step, size, &count, 0, 0);
        else
            return false;

        if(status < 0)
            return false;

        res = size.width*size.height - count;
    }
    else
    {
        IppStatus       status;
        const Mat      *arrays[] = {&src, NULL};
        Mat            planes[1];
        NAryMatIterator it(arrays, planes, 1);
        IppiSize        size  = {(int)it.size*src.channels(), 1};
        res = 0;
        for (size_t i = 0; i < it.nplanes; i++, ++it)
        {
            if(depth == CV_8U)
                status = CV_INSTRUMENT_FUN_IPP(ippiCountInRange_8u_C1R, it.planes->ptr<Ipp8u>(), (int)it.planes->step, size, &count, 0, 0);
            else if(depth == CV_32F)
                status = CV_INSTRUMENT_FUN_IPP(ippiCountInRange_32f_C1R, it.planes->ptr<Ipp32f>(), (int)it.planes->step, size, &count, 0, 0);
            else
                return false;

            if(status < 0 || (int)it.planes->total()*src.channels() < count)
                return false;

            res += (int)it.planes->total()*src.channels() - count;
        }
    }

    return true;
}
#endif

} // cv::

int cv::countNonZero( InputArray _src )
{
    CV_INSTRUMENT_REGION()

    int type = _src.type(), cn = CV_MAT_CN(type);
    CV_Assert( cn == 1 );

#if defined HAVE_OPENCL || defined HAVE_IPP
    int res = -1;
#endif

#ifdef HAVE_OPENCL
    CV_OCL_RUN_(OCL_PERFORMANCE_CHECK(_src.isUMat()) && _src.dims() <= 2,
                ocl_countNonZero(_src, res),
                res)
#endif

    Mat src = _src.getMat();
    CV_IPP_RUN_FAST(ipp_countNonZero(src, res), res);

    CountNonZeroFunc func = getCountNonZeroTab(src.depth());
    CV_Assert( func != 0 );

    const Mat* arrays[] = {&src, 0};
    uchar* ptrs[1];
    NAryMatIterator it(arrays, ptrs);
    int total = (int)it.size, nz = 0;

    for( size_t i = 0; i < it.nplanes; i++, ++it )
        nz += func( ptrs[0], total );

    return nz;
}

void cv::findNonZero( InputArray _src, OutputArray _idx )
{
    CV_INSTRUMENT_REGION()

    Mat src = _src.getMat();
    CV_Assert( src.type() == CV_8UC1 );
    int n = countNonZero(src);
    if( n == 0 )
    {
        _idx.release();
        return;
    }
    if( _idx.kind() == _InputArray::MAT && !_idx.getMatRef().isContinuous() )
        _idx.release();
    _idx.create(n, 1, CV_32SC2);
    Mat idx = _idx.getMat();
    CV_Assert(idx.isContinuous());
    Point* idx_ptr = idx.ptr<Point>();

    for( int i = 0; i < src.rows; i++ )
    {
        const uchar* bin_ptr = src.ptr(i);
        for( int j = 0; j < src.cols; j++ )
            if( bin_ptr[j] )
                *idx_ptr++ = Point(j, i);
    }
}
