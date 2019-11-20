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
// Copyright (C) 2000-2008,2019 Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2014, Itseez Inc., all rights reserved.
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
#include "opencl_kernels_imgproc.hpp"
#include "opencv2/core/hal/intrin.hpp"
#include "sumpixels.hpp"

namespace cv
{

template <typename T, typename ST, typename QT>
struct Integral_SIMD
{
    bool operator()(const T *, size_t,
                    ST *, size_t,
                    QT *, size_t,
                    ST *, size_t,
                    int, int, int) const
    {
        return false;
    }
};


template <>
struct Integral_SIMD<uchar, double, double> {
    Integral_SIMD() {};


    bool operator()(const uchar *src, size_t _srcstep,
                    double *sum,      size_t _sumstep,
                    double *sqsum,    size_t _sqsumstep,
                    double *tilted,   size_t _tiltedstep,
                    int width, int height, int cn) const
    {
#if CV_TRY_AVX512_SKX
        CV_UNUSED(_tiltedstep);
        // TODO:  Add support for 1 channel input (WIP)
        if (CV_CPU_HAS_SUPPORT_AVX512_SKX && !tilted && (cn <= 4)){
            opt_AVX512_SKX::calculate_integral_avx512(src, _srcstep, sum, _sumstep,
                                                      sqsum, _sqsumstep, width, height, cn);
            return true;
        }
#else
        // Avoid warnings in some builds
        CV_UNUSED(src); CV_UNUSED(_srcstep); CV_UNUSED(sum); CV_UNUSED(_sumstep);
        CV_UNUSED(sqsum); CV_UNUSED(_sqsumstep); CV_UNUSED(tilted); CV_UNUSED(_tiltedstep);
        CV_UNUSED(width); CV_UNUSED(height); CV_UNUSED(cn);
#endif
        return false;
    }

};

#if CV_SIMD && CV_SIMD_WIDTH <= 64

template <>
struct Integral_SIMD<uchar, int, double>
{
    Integral_SIMD() {}

    bool operator()(const uchar * src, size_t _srcstep,
                    int * sum, size_t _sumstep,
                    double * sqsum, size_t,
                    int * tilted, size_t,
                    int width, int height, int cn) const
    {
        if (sqsum || tilted || cn != 1)
            return false;

        // the first iteration
        memset(sum, 0, (width + 1) * sizeof(int));

        // the others
        for (int i = 0; i < height; ++i)
        {
            const uchar * src_row = src + _srcstep * i;
            int * prev_sum_row = (int *)((uchar *)sum + _sumstep * i) + 1;
            int * sum_row = (int *)((uchar *)sum + _sumstep * (i + 1)) + 1;

            sum_row[-1] = 0;

            v_int32 prev = vx_setzero_s32();
            int j = 0;
            for ( ; j + v_uint16::nlanes <= width; j += v_uint16::nlanes)
            {
                v_int16 el8 = v_reinterpret_as_s16(vx_load_expand(src_row + j));
                v_int32 el4l, el4h;
#if CV_AVX2 && CV_SIMD_WIDTH == 32
                __m256i vsum = _mm256_add_epi16(el8.val, _mm256_slli_si256(el8.val, 2));
                vsum = _mm256_add_epi16(vsum, _mm256_slli_si256(vsum, 4));
                vsum = _mm256_add_epi16(vsum, _mm256_slli_si256(vsum, 8));
                __m256i shmask = _mm256_set1_epi32(7);
                el4l.val = _mm256_add_epi32(_mm256_cvtepi16_epi32(_v256_extract_low(vsum)), prev.val);
                el4h.val = _mm256_add_epi32(_mm256_cvtepi16_epi32(_v256_extract_high(vsum)), _mm256_permutevar8x32_epi32(el4l.val, shmask));
                prev.val = _mm256_permutevar8x32_epi32(el4h.val, shmask);
#else
                el8 += v_rotate_left<1>(el8);
                el8 += v_rotate_left<2>(el8);
#if CV_SIMD_WIDTH >= 32
                el8 += v_rotate_left<4>(el8);
#if CV_SIMD_WIDTH == 64
                el8 += v_rotate_left<8>(el8);
#endif
#endif
                v_expand(el8, el4l, el4h);
                el4l += prev;
                el4h += el4l;

                prev = v_broadcast_element<v_int32::nlanes - 1>(el4h);
#endif
                v_store(sum_row + j                  , el4l + vx_load(prev_sum_row + j                  ));
                v_store(sum_row + j + v_int32::nlanes, el4h + vx_load(prev_sum_row + j + v_int32::nlanes));
            }

            for (int v = sum_row[j - 1] - prev_sum_row[j - 1]; j < width; ++j)
                sum_row[j] = (v += src_row[j]) + prev_sum_row[j];
        }
        vx_cleanup();

        return true;
    }
};

template <>
struct Integral_SIMD<uchar, float, double>
{
    Integral_SIMD() {}

    bool operator()(const uchar * src, size_t _srcstep,
        float * sum, size_t _sumstep,
        double * sqsum, size_t,
        float * tilted, size_t,
        int width, int height, int cn) const
    {
        if (sqsum || tilted || cn != 1)
            return false;

        // the first iteration
        memset(sum, 0, (width + 1) * sizeof(int));

        // the others
        for (int i = 0; i < height; ++i)
        {
            const uchar * src_row = src + _srcstep * i;
            float * prev_sum_row = (float *)((uchar *)sum + _sumstep * i) + 1;
            float * sum_row = (float *)((uchar *)sum + _sumstep * (i + 1)) + 1;

            sum_row[-1] = 0;

            v_float32 prev = vx_setzero_f32();
            int j = 0;
            for (; j + v_uint16::nlanes <= width; j += v_uint16::nlanes)
            {
                v_int16 el8 = v_reinterpret_as_s16(vx_load_expand(src_row + j));
                v_float32 el4l, el4h;
#if CV_AVX2 && CV_SIMD_WIDTH == 32
                __m256i vsum = _mm256_add_epi16(el8.val, _mm256_slli_si256(el8.val, 2));
                vsum = _mm256_add_epi16(vsum, _mm256_slli_si256(vsum, 4));
                vsum = _mm256_add_epi16(vsum, _mm256_slli_si256(vsum, 8));
                __m256i shmask = _mm256_set1_epi32(7);
                el4l.val = _mm256_add_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_v256_extract_low(vsum))), prev.val);
                el4h.val = _mm256_add_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_v256_extract_high(vsum))), _mm256_permutevar8x32_ps(el4l.val, shmask));
                prev.val = _mm256_permutevar8x32_ps(el4h.val, shmask);
#else
                el8 += v_rotate_left<1>(el8);
                el8 += v_rotate_left<2>(el8);
#if CV_SIMD_WIDTH >= 32
                el8 += v_rotate_left<4>(el8);
#if CV_SIMD_WIDTH == 64
                el8 += v_rotate_left<8>(el8);
#endif
#endif
                v_int32 el4li, el4hi;
                v_expand(el8, el4li, el4hi);
                el4l = v_cvt_f32(el4li) + prev;
                el4h = v_cvt_f32(el4hi) + el4l;

                prev = v_broadcast_element<v_float32::nlanes - 1>(el4h);
#endif
                v_store(sum_row + j                    , el4l + vx_load(prev_sum_row + j                    ));
                v_store(sum_row + j + v_float32::nlanes, el4h + vx_load(prev_sum_row + j + v_float32::nlanes));
            }

            for (float v = sum_row[j - 1] - prev_sum_row[j - 1]; j < width; ++j)
                sum_row[j] = (v += src_row[j]) + prev_sum_row[j];
        }
        vx_cleanup();

        return true;
    }
};

#endif

template<typename T, typename ST, typename QT>
void integral_( const T* src, size_t _srcstep, ST* sum, size_t _sumstep,
                QT* sqsum, size_t _sqsumstep, ST* tilted, size_t _tiltedstep,
                int width, int height, int cn )
{
    int x, y, k;

    if (Integral_SIMD<T, ST, QT>()(src, _srcstep,
                                   sum, _sumstep,
                                   sqsum, _sqsumstep,
                                   tilted, _tiltedstep,
                                   width, height, cn))
        return;

    int srcstep = (int)(_srcstep/sizeof(T));
    int sumstep = (int)(_sumstep/sizeof(ST));
    int tiltedstep = (int)(_tiltedstep/sizeof(ST));
    int sqsumstep = (int)(_sqsumstep/sizeof(QT));

    width *= cn;

    memset( sum, 0, (width+cn)*sizeof(sum[0]));
    sum += sumstep + cn;

    if( sqsum )
    {
        memset( sqsum, 0, (width+cn)*sizeof(sqsum[0]));
        sqsum += sqsumstep + cn;
    }

    if( tilted )
    {
        memset( tilted, 0, (width+cn)*sizeof(tilted[0]));
        tilted += tiltedstep + cn;
    }

    if( sqsum == 0 && tilted == 0 )
    {
        for( y = 0; y < height; y++, src += srcstep - cn, sum += sumstep - cn )
        {
            for( k = 0; k < cn; k++, src++, sum++ )
            {
                ST s = sum[-cn] = 0;
                for( x = 0; x < width; x += cn )
                {
                    s += src[x];
                    sum[x] = sum[x - sumstep] + s;
                }
            }
        }
    }
    else if( tilted == 0 )
    {
        for( y = 0; y < height; y++, src += srcstep - cn,
                        sum += sumstep - cn, sqsum += sqsumstep - cn )
        {
            for( k = 0; k < cn; k++, src++, sum++, sqsum++ )
            {
                ST s = sum[-cn] = 0;
                QT sq = sqsum[-cn] = 0;
                for( x = 0; x < width; x += cn )
                {
                    T it = src[x];
                    s += it;
                    sq += (QT)it*it;
                    ST t = sum[x - sumstep] + s;
                    QT tq = sqsum[x - sqsumstep] + sq;
                    sum[x] = t;
                    sqsum[x] = tq;
                }
            }
        }
    }
    else
    {
        AutoBuffer<ST> _buf(width+cn);
        ST* buf = _buf.data();
        ST s;
        QT sq;
        for( k = 0; k < cn; k++, src++, sum++, tilted++, buf++ )
        {
            sum[-cn] = tilted[-cn] = 0;

            for( x = 0, s = 0, sq = 0; x < width; x += cn )
            {
                T it = src[x];
                buf[x] = tilted[x] = it;
                s += it;
                sq += (QT)it*it;
                sum[x] = s;
                if( sqsum )
                    sqsum[x] = sq;
            }

            if( width == cn )
                buf[cn] = 0;

            if( sqsum )
            {
                sqsum[-cn] = 0;
                sqsum++;
            }
        }

        for( y = 1; y < height; y++ )
        {
            src += srcstep - cn;
            sum += sumstep - cn;
            tilted += tiltedstep - cn;
            buf += -cn;

            if( sqsum )
                sqsum += sqsumstep - cn;

            for( k = 0; k < cn; k++, src++, sum++, tilted++, buf++ )
            {
                T it = src[0];
                ST t0 = s = it;
                QT tq0 = sq = (QT)it*it;

                sum[-cn] = 0;
                if( sqsum )
                    sqsum[-cn] = 0;
                tilted[-cn] = tilted[-tiltedstep];

                sum[0] = sum[-sumstep] + t0;
                if( sqsum )
                    sqsum[0] = sqsum[-sqsumstep] + tq0;
                tilted[0] = tilted[-tiltedstep] + t0 + buf[cn];

                for( x = cn; x < width - cn; x += cn )
                {
                    ST t1 = buf[x];
                    buf[x - cn] = t1 + t0;
                    t0 = it = src[x];
                    tq0 = (QT)it*it;
                    s += t0;
                    sq += tq0;
                    sum[x] = sum[x - sumstep] + s;
                    if( sqsum )
                        sqsum[x] = sqsum[x - sqsumstep] + sq;
                    t1 += buf[x + cn] + t0 + tilted[x - tiltedstep - cn];
                    tilted[x] = t1;
                }

                if( width > cn )
                {
                    ST t1 = buf[x];
                    buf[x - cn] = t1 + t0;
                    t0 = it = src[x];
                    tq0 = (QT)it*it;
                    s += t0;
                    sq += tq0;
                    sum[x] = sum[x - sumstep] + s;
                    if( sqsum )
                        sqsum[x] = sqsum[x - sqsumstep] + sq;
                    tilted[x] = t0 + t1 + tilted[x - tiltedstep - cn];
                    buf[x] = t0;
                }

                if( sqsum )
                    sqsum++;
            }
        }
    }
}


#ifdef HAVE_OPENCL

static bool ocl_integral( InputArray _src, OutputArray _sum, int sdepth )
{
    bool doubleSupport = ocl::Device::getDefault().doubleFPConfig() > 0;

    if ( (_src.type() != CV_8UC1) ||
        !(sdepth == CV_32S || sdepth == CV_32F || (doubleSupport && sdepth == CV_64F)))
        return false;

    static const int tileSize = 16;

    String build_opt = format("-D sumT=%s -D LOCAL_SUM_SIZE=%d%s",
                                ocl::typeToStr(sdepth), tileSize,
                                doubleSupport ? " -D DOUBLE_SUPPORT" : "");

    ocl::Kernel kcols("integral_sum_cols", ocl::imgproc::integral_sum_oclsrc, build_opt);
    if (kcols.empty())
        return false;

    UMat src = _src.getUMat();
    Size src_size = src.size();
    Size bufsize(((src_size.height + tileSize - 1) / tileSize) * tileSize, ((src_size.width + tileSize - 1) / tileSize) * tileSize);
    UMat buf(bufsize, sdepth);
    kcols.args(ocl::KernelArg::ReadOnly(src), ocl::KernelArg::WriteOnlyNoSize(buf));
    size_t gt = src.cols, lt = tileSize;
    if (!kcols.run(1, &gt, &lt, false))
        return false;

    ocl::Kernel krows("integral_sum_rows", ocl::imgproc::integral_sum_oclsrc, build_opt);
    if (krows.empty())
        return false;

    Size sumsize(src_size.width + 1, src_size.height + 1);
    _sum.create(sumsize, sdepth);
    UMat sum = _sum.getUMat();

    krows.args(ocl::KernelArg::ReadOnlyNoSize(buf), ocl::KernelArg::WriteOnly(sum));
    gt = src.rows;
    return krows.run(1, &gt, &lt, false);
}

static bool ocl_integral( InputArray _src, OutputArray _sum, OutputArray _sqsum, int sdepth, int sqdepth )
{
    bool doubleSupport = ocl::Device::getDefault().doubleFPConfig() > 0;

    if ( _src.type() != CV_8UC1 || (!doubleSupport && (sdepth == CV_64F || sqdepth == CV_64F)) )
        return false;

    static const int tileSize = 16;

    String build_opt = format("-D SUM_SQUARE -D sumT=%s -D sumSQT=%s -D LOCAL_SUM_SIZE=%d%s",
                                ocl::typeToStr(sdepth), ocl::typeToStr(sqdepth),
                                tileSize,
                                doubleSupport ? " -D DOUBLE_SUPPORT" : "");

    ocl::Kernel kcols("integral_sum_cols", ocl::imgproc::integral_sum_oclsrc, build_opt);
    if (kcols.empty())
        return false;

    UMat src = _src.getUMat();
    Size src_size = src.size();
    Size bufsize(((src_size.height + tileSize - 1) / tileSize) * tileSize, ((src_size.width + tileSize - 1) / tileSize) * tileSize);
    UMat buf(bufsize, sdepth);
    UMat buf_sq(bufsize, sqdepth);
    kcols.args(ocl::KernelArg::ReadOnly(src), ocl::KernelArg::WriteOnlyNoSize(buf), ocl::KernelArg::WriteOnlyNoSize(buf_sq));
    size_t gt = src.cols, lt = tileSize;
    if (!kcols.run(1, &gt, &lt, false))
        return false;

    ocl::Kernel krows("integral_sum_rows", ocl::imgproc::integral_sum_oclsrc, build_opt);
    if (krows.empty())
        return false;

    Size sumsize(src_size.width + 1, src_size.height + 1);
    _sum.create(sumsize, sdepth);
    UMat sum = _sum.getUMat();
    _sqsum.create(sumsize, sqdepth);
    UMat sum_sq = _sqsum.getUMat();

    krows.args(ocl::KernelArg::ReadOnlyNoSize(buf), ocl::KernelArg::ReadOnlyNoSize(buf_sq), ocl::KernelArg::WriteOnly(sum), ocl::KernelArg::WriteOnlyNoSize(sum_sq));
    gt = src.rows;
    return krows.run(1, &gt, &lt, false);
}

#endif

}

#if defined(HAVE_IPP)
namespace cv
{
static bool ipp_integral(
    int depth, int sdepth, int sqdepth,
    const uchar* src, size_t srcstep,
    uchar* sum, size_t sumstep,
    uchar* sqsum, size_t sqsumstep,
    uchar* tilted, size_t tstep,
    int width, int height, int cn)
{
    CV_INSTRUMENT_REGION_IPP();

    IppiSize size = {width, height};

    if(cn > 1)
        return false;
    if(tilted)
    {
        CV_UNUSED(tstep);
        return false;
    }

    if(!sqsum)
    {
        if(depth == CV_8U && sdepth == CV_32S)
            return CV_INSTRUMENT_FUN_IPP(ippiIntegral_8u32s_C1R, (const Ipp8u*)src, (int)srcstep, (Ipp32s*)sum, (int)sumstep, size, 0) >= 0;
        else if(depth == CV_8UC1 && sdepth == CV_32F)
            return CV_INSTRUMENT_FUN_IPP(ippiIntegral_8u32f_C1R, (const Ipp8u*)src, (int)srcstep, (Ipp32f*)sum, (int)sumstep, size, 0) >= 0;
        else if(depth == CV_32FC1 && sdepth == CV_32F)
            return CV_INSTRUMENT_FUN_IPP(ippiIntegral_32f_C1R, (const Ipp32f*)src, (int)srcstep, (Ipp32f*)sum, (int)sumstep, size) >= 0;
        else
            return false;
    }
    else
    {
        if(depth == CV_8U && sdepth == CV_32S && sqdepth == CV_32S)
            return CV_INSTRUMENT_FUN_IPP(ippiSqrIntegral_8u32s_C1R, (const Ipp8u*)src, (int)srcstep, (Ipp32s*)sum, (int)sumstep, (Ipp32s*)sqsum, (int)sqsumstep, size, 0, 0) >= 0;
        else if(depth == CV_8U && sdepth == CV_32S && sqdepth == CV_64F)
            return CV_INSTRUMENT_FUN_IPP(ippiSqrIntegral_8u32s64f_C1R, (const Ipp8u*)src, (int)srcstep, (Ipp32s*)sum, (int)sumstep, (Ipp64f*)sqsum, (int)sqsumstep, size, 0, 0) >= 0;
        else if(depth == CV_8U && sdepth == CV_32F && sqdepth == CV_64F)
            return CV_INSTRUMENT_FUN_IPP(ippiSqrIntegral_8u32f64f_C1R, (const Ipp8u*)src, (int)srcstep, (Ipp32f*)sum, (int)sumstep, (Ipp64f*)sqsum, (int)sqsumstep, size, 0, 0) >= 0;
        else
            return false;
    }
}
}
#endif

namespace cv { namespace hal {

void integral(int depth, int sdepth, int sqdepth,
              const uchar* src, size_t srcstep,
              uchar* sum, size_t sumstep,
              uchar* sqsum, size_t sqsumstep,
              uchar* tilted, size_t tstep,
              int width, int height, int cn)
{
    CALL_HAL(integral, cv_hal_integral, depth, sdepth, sqdepth, src, srcstep, sum, sumstep, sqsum, sqsumstep, tilted, tstep, width, height, cn);
    CV_IPP_RUN_FAST(ipp_integral(depth, sdepth, sqdepth, src, srcstep, sum, sumstep, sqsum, sqsumstep, tilted, tstep, width, height, cn));

#define ONE_CALL(A, B, C) integral_<A, B, C>((const A*)src, srcstep, (B*)sum, sumstep, (C*)sqsum, sqsumstep, (B*)tilted, tstep, width, height, cn)

    if( depth == CV_8U && sdepth == CV_32S && sqdepth == CV_64F )
        ONE_CALL(uchar, int, double);
    else if( depth == CV_8U && sdepth == CV_32S && sqdepth == CV_32F )
        ONE_CALL(uchar, int, float);
    else if( depth == CV_8U && sdepth == CV_32S && sqdepth == CV_32S )
        ONE_CALL(uchar, int, int);
    else if( depth == CV_8U && sdepth == CV_32F && sqdepth == CV_64F )
        ONE_CALL(uchar, float, double);
    else if( depth == CV_8U && sdepth == CV_32F && sqdepth == CV_32F )
        ONE_CALL(uchar, float, float);
    else if( depth == CV_8U && sdepth == CV_64F && sqdepth == CV_64F )
        ONE_CALL(uchar, double, double);
    else if( depth == CV_16U && sdepth == CV_64F && sqdepth == CV_64F )
        ONE_CALL(ushort, double, double);
    else if( depth == CV_16S && sdepth == CV_64F && sqdepth == CV_64F )
        ONE_CALL(short, double, double);
    else if( depth == CV_32F && sdepth == CV_32F && sqdepth == CV_64F )
        ONE_CALL(float, float, double);
    else if( depth == CV_32F && sdepth == CV_32F && sqdepth == CV_32F )
        ONE_CALL(float, float, float);
    else if( depth == CV_32F && sdepth == CV_64F && sqdepth == CV_64F )
        ONE_CALL(float, double, double);
    else if( depth == CV_64F && sdepth == CV_64F && sqdepth == CV_64F )
        ONE_CALL(double, double, double);
    else
        CV_Error( CV_StsUnsupportedFormat, "" );

#undef ONE_CALL
}

}} // cv::hal::

void cv::integral( InputArray _src, OutputArray _sum, OutputArray _sqsum, OutputArray _tilted, int sdepth, int sqdepth )
{
    CV_INSTRUMENT_REGION();

    int type = _src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    if( sdepth <= 0 )
        sdepth = depth == CV_8U ? CV_32S : CV_64F;
    if ( sqdepth <= 0 )
         sqdepth = CV_64F;
    sdepth = CV_MAT_DEPTH(sdepth), sqdepth = CV_MAT_DEPTH(sqdepth);

    CV_OCL_RUN(_sum.isUMat() && !_tilted.needed(),
        (_sqsum.needed() ? ocl_integral(_src, _sum, _sqsum, sdepth, sqdepth) : ocl_integral(_src, _sum, sdepth)));

    Size ssize = _src.size(), isize(ssize.width + 1, ssize.height + 1);
    _sum.create( isize, CV_MAKETYPE(sdepth, cn) );
    Mat src = _src.getMat(), sum =_sum.getMat(), sqsum, tilted;

    if( _sqsum.needed() )
    {
        _sqsum.create( isize, CV_MAKETYPE(sqdepth, cn) );
        sqsum = _sqsum.getMat();
    };

    if( _tilted.needed() )
    {
        _tilted.create( isize, CV_MAKETYPE(sdepth, cn) );
        tilted = _tilted.getMat();
    }

    hal::integral(depth, sdepth, sqdepth,
                  src.ptr(), src.step,
                  sum.ptr(), sum.step,
                  sqsum.ptr(), sqsum.step,
                  tilted.ptr(), tilted.step,
                  src.cols, src.rows, cn);
}

void cv::integral( InputArray src, OutputArray sum, int sdepth )
{
    CV_INSTRUMENT_REGION();

    integral( src, sum, noArray(), noArray(), sdepth );
}

void cv::integral( InputArray src, OutputArray sum, OutputArray sqsum, int sdepth, int sqdepth )
{
    CV_INSTRUMENT_REGION();

    integral( src, sum, sqsum, noArray(), sdepth, sqdepth );
}


CV_IMPL void
cvIntegral( const CvArr* image, CvArr* sumImage,
            CvArr* sumSqImage, CvArr* tiltedSumImage )
{
    cv::Mat src = cv::cvarrToMat(image), sum = cv::cvarrToMat(sumImage), sum0 = sum;
    cv::Mat sqsum0, sqsum, tilted0, tilted;
    cv::Mat *psqsum = 0, *ptilted = 0;

    if( sumSqImage )
    {
        sqsum0 = sqsum = cv::cvarrToMat(sumSqImage);
        psqsum = &sqsum;
    }

    if( tiltedSumImage )
    {
        tilted0 = tilted = cv::cvarrToMat(tiltedSumImage);
        ptilted = &tilted;
    }
    cv::integral( src, sum, psqsum ? cv::_OutputArray(*psqsum) : cv::_OutputArray(),
                  ptilted ? cv::_OutputArray(*ptilted) : cv::_OutputArray(), sum.depth() );

    CV_Assert( sum.data == sum0.data && sqsum.data == sqsum0.data && tilted.data == tilted0.data );
}

/* End of file. */
