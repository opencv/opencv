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
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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
#include "opencl_kernels.hpp"

#if defined (HAVE_IPP) && (IPP_VERSION_MAJOR >= 7)
static IppStatus sts = ippInit();
#endif

namespace cv
{

template<typename T, typename ST, typename QT>
void integral_( const T* src, size_t _srcstep, ST* sum, size_t _sumstep,
                QT* sqsum, size_t _sqsumstep, ST* tilted, size_t _tiltedstep,
                Size size, int cn )
{
    int x, y, k;

    int srcstep = (int)(_srcstep/sizeof(T));
    int sumstep = (int)(_sumstep/sizeof(ST));
    int tiltedstep = (int)(_tiltedstep/sizeof(ST));
    int sqsumstep = (int)(_sqsumstep/sizeof(QT));

    size.width *= cn;

    memset( sum, 0, (size.width+cn)*sizeof(sum[0]));
    sum += sumstep + cn;

    if( sqsum )
    {
        memset( sqsum, 0, (size.width+cn)*sizeof(sqsum[0]));
        sqsum += sqsumstep + cn;
    }

    if( tilted )
    {
        memset( tilted, 0, (size.width+cn)*sizeof(tilted[0]));
        tilted += tiltedstep + cn;
    }

    if( sqsum == 0 && tilted == 0 )
    {
        for( y = 0; y < size.height; y++, src += srcstep - cn, sum += sumstep - cn )
        {
            for( k = 0; k < cn; k++, src++, sum++ )
            {
                ST s = sum[-cn] = 0;
                for( x = 0; x < size.width; x += cn )
                {
                    s += src[x];
                    sum[x] = sum[x - sumstep] + s;
                }
            }
        }
    }
    else if( tilted == 0 )
    {
        for( y = 0; y < size.height; y++, src += srcstep - cn,
                        sum += sumstep - cn, sqsum += sqsumstep - cn )
        {
            for( k = 0; k < cn; k++, src++, sum++, sqsum++ )
            {
                ST s = sum[-cn] = 0;
                QT sq = sqsum[-cn] = 0;
                for( x = 0; x < size.width; x += cn )
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
        AutoBuffer<ST> _buf(size.width+cn);
        ST* buf = _buf;
        ST s;
        QT sq;
        for( k = 0; k < cn; k++, src++, sum++, tilted++, buf++ )
        {
            sum[-cn] = tilted[-cn] = 0;

            for( x = 0, s = 0, sq = 0; x < size.width; x += cn )
            {
                T it = src[x];
                buf[x] = tilted[x] = it;
                s += it;
                sq += (QT)it*it;
                sum[x] = s;
                if( sqsum )
                    sqsum[x] = sq;
            }

            if( size.width == cn )
                buf[cn] = 0;

            if( sqsum )
            {
                sqsum[-cn] = 0;
                sqsum++;
            }
        }

        for( y = 1; y < size.height; y++ )
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

                for( x = cn; x < size.width - cn; x += cn )
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

                if( size.width > cn )
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


#define DEF_INTEGRAL_FUNC(suffix, T, ST, QT) \
static void integral_##suffix( T* src, size_t srcstep, ST* sum, size_t sumstep, QT* sqsum, size_t sqsumstep, \
                              ST* tilted, size_t tiltedstep, Size size, int cn ) \
{ integral_(src, srcstep, sum, sumstep, sqsum, sqsumstep, tilted, tiltedstep, size, cn); }

DEF_INTEGRAL_FUNC(8u32s, uchar, int, double)
DEF_INTEGRAL_FUNC(8u32f64f, uchar, float, double)
DEF_INTEGRAL_FUNC(8u64f64f, uchar, double, double)
DEF_INTEGRAL_FUNC(32f32f64f, float, float, double)
DEF_INTEGRAL_FUNC(32f64f64f, float, double, double)
DEF_INTEGRAL_FUNC(64f64f64f, double, double, double)

DEF_INTEGRAL_FUNC(8u32s32f, uchar, int, float)
DEF_INTEGRAL_FUNC(8u32f32f, uchar, float, float)
DEF_INTEGRAL_FUNC(32f32f32f, float, float, float)

typedef void (*IntegralFunc)(const uchar* src, size_t srcstep, uchar* sum, size_t sumstep,
                             uchar* sqsum, size_t sqsumstep, uchar* tilted, size_t tstep,
                             Size size, int cn );

#ifdef HAVE_OPENCL

enum { vlen = 4 };

static bool ocl_integral( InputArray _src, OutputArray _sum, int sdepth )
{
    if ( _src.type() != CV_8UC1 || _src.step() % vlen != 0 || _src.offset() % vlen != 0  ||
         !(sdepth == CV_32S || sdepth == CV_32F) )
        return false;

    ocl::Kernel k1("integral_sum_cols", ocl::imgproc::integral_sum_oclsrc,
                   format("-D sdepth=%d", sdepth));
    if (k1.empty())
        return false;

    Size size = _src.size(), t_size = Size(((size.height + vlen - 1) / vlen) * vlen, size.width),
            ssize(size.width + 1, size.height + 1);
    _sum.create(ssize, sdepth);
    UMat src = _src.getUMat(), t_sum(t_size, sdepth), sum = _sum.getUMat();
    t_sum = t_sum(Range::all(), Range(0, size.height));

    int offset = (int)src.offset / vlen, pre_invalid = (int)src.offset % vlen;
    int vcols = (pre_invalid + src.cols + vlen - 1) / vlen;
    int sum_offset = (int)sum.offset / vlen;

    k1.args(ocl::KernelArg::PtrReadOnly(src), ocl::KernelArg::PtrWriteOnly(t_sum),
            offset, pre_invalid, src.rows, src.cols, (int)src.step, (int)t_sum.step);
    size_t gt = ((vcols + 1) / 2) * 256, lt = 256;
    if (!k1.run(1, &gt, &lt, false))
        return false;

    ocl::Kernel k2("integral_sum_rows", ocl::imgproc::integral_sum_oclsrc,
                   format("-D sdepth=%d", sdepth));
    k2.args(ocl::KernelArg::PtrReadWrite(t_sum), ocl::KernelArg::PtrWriteOnly(sum),
            t_sum.rows, t_sum.cols, (int)t_sum.step, (int)sum.step, sum_offset);

    size_t gt2 = t_sum.cols  * 32, lt2 = 256;
    return k2.run(1, &gt2, &lt2, false);
}

static bool ocl_integral( InputArray _src, OutputArray _sum, OutputArray _sqsum, int sdepth, int sqdepth )
{
    bool doubleSupport = ocl::Device::getDefault().doubleFPConfig() > 0;

    if ( _src.type() != CV_8UC1 || _src.step() % vlen != 0 || _src.offset() % vlen != 0 ||
         (!doubleSupport && (sdepth == CV_64F || sqdepth == CV_64F)) )
        return false;

    char cvt[40];
    String opts = format("-D sdepth=%d -D sqdepth=%d -D TYPE=%s -D TYPE4=%s4 -D convert_TYPE4=%s%s",
                         sdepth, sqdepth, ocl::typeToStr(sqdepth), ocl::typeToStr(sqdepth),
                         ocl::convertTypeStr(sdepth, sqdepth, 4, cvt),
                         doubleSupport ? " -D DOUBLE_SUPPORT" : "");

    ocl::Kernel k1("integral_cols", ocl::imgproc::integral_sqrsum_oclsrc, opts);
    if (k1.empty())
        return false;

    Size size = _src.size(), dsize = Size(size.width + 1, size.height + 1),
            t_size = Size(((size.height + vlen - 1) / vlen) * vlen, size.width);
    UMat src = _src.getUMat(), t_sum(t_size, sdepth), t_sqsum(t_size, sqdepth);
    t_sum = t_sum(Range::all(), Range(0, size.height));
    t_sqsum = t_sqsum(Range::all(), Range(0, size.height));

    _sum.create(dsize, sdepth);
    _sqsum.create(dsize, sqdepth);
    UMat sum = _sum.getUMat(), sqsum = _sqsum.getUMat();

    int offset = (int)src.offset / vlen;
    int pre_invalid = src.offset % vlen;
    int vcols = (pre_invalid + src.cols + vlen - 1) / vlen;
    int sum_offset = (int)(sum.offset / sum.elemSize());
    int sqsum_offset = (int)(sqsum.offset / sqsum.elemSize());

    k1.args(ocl::KernelArg::PtrReadOnly(src), ocl::KernelArg::PtrWriteOnly(t_sum),
            ocl::KernelArg::PtrWriteOnly(t_sqsum), offset, pre_invalid, src.rows,
            src.cols, (int)src.step, (int)t_sum.step, (int)t_sqsum.step);

    size_t gt = ((vcols + 1) / 2) * 256, lt = 256;
    if (!k1.run(1, &gt, &lt, false))
        return false;

    ocl::Kernel k2("integral_rows", ocl::imgproc::integral_sqrsum_oclsrc, opts);
    if (k2.empty())
        return false;

    k2.args(ocl::KernelArg::PtrReadOnly(t_sum), ocl::KernelArg::PtrReadOnly(t_sqsum),
            ocl::KernelArg::PtrWriteOnly(sum), ocl::KernelArg::PtrWriteOnly(sqsum),
            t_sum.rows, t_sum.cols, (int)t_sum.step, (int)t_sqsum.step,
            (int)sum.step, (int)sqsum.step, sum_offset, sqsum_offset);

    size_t gt2 = t_sum.cols  * 32, lt2 = 256;
    return k2.run(1, &gt2, &lt2, false);
}

#endif

}


void cv::integral( InputArray _src, OutputArray _sum, OutputArray _sqsum, OutputArray _tilted, int sdepth, int sqdepth )
{
    int type = _src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    if( sdepth <= 0 )
        sdepth = depth == CV_8U ? CV_32S : CV_64F;
    if ( sqdepth <= 0 )
         sqdepth = CV_64F;
    sdepth = CV_MAT_DEPTH(sdepth), sqdepth = CV_MAT_DEPTH(sqdepth);

#ifdef HAVE_OPENCL
    if (ocl::useOpenCL() && _sum.isUMat() && !_tilted.needed())
    {
        if (!_sqsum.needed())
        {
            CV_OCL_RUN(ocl::useOpenCL(), ocl_integral(_src, _sum, sdepth))
        }
        else if (_sqsum.isUMat())
            CV_OCL_RUN(ocl::useOpenCL(), ocl_integral(_src, _sum, _sqsum, sdepth, sqdepth))
    }
#endif

    Size ssize = _src.size(), isize(ssize.width + 1, ssize.height + 1);
    _sum.create( isize, CV_MAKETYPE(sdepth, cn) );
    Mat src = _src.getMat(), sum =_sum.getMat(), sqsum, tilted;

    if( _sqsum.needed() )
    {
        _sqsum.create( isize, CV_MAKETYPE(sqdepth, cn) );
        sqsum = _sqsum.getMat();
    };

#if defined (HAVE_IPP) && (IPP_VERSION_MAJOR >= 7)
    if( ( depth == CV_8U ) && ( sdepth == CV_32F || sdepth == CV_32S ) && ( !_tilted.needed() ) && ( !_sqsum.needed() || sqdepth == CV_64F ) && ( cn == 1 ) )
    {
        IppiSize srcRoiSize = ippiSize( src.cols, src.rows );
        if( sdepth == CV_32F )
        {
            if( _sqsum.needed() )
            {
                ippiSqrIntegral_8u32f64f_C1R( (const Ipp8u*)src.data, (int)src.step, (Ipp32f*)sum.data, (int)sum.step, (Ipp64f*)sqsum.data, (int)sqsum.step, srcRoiSize, 0, 0 );
            }
            else
            {
                ippiIntegral_8u32f_C1R( (const Ipp8u*)src.data, (int)src.step, (Ipp32f*)sum.data, (int)sum.step, srcRoiSize, 0 );
            }
        }
        else if( sdepth == CV_32S )
        {
            if( _sqsum.needed() )
            {
                ippiSqrIntegral_8u32s64f_C1R( (const Ipp8u*)src.data, (int)src.step, (Ipp32s*)sum.data, (int)sum.step, (Ipp64f*)sqsum.data, (int)sqsum.step, srcRoiSize, 0, 0 );
            }
            else
            {
                ippiIntegral_8u32s_C1R( (const Ipp8u*)src.data, (int)src.step, (Ipp32s*)sum.data, (int)sum.step, srcRoiSize, 0 );
            }
        }
        return;
    }
#endif

    if( _tilted.needed() )
    {
        _tilted.create( isize, CV_MAKETYPE(sdepth, cn) );
        tilted = _tilted.getMat();
    }

    IntegralFunc func = 0;
    if( depth == CV_8U && sdepth == CV_32S && sqdepth == CV_64F )
        func = (IntegralFunc)GET_OPTIMIZED(integral_8u32s);
    else if( depth == CV_8U && sdepth == CV_32S && sqdepth == CV_32F )
        func = (IntegralFunc)integral_8u32s32f;
    else if( depth == CV_8U && sdepth == CV_32F && sqdepth == CV_64F )
        func = (IntegralFunc)integral_8u32f64f;
    else if( depth == CV_8U && sdepth == CV_32F && sqdepth == CV_32F )
        func = (IntegralFunc)integral_8u32f32f;
    else if( depth == CV_8U && sdepth == CV_64F && sqdepth == CV_64F )
        func = (IntegralFunc)integral_8u64f64f;
    else if( depth == CV_32F && sdepth == CV_32F && sqdepth == CV_64F )
        func = (IntegralFunc)integral_32f32f64f;
    else if( depth == CV_32F && sdepth == CV_32F && sqdepth == CV_32F )
        func = (IntegralFunc)integral_32f32f32f;
    else if( depth == CV_32F && sdepth == CV_64F && sqdepth == CV_64F )
        func = (IntegralFunc)integral_32f64f64f;
    else if( depth == CV_64F && sdepth == CV_64F && sqdepth == CV_64F )
        func = (IntegralFunc)integral_64f64f64f;
    else
        CV_Error( CV_StsUnsupportedFormat, "" );

    func( src.data, src.step, sum.data, sum.step, sqsum.data, sqsum.step,
          tilted.data, tilted.step, src.size(), cn );
}

void cv::integral( InputArray src, OutputArray sum, int sdepth )
{
    integral( src, sum, noArray(), noArray(), sdepth );
}

void cv::integral( InputArray src, OutputArray sum, OutputArray sqsum, int sdepth, int sqdepth )
{
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
