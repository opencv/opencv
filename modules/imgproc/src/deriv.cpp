/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Copyright (C) 2014, Itseez, Inc, all rights reserved.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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

/****************************************************************************************\
                             Sobel & Scharr Derivative Filters
\****************************************************************************************/

namespace cv
{

static void getScharrKernels( OutputArray _kx, OutputArray _ky,
                              int dx, int dy, bool normalize, int ktype )
{
    const int ksize = 3;

    CV_Assert( ktype == CV_32F || ktype == CV_64F );
    _kx.create(ksize, 1, ktype, -1, true);
    _ky.create(ksize, 1, ktype, -1, true);
    Mat kx = _kx.getMat();
    Mat ky = _ky.getMat();

    CV_Assert( dx >= 0 && dy >= 0 && dx+dy == 1 );

    for( int k = 0; k < 2; k++ )
    {
        Mat* kernel = k == 0 ? &kx : &ky;
        int order = k == 0 ? dx : dy;
        int kerI[3];

        if( order == 0 )
            kerI[0] = 3, kerI[1] = 10, kerI[2] = 3;
        else if( order == 1 )
            kerI[0] = -1, kerI[1] = 0, kerI[2] = 1;

        Mat temp(kernel->rows, kernel->cols, CV_32S, &kerI[0]);
        double scale = !normalize || order == 1 ? 1. : 1./32;
        temp.convertTo(*kernel, ktype, scale);
    }
}


static void getSobelKernels( OutputArray _kx, OutputArray _ky,
                             int dx, int dy, int _ksize, bool normalize, int ktype )
{
    int i, j, ksizeX = _ksize, ksizeY = _ksize;
    if( ksizeX == 1 && dx > 0 )
        ksizeX = 3;
    if( ksizeY == 1 && dy > 0 )
        ksizeY = 3;

    CV_Assert( ktype == CV_32F || ktype == CV_64F );

    _kx.create(ksizeX, 1, ktype, -1, true);
    _ky.create(ksizeY, 1, ktype, -1, true);
    Mat kx = _kx.getMat();
    Mat ky = _ky.getMat();

    if( _ksize % 2 == 0 || _ksize > 31 )
        CV_Error( CV_StsOutOfRange, "The kernel size must be odd and not larger than 31" );
    std::vector<int> kerI(std::max(ksizeX, ksizeY) + 1);

    CV_Assert( dx >= 0 && dy >= 0 && dx+dy > 0 );

    for( int k = 0; k < 2; k++ )
    {
        Mat* kernel = k == 0 ? &kx : &ky;
        int order = k == 0 ? dx : dy;
        int ksize = k == 0 ? ksizeX : ksizeY;

        CV_Assert( ksize > order );

        if( ksize == 1 )
            kerI[0] = 1;
        else if( ksize == 3 )
        {
            if( order == 0 )
                kerI[0] = 1, kerI[1] = 2, kerI[2] = 1;
            else if( order == 1 )
                kerI[0] = -1, kerI[1] = 0, kerI[2] = 1;
            else
                kerI[0] = 1, kerI[1] = -2, kerI[2] = 1;
        }
        else
        {
            int oldval, newval;
            kerI[0] = 1;
            for( i = 0; i < ksize; i++ )
                kerI[i+1] = 0;

            for( i = 0; i < ksize - order - 1; i++ )
            {
                oldval = kerI[0];
                for( j = 1; j <= ksize; j++ )
                {
                    newval = kerI[j]+kerI[j-1];
                    kerI[j-1] = oldval;
                    oldval = newval;
                }
            }

            for( i = 0; i < order; i++ )
            {
                oldval = -kerI[0];
                for( j = 1; j <= ksize; j++ )
                {
                    newval = kerI[j-1] - kerI[j];
                    kerI[j-1] = oldval;
                    oldval = newval;
                }
            }
        }

        Mat temp(kernel->rows, kernel->cols, CV_32S, &kerI[0]);
        double scale = !normalize ? 1. : 1./(1 << (ksize-order-1));
        temp.convertTo(*kernel, ktype, scale);
    }
}

}

void cv::getDerivKernels( OutputArray kx, OutputArray ky, int dx, int dy,
                          int ksize, bool normalize, int ktype )
{
    if( ksize <= 0 )
        getScharrKernels( kx, ky, dx, dy, normalize, ktype );
    else
        getSobelKernels( kx, ky, dx, dy, ksize, normalize, ktype );
}


cv::Ptr<cv::FilterEngine> cv::createDerivFilter(int srcType, int dstType,
                                                int dx, int dy, int ksize, int borderType )
{
    Mat kx, ky;
    getDerivKernels( kx, ky, dx, dy, ksize, false, CV_32F );
    return createSeparableLinearFilter(srcType, dstType,
        kx, ky, Point(-1,-1), 0, borderType );
}

#if defined (HAVE_IPP) && (IPP_VERSION_MAJOR >= 7)

#define IPP_RETURN_ERROR    {setIppErrorStatus(); return false;}

namespace cv
{
#if IPP_VERSION_X100 >= 801
static bool IPPDerivScharr(InputArray _src, OutputArray _dst, int ddepth, int dx, int dy, double scale, double delta, int borderType)
{
    if ((0 > dx) || (0 > dy) || (1 != dx + dy))
        return false;
    if (fabs(delta) > FLT_EPSILON)
        return false;

    IppiBorderType ippiBorderType = ippiGetBorderType(borderType & (~BORDER_ISOLATED));
    if ((int)ippiBorderType < 0)
        return false;

    int stype = _src.type(), sdepth = CV_MAT_DEPTH(stype), cn = CV_MAT_CN(stype);
    if (ddepth < 0)
        ddepth = sdepth;
    int dtype = CV_MAKETYPE(ddepth, cn);

    Mat src = _src.getMat();
    if (0 == (BORDER_ISOLATED & borderType))
    {
        Size size; Point offset;
        src.locateROI(size, offset);
        if (0 < offset.x)
            ippiBorderType = (IppiBorderType)(ippiBorderType | ippBorderInMemLeft);
        if (0 < offset.y)
            ippiBorderType = (IppiBorderType)(ippiBorderType | ippBorderInMemTop);
        if (offset.x + src.cols < size.width)
            ippiBorderType = (IppiBorderType)(ippiBorderType | ippBorderInMemRight);
        if (offset.y + src.rows < size.height)
            ippiBorderType = (IppiBorderType)(ippiBorderType | ippBorderInMemBottom);
    }

    bool horz = (0 == dx) && (1 == dy);
    IppiSize roiSize = {src.cols, src.rows};

    _dst.create( _src.size(), dtype);
    Mat dst = _dst.getMat();
    IppStatus sts = ippStsErr;
    if ((CV_8U == stype) && (CV_16S == dtype))
    {
        int bufferSize = 0; Ipp8u *pBuffer;
        if (horz)
        {
            if (0 > ippiFilterScharrHorizMaskBorderGetBufferSize(roiSize, ippMskSize3x3, ipp8u, ipp16s, 1, &bufferSize))
                IPP_RETURN_ERROR
            pBuffer = ippsMalloc_8u(bufferSize);
            if (NULL == pBuffer)
                IPP_RETURN_ERROR
            sts = ippiFilterScharrHorizMaskBorder_8u16s_C1R(src.data, (int)src.step, (Ipp16s *)dst.data, (int)dst.step, roiSize, ippMskSize3x3, ippiBorderType, 0, pBuffer);
        }
        else
        {
            if (0 > ippiFilterScharrVertMaskBorderGetBufferSize(roiSize, ippMskSize3x3, ipp8u, ipp16s, 1, &bufferSize))
                IPP_RETURN_ERROR
            pBuffer = ippsMalloc_8u(bufferSize);
            if (NULL == pBuffer)
                IPP_RETURN_ERROR
            sts = ippiFilterScharrVertMaskBorder_8u16s_C1R(src.data, (int)src.step, (Ipp16s *)dst.data, (int)dst.step, roiSize, ippMskSize3x3, ippiBorderType, 0, pBuffer);
        }
        ippsFree(pBuffer);
    }
    else if ((CV_16S == stype) && (CV_16S == dtype))
    {
        int bufferSize = 0; Ipp8u *pBuffer;
        if (horz)
        {
            if (0 > ippiFilterScharrHorizMaskBorderGetBufferSize(roiSize, ippMskSize3x3, ipp16s, ipp16s, 1, &bufferSize))
                IPP_RETURN_ERROR
            pBuffer = ippsMalloc_8u(bufferSize);
            if (NULL == pBuffer)
                IPP_RETURN_ERROR
            sts = ippiFilterScharrHorizMaskBorder_16s_C1R((Ipp16s *)src.data, (int)src.step, (Ipp16s *)dst.data, (int)dst.step, roiSize, ippMskSize3x3, ippiBorderType, 0, pBuffer);
        }
        else
        {
            if (0 > ippiFilterScharrVertMaskBorderGetBufferSize(roiSize, ippMskSize3x3, ipp16s, ipp16s, 1, &bufferSize))
                IPP_RETURN_ERROR
            pBuffer = ippsMalloc_8u(bufferSize);
            if (NULL == pBuffer)
                IPP_RETURN_ERROR
            sts = ippiFilterScharrVertMaskBorder_16s_C1R((Ipp16s *)src.data, (int)src.step, (Ipp16s *)dst.data, (int)dst.step, roiSize, ippMskSize3x3, ippiBorderType, 0, pBuffer);
        }
        ippsFree(pBuffer);
    }
    else if ((CV_32F == stype) && (CV_32F == dtype))
    {
        int bufferSize = 0; Ipp8u *pBuffer;
        if (horz)
        {
            if (0 > ippiFilterScharrHorizMaskBorderGetBufferSize(roiSize, ippMskSize3x3, ipp32f, ipp32f, 1, &bufferSize))
                IPP_RETURN_ERROR
            pBuffer = ippsMalloc_8u(bufferSize);
            if (NULL == pBuffer)
                IPP_RETURN_ERROR
            sts = ippiFilterScharrHorizMaskBorder_32f_C1R((Ipp32f *)src.data, (int)src.step, (Ipp32f *)dst.data, (int)dst.step, roiSize, ippMskSize3x3, ippiBorderType, 0, pBuffer);
        }
        else
        {
            if (0 > ippiFilterScharrVertMaskBorderGetBufferSize(roiSize, ippMskSize3x3, ipp32f, ipp32f, 1, &bufferSize))
                IPP_RETURN_ERROR
            pBuffer = ippsMalloc_8u(bufferSize);
            if (NULL == pBuffer)
                IPP_RETURN_ERROR
            sts = ippiFilterScharrVertMaskBorder_32f_C1R((Ipp32f *)src.data, (int)src.step, (Ipp32f *)dst.data, (int)dst.step, roiSize, ippMskSize3x3, ippiBorderType, 0, pBuffer);
        }
        ippsFree(pBuffer);
        if (sts < 0)
            IPP_RETURN_ERROR;

        if (FLT_EPSILON < fabs(scale - 1.0))
            sts = ippiMulC_32f_C1R((Ipp32f *)dst.data, (int)dst.step, (Ipp32f)scale, (Ipp32f *)dst.data, (int)dst.step, roiSize);
    }
    return (0 <= sts);
}
#elif IPP_VERSION_X100 >= 700
static bool IPPDerivScharr(InputArray _src, OutputArray _dst, int ddepth, int dx, int dy, double scale, double delta, int borderType)
{
    if (BORDER_REPLICATE != borderType)
        return false;
    if ((0 > dx) || (0 > dy) || (1 != dx + dy))
        return false;
    if (fabs(delta) > FLT_EPSILON)
        return false;

    Mat src = _src.getMat(), dst = _dst.getMat();

    int bufSize = 0;
    cv::AutoBuffer<char> buffer;
    IppiSize roi = ippiSize(src.cols, src.rows);

    if( ddepth < 0 )
        ddepth = src.depth();

    dst.create( src.size(), CV_MAKETYPE(ddepth, src.channels()) );

    switch(src.type())
    {
    case CV_8UC1:
        {
            if(scale != 1)
                return false;

            switch(dst.type())
            {
            case CV_16S:
                {
                    if ((dx == 1) && (dy == 0))
                    {
                        if (0 > ippiFilterScharrVertGetBufferSize_8u16s_C1R(roi,&bufSize))
                            return false;
                        buffer.allocate(bufSize);
                        return (0 <= ippiFilterScharrVertBorder_8u16s_C1R((const Ipp8u*)src.data, (int)src.step,
                                        (Ipp16s*)dst.data, (int)dst.step, roi, ippBorderRepl, 0, (Ipp8u*)(char*)buffer));
                    }
                    if ((dx == 0) && (dy == 1))
                    {
                        if (0 > ippiFilterScharrHorizGetBufferSize_8u16s_C1R(roi,&bufSize))
                            return false;
                        buffer.allocate(bufSize);
                        return (0 <= ippiFilterScharrHorizBorder_8u16s_C1R((const Ipp8u*)src.data, (int)src.step,
                                            (Ipp16s*)dst.data, (int)dst.step, roi, ippBorderRepl, 0, (Ipp8u*)(char*)buffer));
                    }
                    return false;
                }
            default:
                return false;
            }
        }
    case CV_32FC1:
        {
            switch(dst.type())
            {
            case CV_32FC1:
                {
                    if ((dx == 1) && (dy == 0))
                    {
                        if (0 > ippiFilterScharrVertGetBufferSize_32f_C1R(ippiSize(src.cols, src.rows),&bufSize))
                            return false;
                        buffer.allocate(bufSize);

                        if (0 > ippiFilterScharrVertBorder_32f_C1R((const Ipp32f*)src.data, (int)src.step,
                                        (Ipp32f*)dst.data, (int)dst.step, ippiSize(src.cols, src.rows),
                                        ippBorderRepl, 0, (Ipp8u*)(char*)buffer))
                        {
                            return false;
                        }

                        if (scale != 1)
                            /* IPP is fast, so MulC produce very little perf degradation.*/
                            //ippiMulC_32f_C1IR((Ipp32f)scale, (Ipp32f*)dst.data, (int)dst.step, ippiSize(dst.cols*dst.channels(), dst.rows));
                            ippiMulC_32f_C1R((Ipp32f*)dst.data, (int)dst.step, (Ipp32f)scale, (Ipp32f*)dst.data, (int)dst.step, ippiSize(dst.cols*dst.channels(), dst.rows));
                        return true;
                    }
                    if ((dx == 0) && (dy == 1))
                    {
                        if (0 > ippiFilterScharrHorizGetBufferSize_32f_C1R(ippiSize(src.cols, src.rows),&bufSize))
                            return false;
                        buffer.allocate(bufSize);

                        if (0 > ippiFilterScharrHorizBorder_32f_C1R((const Ipp32f*)src.data, (int)src.step,
                                        (Ipp32f*)dst.data, (int)dst.step, ippiSize(src.cols, src.rows),
                                        ippBorderRepl, 0, (Ipp8u*)(char*)buffer))
                            return false;

                        if (scale != 1)
                            ippiMulC_32f_C1R((Ipp32f *)dst.data, (int)dst.step, (Ipp32f)scale, (Ipp32f *)dst.data, (int)dst.step, ippiSize(dst.cols*dst.channels(), dst.rows));
                        return true;
                    }
                }
            default:
                return false;
            }
        }
    default:
        return false;
    }
}
#endif

static bool IPPDerivSobel(InputArray _src, OutputArray _dst, int ddepth, int dx, int dy, int ksize, double scale, double delta, int borderType)
{
    if ((borderType != BORDER_REPLICATE) || ((3 != ksize) && (5 != ksize)))
        return false;
    if (fabs(delta) > FLT_EPSILON)
        return false;
    if (1 != _src.channels())
        return false;

    int bufSize = 0;
    cv::AutoBuffer<char> buffer;
    Mat src = _src.getMat(), dst = _dst.getMat();
    if ( ddepth < 0 )
        ddepth = src.depth();

    if (src.type() == CV_8U && dst.type() == CV_16S && scale == 1)
    {
        if ((dx == 1) && (dy == 0))
        {
            if (0 > ippiFilterSobelNegVertGetBufferSize_8u16s_C1R(ippiSize(src.cols, src.rows), (IppiMaskSize)(ksize*10+ksize),&bufSize))
                IPP_RETURN_ERROR
            buffer.allocate(bufSize);

            if (0 > ippiFilterSobelNegVertBorder_8u16s_C1R((const Ipp8u*)src.data, (int)src.step,
                                (Ipp16s*)dst.data, (int)dst.step, ippiSize(src.cols, src.rows), (IppiMaskSize)(ksize*10+ksize),
                                ippBorderRepl, 0, (Ipp8u*)(char*)buffer))
                IPP_RETURN_ERROR
            return true;
        }

        if ((dx == 0) && (dy == 1))
        {
            if (0 > ippiFilterSobelHorizGetBufferSize_8u16s_C1R(ippiSize(src.cols, src.rows), (IppiMaskSize)(ksize*10+ksize),&bufSize))
                IPP_RETURN_ERROR
            buffer.allocate(bufSize);

            if (0 > ippiFilterSobelHorizBorder_8u16s_C1R((const Ipp8u*)src.data, (int)src.step,
                                (Ipp16s*)dst.data, (int)dst.step, ippiSize(src.cols, src.rows), (IppiMaskSize)(ksize*10+ksize),
                                ippBorderRepl, 0, (Ipp8u*)(char*)buffer))
                IPP_RETURN_ERROR
            return true;
        }

#if !defined(HAVE_IPP_ICV_ONLY)
        if ((dx == 2) && (dy == 0))
        {
            if (0 > ippiFilterSobelVertSecondGetBufferSize_8u16s_C1R(ippiSize(src.cols, src.rows), (IppiMaskSize)(ksize*10+ksize),&bufSize))
                IPP_RETURN_ERROR
            buffer.allocate(bufSize);

            if (0 > ippiFilterSobelVertSecondBorder_8u16s_C1R((const Ipp8u*)src.data, (int)src.step,
                                (Ipp16s*)dst.data, (int)dst.step, ippiSize(src.cols, src.rows), (IppiMaskSize)(ksize*10+ksize),
                                ippBorderRepl, 0, (Ipp8u*)(char*)buffer))
                IPP_RETURN_ERROR
            return true;
        }

        if ((dx == 0) && (dy == 2))
        {
            if (0 > ippiFilterSobelHorizSecondGetBufferSize_8u16s_C1R(ippiSize(src.cols, src.rows), (IppiMaskSize)(ksize*10+ksize),&bufSize))
                IPP_RETURN_ERROR
            buffer.allocate(bufSize);

            if (0 > ippiFilterSobelHorizSecondBorder_8u16s_C1R((const Ipp8u*)src.data, (int)src.step,
                                (Ipp16s*)dst.data, (int)dst.step, ippiSize(src.cols, src.rows), (IppiMaskSize)(ksize*10+ksize),
                                ippBorderRepl, 0, (Ipp8u*)(char*)buffer))
                IPP_RETURN_ERROR
            return true;
        }
#endif
    }

    if (src.type() == CV_32F && dst.type() == CV_32F)
    {
#if 0
        if ((dx == 1) && (dy == 0))
        {
            if (0 > ippiFilterSobelNegVertGetBufferSize_32f_C1R(ippiSize(src.cols, src.rows), (IppiMaskSize)(ksize*10+ksize), &bufSize))
                IPP_RETURN_ERROR
            buffer.allocate(bufSize);

            if (0 > ippiFilterSobelNegVertBorder_32f_C1R((const Ipp32f*)src.data, (int)src.step,
                            (Ipp32f*)dst.data, (int)dst.step, ippiSize(src.cols, src.rows), (IppiMaskSize)(ksize*10+ksize),
                            ippBorderRepl, 0, (Ipp8u*)(char*)buffer))
                IPP_RETURN_ERROR
            if(scale != 1)
                ippiMulC_32f_C1R((Ipp32f *)dst.data, (int)dst.step, (Ipp32f)scale, (Ipp32f *)dst.data, (int)dst.step, ippiSize(dst.cols*dst.channels(), dst.rows));
            return true;
        }

        if ((dx == 0) && (dy == 1))
        {
            if (0 > ippiFilterSobelHorizGetBufferSize_32f_C1R(ippiSize(src.cols, src.rows), (IppiMaskSize)(ksize*10+ksize),&bufSize))
                IPP_RETURN_ERROR
            buffer.allocate(bufSize);
            if (0 > ippiFilterSobelHorizBorder_32f_C1R((const Ipp32f*)src.data, (int)src.step,
                            (Ipp32f*)dst.data, (int)dst.step, ippiSize(src.cols, src.rows), (IppiMaskSize)(ksize*10+ksize),
                            ippBorderRepl, 0, (Ipp8u*)(char*)buffer))
                IPP_RETURN_ERROR
            if(scale != 1)
                ippiMulC_32f_C1R((Ipp32f *)dst.data, (int)dst.step, (Ipp32f)scale, (Ipp32f *)dst.data, (int)dst.step, ippiSize(dst.cols*dst.channels(), dst.rows));
            return true;
        }
#endif
#if !defined(HAVE_IPP_ICV_ONLY)
        if((dx == 2) && (dy == 0))
        {
            if (0 > ippiFilterSobelVertSecondGetBufferSize_32f_C1R(ippiSize(src.cols, src.rows), (IppiMaskSize)(ksize*10+ksize),&bufSize))
                IPP_RETURN_ERROR
            buffer.allocate(bufSize);

            if (0 > ippiFilterSobelVertSecondBorder_32f_C1R((const Ipp32f*)src.data, (int)src.step,
                            (Ipp32f*)dst.data, (int)dst.step, ippiSize(src.cols, src.rows), (IppiMaskSize)(ksize*10+ksize),
                            ippBorderRepl, 0, (Ipp8u*)(char*)buffer))
                IPP_RETURN_ERROR
            if(scale != 1)
                ippiMulC_32f_C1R((Ipp32f *)dst.data, (int)dst.step, (Ipp32f)scale, (Ipp32f *)dst.data, (int)dst.step, ippiSize(dst.cols*dst.channels(), dst.rows));
            return true;
        }

        if((dx == 0) && (dy == 2))
        {
            if (0 > ippiFilterSobelHorizSecondGetBufferSize_32f_C1R(ippiSize(src.cols, src.rows), (IppiMaskSize)(ksize*10+ksize),&bufSize))
                IPP_RETURN_ERROR
            buffer.allocate(bufSize);

            if (0 > ippiFilterSobelHorizSecondBorder_32f_C1R((const Ipp32f*)src.data, (int)src.step,
                            (Ipp32f*)dst.data, (int)dst.step, ippiSize(src.cols, src.rows), (IppiMaskSize)(ksize*10+ksize),
                            ippBorderRepl, 0, (Ipp8u*)(char*)buffer))
                IPP_RETURN_ERROR

            if(scale != 1)
                ippiMulC_32f_C1R((Ipp32f *)dst.data, (int)dst.step, (Ipp32f)scale, (Ipp32f *)dst.data, (int)dst.step, ippiSize(dst.cols*dst.channels(), dst.rows));
            return true;
        }
#endif
    }
    return false;
}

}

#endif

void cv::Sobel( InputArray _src, OutputArray _dst, int ddepth, int dx, int dy,
                int ksize, double scale, double delta, int borderType )
{
    int stype = _src.type(), sdepth = CV_MAT_DEPTH(stype), cn = CV_MAT_CN(stype);
    if (ddepth < 0)
        ddepth = sdepth;
    int dtype = CV_MAKE_TYPE(ddepth, cn);
    _dst.create( _src.size(), dtype );

#ifdef HAVE_TEGRA_OPTIMIZATION
    if (scale == 1.0 && delta == 0)
    {
        Mat src = _src.getMat(), dst = _dst.getMat();
        if (ksize == 3 && tegra::sobel3x3(src, dst, dx, dy, borderType))
            return;
        if (ksize == -1 && tegra::scharr(src, dst, dx, dy, borderType))
            return;
    }
#endif

#ifdef HAVE_IPP
    if (ksize < 0)
    {
        if (IPPDerivScharr(_src, _dst, ddepth, dx, dy, scale, delta, borderType))
            return;
    }
    else if (0 < ksize)
    {
        if (IPPDerivSobel(_src, _dst, ddepth, dx, dy, ksize, scale, delta, borderType))
            return;
    }
#endif
    int ktype = std::max(CV_32F, std::max(ddepth, sdepth));

    Mat kx, ky;
    getDerivKernels( kx, ky, dx, dy, ksize, false, ktype );
    if( scale != 1 )
    {
        // usually the smoothing part is the slowest to compute,
        // so try to scale it instead of the faster differenciating part
        if( dx == 0 )
            kx *= scale;
        else
            ky *= scale;
    }
    sepFilter2D( _src, _dst, ddepth, kx, ky, Point(-1, -1), delta, borderType );
}


void cv::Scharr( InputArray _src, OutputArray _dst, int ddepth, int dx, int dy,
                 double scale, double delta, int borderType )
{
    int stype = _src.type(), sdepth = CV_MAT_DEPTH(stype), cn = CV_MAT_CN(stype);
    if (ddepth < 0)
        ddepth = sdepth;
    int dtype = CV_MAKETYPE(ddepth, cn);
    _dst.create( _src.size(), dtype );

#ifdef HAVE_TEGRA_OPTIMIZATION
    if (scale == 1.0 && delta == 0)
    {
        Mat src = _src.getMat(), dst = _dst.getMat();
        if (tegra::scharr(src, dst, dx, dy, borderType))
            return;
    }
#endif

#if defined (HAVE_IPP) && (IPP_VERSION_MAJOR >= 7)
    if (IPPDerivScharr(_src, _dst, ddepth, dx, dy, scale, delta, borderType))
        return;
#endif
    int ktype = std::max(CV_32F, std::max(ddepth, sdepth));

    Mat kx, ky;
    getScharrKernels( kx, ky, dx, dy, false, ktype );
    if( scale != 1 )
    {
        // usually the smoothing part is the slowest to compute,
        // so try to scale it instead of the faster differenciating part
        if( dx == 0 )
            kx *= scale;
        else
            ky *= scale;
    }
    sepFilter2D( _src, _dst, ddepth, kx, ky, Point(-1, -1), delta, borderType );
}

#ifdef HAVE_OPENCL

namespace cv {

static bool ocl_Laplacian5(InputArray _src, OutputArray _dst,
                           const Mat & kd, const Mat & ks, double scale, double delta,
                           int borderType, int depth, int ddepth)
{
    int iscale = cvRound(scale), idelta = cvRound(delta);
    bool doubleSupport = ocl::Device::getDefault().doubleFPConfig() > 0,
            floatCoeff = std::fabs(delta - idelta) > DBL_EPSILON || std::fabs(scale - iscale) > DBL_EPSILON;
    int cn = _src.channels(), wdepth = std::max(depth, floatCoeff ? CV_32F : CV_32S), kercn = 1;

    if (!doubleSupport && wdepth == CV_64F)
        return false;

    char cvt[2][40];
    ocl::Kernel k("sumConvert", ocl::imgproc::laplacian5_oclsrc,
                  format("-D srcT=%s -D WT=%s -D dstT=%s -D coeffT=%s -D wdepth=%d "
                         "-D convertToWT=%s -D convertToDT=%s%s",
                         ocl::typeToStr(CV_MAKE_TYPE(depth, kercn)),
                         ocl::typeToStr(CV_MAKE_TYPE(wdepth, kercn)),
                         ocl::typeToStr(CV_MAKE_TYPE(ddepth, kercn)),
                         ocl::typeToStr(wdepth), wdepth,
                         ocl::convertTypeStr(depth, wdepth, kercn, cvt[0]),
                         ocl::convertTypeStr(wdepth, ddepth, kercn, cvt[1]),
                         doubleSupport ? " -D DOUBLE_SUPPORT" : ""));
    if (k.empty())
        return false;

    UMat d2x, d2y;
    sepFilter2D(_src, d2x, depth, kd, ks, Point(-1, -1), 0, borderType);
    sepFilter2D(_src, d2y, depth, ks, kd, Point(-1, -1), 0, borderType);

    UMat dst = _dst.getUMat();

    ocl::KernelArg d2xarg = ocl::KernelArg::ReadOnlyNoSize(d2x),
            d2yarg = ocl::KernelArg::ReadOnlyNoSize(d2y),
            dstarg = ocl::KernelArg::WriteOnly(dst, cn, kercn);

    if (wdepth >= CV_32F)
        k.args(d2xarg, d2yarg, dstarg, (float)scale, (float)delta);
    else
        k.args(d2xarg, d2yarg, dstarg, iscale, idelta);

    size_t globalsize[] = { dst.cols * cn / kercn, dst.rows };
    return k.run(2, globalsize, NULL, false);
}

}

#endif

void cv::Laplacian( InputArray _src, OutputArray _dst, int ddepth, int ksize,
                    double scale, double delta, int borderType )
{
    int stype = _src.type(), sdepth = CV_MAT_DEPTH(stype), cn = CV_MAT_CN(stype);
    if (ddepth < 0)
        ddepth = sdepth;
    _dst.create( _src.size(), CV_MAKETYPE(ddepth, cn) );

#ifdef HAVE_IPP
    if ((ksize == 3 || ksize == 5) && ((borderType & BORDER_ISOLATED) != 0 || !_src.isSubmatrix()) &&
        ((stype == CV_8UC1 && ddepth == CV_16S) || (ddepth == CV_32F && stype == CV_32FC1)))
    {
        int iscale = saturate_cast<int>(scale), idelta = saturate_cast<int>(delta);
        bool floatScale = std::fabs(scale - iscale) > DBL_EPSILON, needScale = iscale != 1;
        bool floatDelta = std::fabs(delta - idelta) > DBL_EPSILON, needDelta = delta != 0;
        int borderTypeNI = borderType & ~BORDER_ISOLATED;
        Mat src = _src.getMat(), dst = _dst.getMat();

        if (src.data != dst.data)
        {
            Ipp32s bufsize;
            IppStatus status = (IppStatus)-1;
            IppiSize roisize = { src.cols, src.rows };
            IppiMaskSize masksize = ksize == 3 ? ippMskSize3x3 : ippMskSize5x5;
            IppiBorderType borderTypeIpp = ippiGetBorderType(borderTypeNI);

#define IPP_FILTER_LAPLACIAN(ippsrctype, ippdsttype, ippfavor) \
    do \
    { \
        if (borderTypeIpp >= 0 && ippiFilterLaplacianGetBufferSize_##ippfavor##_C1R(roisize, masksize, &bufsize) >= 0) \
        { \
            Ipp8u * buffer = ippsMalloc_8u(bufsize); \
            status = ippiFilterLaplacianBorder_##ippfavor##_C1R((const ippsrctype *)src.data, (int)src.step, (ippdsttype *)dst.data, \
                                                                (int)dst.step, roisize, masksize, borderTypeIpp, 0, buffer); \
            ippsFree(buffer); \
        } \
    } while ((void)0, 0)

            CV_SUPPRESS_DEPRECATED_START
            if (sdepth == CV_8U && ddepth == CV_16S && !floatScale && !floatDelta)
            {
                IPP_FILTER_LAPLACIAN(Ipp8u, Ipp16s, 8u16s);

                if (needScale && status >= 0)
                    status = ippiMulC_16s_C1IRSfs((Ipp16s)iscale, (Ipp16s *)dst.data, (int)dst.step, roisize, 0);
                if (needDelta && status >= 0)
                    status = ippiAddC_16s_C1IRSfs((Ipp16s)idelta, (Ipp16s *)dst.data, (int)dst.step, roisize, 0);
            }
            else if (sdepth == CV_32F && ddepth == CV_32F)
            {
                IPP_FILTER_LAPLACIAN(Ipp32f, Ipp32f, 32f);

                if (needScale && status >= 0)
                    status = ippiMulC_32f_C1IR((Ipp32f)scale, (Ipp32f *)dst.data, (int)dst.step, roisize);
                if (needDelta && status >= 0)
                    status = ippiAddC_32f_C1IR((Ipp32f)delta, (Ipp32f *)dst.data, (int)dst.step, roisize);
            }
            CV_SUPPRESS_DEPRECATED_END

            if (status >= 0)
                return;
            setIppErrorStatus();
        }
    }
#undef IPP_FILTER_LAPLACIAN
#endif

#ifdef HAVE_TEGRA_OPTIMIZATION
    if (scale == 1.0 && delta == 0)
    {
        Mat src = _src.getMat(), dst = _dst.getMat();
        if (ksize == 1 && tegra::laplace1(src, dst, borderType))
            return;
        if (ksize == 3 && tegra::laplace3(src, dst, borderType))
            return;
        if (ksize == 5 && tegra::laplace5(src, dst, borderType))
            return;
    }
#endif

    if( ksize == 1 || ksize == 3 )
    {
        float K[2][9] =
        {
            { 0, 1, 0, 1, -4, 1, 0, 1, 0 },
            { 2, 0, 2, 0, -8, 0, 2, 0, 2 }
        };
        Mat kernel(3, 3, CV_32F, K[ksize == 3]);
        if( scale != 1 )
            kernel *= scale;
        filter2D( _src, _dst, ddepth, kernel, Point(-1, -1), delta, borderType );
    }
    else
    {
        int ktype = std::max(CV_32F, std::max(ddepth, sdepth));
        int wdepth = sdepth == CV_8U && ksize <= 5 ? CV_16S : sdepth <= CV_32F ? CV_32F : CV_64F;
        int wtype = CV_MAKETYPE(wdepth, cn);
        Mat kd, ks;
        getSobelKernels( kd, ks, 2, 0, ksize, false, ktype );

        CV_OCL_RUN(_dst.isUMat(),
                   ocl_Laplacian5(_src, _dst, kd, ks, scale,
                                  delta, borderType, wdepth, ddepth))

        const size_t STRIPE_SIZE = 1 << 14;
        Ptr<FilterEngine> fx = createSeparableLinearFilter(stype,
            wtype, kd, ks, Point(-1,-1), 0, borderType, borderType, Scalar() );
        Ptr<FilterEngine> fy = createSeparableLinearFilter(stype,
            wtype, ks, kd, Point(-1,-1), 0, borderType, borderType, Scalar() );

        Mat src = _src.getMat(), dst = _dst.getMat();
        int y = fx->start(src), dsty = 0, dy = 0;
        fy->start(src);
        const uchar* sptr = src.data + y*src.step;

        int dy0 = std::min(std::max((int)(STRIPE_SIZE/(CV_ELEM_SIZE(stype)*src.cols)), 1), src.rows);
        Mat d2x( dy0 + kd.rows - 1, src.cols, wtype );
        Mat d2y( dy0 + kd.rows - 1, src.cols, wtype );

        for( ; dsty < src.rows; sptr += dy0*src.step, dsty += dy )
        {
            fx->proceed( sptr, (int)src.step, dy0, d2x.data, (int)d2x.step );
            dy = fy->proceed( sptr, (int)src.step, dy0, d2y.data, (int)d2y.step );
            if( dy > 0 )
            {
                Mat dstripe = dst.rowRange(dsty, dsty + dy);
                d2x.rows = d2y.rows = dy; // modify the headers, which should work
                d2x += d2y;
                d2x.convertTo( dstripe, ddepth, scale, delta );
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////

CV_IMPL void
cvSobel( const void* srcarr, void* dstarr, int dx, int dy, int aperture_size )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);

    CV_Assert( src.size() == dst.size() && src.channels() == dst.channels() );

    cv::Sobel( src, dst, dst.depth(), dx, dy, aperture_size, 1, 0, cv::BORDER_REPLICATE );
    if( CV_IS_IMAGE(srcarr) && ((IplImage*)srcarr)->origin && dy % 2 != 0 )
        dst *= -1;
}


CV_IMPL void
cvLaplace( const void* srcarr, void* dstarr, int aperture_size )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);

    CV_Assert( src.size() == dst.size() && src.channels() == dst.channels() );

    cv::Laplacian( src, dst, dst.depth(), aperture_size, 1, 0, cv::BORDER_REPLICATE );
}

/* End of file. */
