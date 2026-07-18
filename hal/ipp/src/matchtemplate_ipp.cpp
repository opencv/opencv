// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "ipp_hal_imgproc.hpp"

#include <opencv2/core.hpp>
#include "precomp_ipp.hpp"

#if IPP_VERSION_X100 >= 700

using namespace cv;

#if IPP_VERSION_X100 >= 201700
#define IPP_HAL_MALLOC(SIZE) ippMalloc_L(SIZE)
#else
#define IPP_HAL_MALLOC(SIZE) ippMalloc((int)(SIZE))
#endif

typedef IppStatus (CV_STDCALL * ippimatchTemplate)(const void*, int, IppiSize, const void*, int, IppiSize, Ipp32f* , int , IppEnum , Ipp8u*);

static bool ipp_crossCorr(const Mat& src, const Mat& tpl, Mat& dst, bool normed)
{
    IppStatus status;

    IppiSize srcRoiSize = {src.cols,src.rows};
    IppiSize tplRoiSize = {tpl.cols,tpl.rows};

    int bufSize=0;

    int depth = src.depth();

    ippimatchTemplate ippiCrossCorrNorm =
            depth==CV_8U ? (ippimatchTemplate)ippiCrossCorrNorm_8u32f_C1R:
            depth==CV_32F? (ippimatchTemplate)ippiCrossCorrNorm_32f_C1R: 0;

    if (ippiCrossCorrNorm==0)
        return false;

    IppEnum funCfg = (IppEnum)(+ippAlgAuto | ippiROIValid);
    if(normed)
        funCfg |= ippiNorm;
    else
        funCfg |= ippiNormNone;

    status = ippiCrossCorrNormGetBufferSize(srcRoiSize, tplRoiSize, funCfg, &bufSize);
    if ( status < 0 )
        return false;

    Ipp8u* buffer = bufSize > 0 ? (Ipp8u*)IPP_HAL_MALLOC(bufSize) : 0;
    if (bufSize > 0 && buffer == 0)
        return false;

    status = CV_INSTRUMENT_FUN_IPP(ippiCrossCorrNorm, src.ptr(), (int)src.step, srcRoiSize, tpl.ptr(), (int)tpl.step, tplRoiSize, dst.ptr<Ipp32f>(), (int)dst.step, funCfg, buffer);

    if (buffer)
        ippFree(buffer);
    return status >= 0;
}

static bool ipp_sqrDistance(const Mat& src, const Mat& tpl, Mat& dst)
{
    IppStatus status;

    IppiSize srcRoiSize = {src.cols,src.rows};
    IppiSize tplRoiSize = {tpl.cols,tpl.rows};

    int bufSize=0;

    int depth = src.depth();

    ippimatchTemplate ippiSqrDistanceNorm =
            depth==CV_8U ? (ippimatchTemplate)ippiSqrDistanceNorm_8u32f_C1R:
            depth==CV_32F? (ippimatchTemplate)ippiSqrDistanceNorm_32f_C1R: 0;

    if (ippiSqrDistanceNorm==0)
        return false;

    IppEnum funCfg = (IppEnum)(+ippAlgAuto | ippiROIValid | ippiNormNone);
    status = ippiSqrDistanceNormGetBufferSize(srcRoiSize, tplRoiSize, funCfg, &bufSize);
    if ( status < 0 )
        return false;

    Ipp8u* buffer = bufSize > 0 ? (Ipp8u*)IPP_HAL_MALLOC(bufSize) : 0;
    if (bufSize > 0 && buffer == 0)
        return false;

    status = CV_INSTRUMENT_FUN_IPP(ippiSqrDistanceNorm, src.ptr(), (int)src.step, srcRoiSize, tpl.ptr(), (int)tpl.step, tplRoiSize, dst.ptr<Ipp32f>(), (int)dst.step, funCfg, buffer);

    if (buffer)
        ippFree(buffer);
    dst = cv::max(dst, 0); // handle edge case from rounding in variance computation which can result in negative values
    return status >= 0;
}

int ipp_hal_matchTemplate(const uchar* src_data, size_t src_step, int src_width, int src_height,
                          const uchar* templ_data, size_t templ_step, int templ_width, int templ_height,
                          float* result_data, size_t result_step, int depth, int cn, int method)
{
    CV_HAL_CHECK_USE_IPP();

    if(cn != 1)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if(depth != CV_8U && depth != CV_32F)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    Mat img(src_height, src_width, CV_MAKETYPE(depth, cn), (void*)src_data, src_step);
    Mat templ(templ_height, templ_width, CV_MAKETYPE(depth, cn), (void*)templ_data, templ_step);
    Mat result(src_height - templ_height + 1, src_width - templ_width + 1, CV_32FC1, (void*)result_data, result_step);

    // These functions are not efficient if template size is comparable with image size
    if(templ.size().area()*4 > img.size().area())
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    // CV_8U SQDIFF/SQDIFF_NORMED suffer from float32 catastrophic cancellation
    // in IPP's internal accumulators; fall through to the double-precision path instead.
    if(depth == CV_8U && (method == cv::TM_SQDIFF || method == cv::TM_SQDIFF_NORMED))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    if(method == cv::TM_SQDIFF)
    {
        if(ipp_sqrDistance(img, templ, result))
            return CV_HAL_ERROR_OK;
    }
    else if(method == cv::TM_CCORR_NORMED)
    {
        if(ipp_crossCorr(img, templ, result, true))
            return CV_HAL_ERROR_OK;
    }
    else if(method == cv::TM_SQDIFF_NORMED || method == cv::TM_CCORR ||
            method == cv::TM_CCOEFF || method == cv::TM_CCOEFF_NORMED)
    {
        // Raw cross-correlation; caller finishes SQDIFF_NORMED/CCOEFF/CCOEFF_NORMED via
        // common_matchTemplate (TM_CCORR is already final).
        if(ipp_crossCorr(img, templ, result, false))
            return CV_HAL_ERROR_OK;
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

#endif // IPP_VERSION_X100 >= 700
