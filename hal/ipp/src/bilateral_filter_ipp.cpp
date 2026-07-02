// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "ipp_hal_imgproc.hpp"

#ifdef HAVE_IPP_IW

#include <opencv2/core.hpp>
#include "precomp_ipp.hpp"

#define IPP_BILATERAL_PARALLEL 1

class ipp_bilateralFilterParallel: public cv::ParallelLoopBody
{
public:
    ipp_bilateralFilterParallel(::ipp::IwiImage &_src, ::ipp::IwiImage &_dst, int _radius, Ipp32f _valSquareSigma, Ipp32f _posSquareSigma, ::ipp::IwiBorderType _borderType, bool *_ok):
        src(_src), dst(_dst)
    {
        pOk = _ok;

        radius          = _radius;
        valSquareSigma  = _valSquareSigma;
        posSquareSigma  = _posSquareSigma;
        borderType      = _borderType;

        *pOk = true;
    }
    ~ipp_bilateralFilterParallel() {}

    virtual void operator() (const cv::Range& range) const CV_OVERRIDE
    {
        if(*pOk == false)
            return;

        try
        {
            ::ipp::IwiTile tile = ::ipp::IwiRoi(0, range.start, dst.m_size.width, range.end - range.start);
            CV_INSTRUMENT_FUN_IPP(::ipp::iwiFilterBilateral, src, dst, radius, valSquareSigma, posSquareSigma, ::ipp::IwDefault(), borderType, tile);
        }
        catch(const ::ipp::IwException &)
        {
            *pOk = false;
            return;
        }
    }
private:
    ::ipp::IwiImage &src;
    ::ipp::IwiImage &dst;

    int                  radius;
    Ipp32f               valSquareSigma;
    Ipp32f               posSquareSigma;
    ::ipp::IwiBorderType borderType;

    bool  *pOk;
    const ipp_bilateralFilterParallel& operator= (const ipp_bilateralFilterParallel&);
};

int ipp_hal_bilateralFilter(const uchar* src_data, size_t src_step,
                            uchar* dst_data, size_t dst_step,
                            int width, int height, int full_width, int full_height, int offset_x, int offset_y,
                            int depth, int cn, int d,
                            double sigma_color, double sigma_space, int border_type)
{
    CV_HAL_CHECK_USE_IPP();

    if(!((depth == CV_8U || depth == CV_32F) && (cn == 1 || cn == 3)))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    constexpr double eps = 1e-6;
    if(sigma_color <= eps || sigma_space <= eps)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    int bt = border_type & ~cv::BORDER_ISOLATED;
    IppiBorderType ippBorderType =
        bt == cv::BORDER_CONSTANT    ? ippBorderConst  :
        bt == cv::BORDER_REPLICATE   ? ippBorderRepl   :
        bt == cv::BORDER_REFLECT_101 ? ippBorderMirror :
        bt == cv::BORDER_TRANSPARENT ? ippBorderTransp :
        (IppiBorderType)-1;
    if((int)ippBorderType == -1)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    int    radius         = IPP_MAX(((d <= 0) ? cvRound(sigma_space*1.5) : d/2), 1);
    Ipp32f valSquareSigma = (Ipp32f)(sigma_color*sigma_color);
    Ipp32f posSquareSigma = (Ipp32f)(sigma_space*sigma_space);

    IwSize marginLeft   = offset_x;
    IwSize marginTop    = offset_y;
    IwSize marginRight  = full_width  - width  - offset_x;
    IwSize marginBottom = full_height - height - offset_y;

    try
    {
        ::ipp::IwiBorderSize inMemBorder(marginLeft, marginTop, marginRight, marginBottom);

        ::ipp::IwiImage iwSrc, iwDst;
        iwSrc.Init(IwiSize{width, height}, ippiGetDataType(depth), cn,
                   inMemBorder, (void*)src_data, IwSize(src_step));
        iwDst.Init(IwiSize{width, height}, ippiGetDataType(depth), cn,
                   IwiBorderSize(), dst_data, IwSize(dst_step));

        // already have physical border
        int inMemFlags = 0;
        if(!(border_type & cv::BORDER_ISOLATED))
        {
            if(marginLeft)
            {
                if(marginLeft >= radius) inMemFlags |= ippBorderInMemLeft;
                else return CV_HAL_ERROR_NOT_IMPLEMENTED;
            }
            if(marginTop)
            {
                if(marginTop >= radius) inMemFlags |= ippBorderInMemTop;
                else return CV_HAL_ERROR_NOT_IMPLEMENTED;
            }
            if(marginRight)
            {
                if(marginRight >= radius) inMemFlags |= ippBorderInMemRight;
                else return CV_HAL_ERROR_NOT_IMPLEMENTED;
            }
            if(marginBottom)
            {
                if(marginBottom >= radius) inMemFlags |= ippBorderInMemBottom;
                else return CV_HAL_ERROR_NOT_IMPLEMENTED;
            }
        }

        ::ipp::IwiBorderType ippBorder((IppiBorderType)(ippBorderType | inMemFlags));

        const int threads = ippiSuggestThreadsNum(iwDst, 2);
        if(IPP_BILATERAL_PARALLEL && threads > 1) {
            bool  ok      = true;
            cv::Range range(0, (int)iwDst.m_size.height);
            ipp_bilateralFilterParallel invoker(iwSrc, iwDst, radius, valSquareSigma, posSquareSigma, ippBorder, &ok);
            if(!ok)
                return CV_HAL_ERROR_NOT_IMPLEMENTED;

            int maxTiles = (int)iwDst.m_size.height / radius;
            int numTiles = threads * 4;
            if(numTiles > maxTiles) {
                numTiles = (maxTiles / threads) * threads;
            }
            cv::parallel_for_(range, invoker, numTiles);

            if(!ok)
                return CV_HAL_ERROR_NOT_IMPLEMENTED;
        } else {
            CV_INSTRUMENT_FUN_IPP(::ipp::iwiFilterBilateral, iwSrc, iwDst, radius, valSquareSigma, posSquareSigma, ::ipp::IwDefault(), ippBorder);
        }
    }
    catch(const ::ipp::IwException &)
    {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }
    return CV_HAL_ERROR_OK;
}

#endif
