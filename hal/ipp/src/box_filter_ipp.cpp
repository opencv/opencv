// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "ipp_hal_imgproc.hpp"

#include <opencv2/core.hpp>
#include "precomp_ipp.hpp"

#if defined(HAVE_IPP_IW) && !DISABLE_IPP_BOX_FILTER

namespace cv { namespace ipp { unsigned long long getIppTopFeatures(); } }

// Copied from core/private.hpp (gated by HAVE_IPP, which the plugin lacks).
// boxGetIppBorderType: distinct name, adds the REFLECT_101 precomp's version drops.
static inline IppiBorderType boxGetIppBorderType(int borderTypeNI)
{
    return borderTypeNI == cv::BORDER_CONSTANT    ? ippBorderConst  :
           borderTypeNI == cv::BORDER_TRANSPARENT ? ippBorderTransp :
           borderTypeNI == cv::BORDER_REPLICATE   ? ippBorderRepl   :
           borderTypeNI == cv::BORDER_REFLECT_101 ? ippBorderMirror :
           (IppiBorderType)-1;
}

static inline bool ippiCheckAnchor(int x, int y, int kernelWidth, int kernelHeight)
{
    return (x == (kernelWidth - 1)/2 && y == (kernelHeight - 1)/2);
}

static inline IppiBorderType ippiGetBorder(::ipp::IwiImage &image, int ocvBorderType, ::ipp::IwiBorderSize &borderSize)
{
    int            inMemFlags = 0;
    IppiBorderType border     = boxGetIppBorderType(ocvBorderType & ~cv::BORDER_ISOLATED);
    if((int)border == -1)
        return (IppiBorderType)0;

    if(!(ocvBorderType & cv::BORDER_ISOLATED))
    {
        if(image.m_inMemSize.left)
        {
            if(image.m_inMemSize.left >= borderSize.left)
                inMemFlags |= ippBorderInMemLeft;
            else
                return (IppiBorderType)0;
        }
        else
            borderSize.left = 0;
        if(image.m_inMemSize.top)
        {
            if(image.m_inMemSize.top >= borderSize.top)
                inMemFlags |= ippBorderInMemTop;
            else
                return (IppiBorderType)0;
        }
        else
            borderSize.top = 0;
        if(image.m_inMemSize.right)
        {
            if(image.m_inMemSize.right >= borderSize.right)
                inMemFlags |= ippBorderInMemRight;
            else
                return (IppiBorderType)0;
        }
        else
            borderSize.right = 0;
        if(image.m_inMemSize.bottom)
        {
            if(image.m_inMemSize.bottom >= borderSize.bottom)
                inMemFlags |= ippBorderInMemBottom;
            else
                return (IppiBorderType)0;
        }
        else
            borderSize.bottom = 0;
    }
    else
        borderSize.left = borderSize.right = borderSize.top = borderSize.bottom = 0;

    return (IppiBorderType)(border | inMemFlags);
}

int ipp_hal_boxFilter(const uchar* src_data, size_t src_step,
                      uchar* dst_data, size_t dst_step,
                      int width, int height, int src_depth, int dst_depth, int cn,
                      int margin_left, int margin_top, int margin_right, int margin_bottom,
                      size_t ksize_width, size_t ksize_height,
                      int anchor_x, int anchor_y,
                      bool normalize, int border_type)
{
    CV_HAL_CHECK_USE_IPP();

#if IPP_VERSION_X100 < 201801
    // Problem with SSE42 optimization for 16s and some 8u modes
    if(cv::ipp::getIppTopFeatures() == ippCPUID_SSE42 &&
       (((src_depth == CV_16S || src_depth == CV_16U) && (cn == 3 || cn == 4)) ||
        (src_depth == CV_8U && cn == 3 && (ksize_width > 5 || ksize_height > 5))))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    // Other optimizations has some degradations too
    if(((src_depth == CV_16S || src_depth == CV_16U) && cn == 4) ||
       (src_depth == CV_8U && cn == 1 && (ksize_width > 5 || ksize_height > 5)))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
#endif

    if(!normalize)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    if(!ippiCheckAnchor(anchor_x, anchor_y, (int)ksize_width, (int)ksize_height))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    try
    {
        // raw-pointer equivalent of ippiGetImage: margins are the in-memory border
        ::ipp::IwiBorderSize inMemBorder;
        inMemBorder.left   = (IwSize)margin_left;
        inMemBorder.top    = (IwSize)margin_top;
        inMemBorder.right  = (IwSize)margin_right;
        inMemBorder.bottom = (IwSize)margin_bottom;

        ::ipp::IwiImage iwSrc, iwDst;
        iwSrc.Init(IwiSize{width, height}, ippiGetDataType(src_depth), cn,
                   inMemBorder, (void*)src_data, IwSize(src_step));
        iwDst.Init(IwiSize{width, height}, ippiGetDataType(dst_depth), cn,
                   ::ipp::IwiBorderSize(), dst_data, IwSize(dst_step));

        ::ipp::IwiSize       iwKSize{(int)ksize_width, (int)ksize_height};
        ::ipp::IwiBorderSize borderSize(iwKSize);
        ::ipp::IwiBorderType ippBorder(ippiGetBorder(iwSrc, border_type, borderSize));
        if(!ippBorder)
            return CV_HAL_ERROR_NOT_IMPLEMENTED;

        CV_INSTRUMENT_FUN_IPP(::ipp::iwiFilterBox, iwSrc, iwDst, iwKSize, ::ipp::IwDefault(), ippBorder);
    }
    catch (const ::ipp::IwException &)
    {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    return CV_HAL_ERROR_OK;
}

#endif // defined(HAVE_IPP_IW) && !DISABLE_IPP_BOX_FILTER
