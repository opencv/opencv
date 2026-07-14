// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "ipp_hal_imgproc.hpp"

#include <opencv2/core.hpp>
#include "precomp_ipp.hpp"

#include <cmath>
#include <cfloat>

#if defined(HAVE_IPP_IW) && !DISABLE_IPP_FILTER2D

// Copied from core/private.hpp (gated by HAVE_IPP, which the plugin lacks).
// filterGetIppBorderType: distinct name; adds the BORDER_REFLECT_101 -> ippBorderMirror
// case that precomp_ipp.hpp's ippiGetBorderType omits.
static inline IppiBorderType filterGetIppBorderType(int borderTypeNI)
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
    IppiBorderType border     = filterGetIppBorderType(ocvBorderType & ~cv::BORDER_ISOLATED);
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

int ipp_hal_filter2D(const uchar * src_data, size_t src_step, int src_type,
                     uchar * dst_data, size_t dst_step, int dst_type,
                     int width, int height, int full_width, int full_height,
                     int offset_x, int offset_y,
                     const uchar * kernel_data, size_t kernel_step, int kernel_type,
                     int kernel_width, int kernel_height,
                     int anchor_x, int anchor_y, double delta, int borderType,
                     bool isSubmatrix, bool allowInplace)
{
    CV_HAL_CHECK_USE_IPP();

    IppDataType type     = ippiGetDataType(CV_MAT_DEPTH(src_type));
    int         channels = CV_MAT_CN(src_type);

    CV_UNUSED(isSubmatrix);
    CV_UNUSED(allowInplace);

#if IPP_VERSION_X100 >= 201700 && IPP_VERSION_X100 <= 201702 // IPP bug with 1x1 kernel
    if(kernel_width == 1 && kernel_height == 1)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
#endif

#if IPP_DISABLE_FILTER2D_BIG_MASK
    // Too big difference compared to OpenCV FFT-based convolution
    if(kernel_type == CV_32FC1 && (type == ipp16s || type == ipp16u) && (kernel_width > 7 || kernel_height > 7))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    // Poor optimization for big kernels
    if(kernel_width > 7 || kernel_height > 7)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
#endif

    if(src_data == dst_data)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    if(src_type != dst_type)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    if(kernel_type != CV_16SC1 && kernel_type != CV_32FC1)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    // TODO: Implement offset for 8u, 16u
    if(std::fabs(delta) >= DBL_EPSILON)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    if(!ippiCheckAnchor(anchor_x, anchor_y, kernel_width, kernel_height))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    try
    {
        ::ipp::IwiSize          iwSize(width, height);
        ::ipp::IwiSize          kernelSize(kernel_width, kernel_height);
        ::ipp::IwiImage         iwKernel(ippiSize(kernel_width, kernel_height), ippiGetDataType(CV_MAT_DEPTH(kernel_type)), CV_MAT_CN(kernel_type), 0, (void*)kernel_data, kernel_step);
        ::ipp::IwiImage         iwSrc(iwSize, type, channels, ::ipp::IwiBorderSize(offset_x, offset_y, full_width-offset_x-width, full_height-offset_y-height), (void*)src_data, src_step);
        ::ipp::IwiImage         iwDst(iwSize, type, channels, ::ipp::IwiBorderSize(offset_x, offset_y, full_width-offset_x-width, full_height-offset_y-height), (void*)dst_data, dst_step);

        ::ipp::IwiBorderSize    iwBorderSize = ::ipp::iwiSizeToBorderSize(kernelSize);
        ::ipp::IwiBorderType    iwBorderType = ippiGetBorder(iwSrc, borderType, iwBorderSize);
        if(!iwBorderType)
            return CV_HAL_ERROR_NOT_IMPLEMENTED;

        CV_INSTRUMENT_FUN_IPP(::ipp::iwiFilter, iwSrc, iwDst, iwKernel, ::ipp::IwiFilterParams(1, 0, ippAlgHintNone, ippRndFinancial), iwBorderType);
    }
    catch (const ::ipp::IwException &)
    {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    return CV_HAL_ERROR_OK;
}

#endif // defined(HAVE_IPP_IW) && !DISABLE_IPP_FILTER2D
