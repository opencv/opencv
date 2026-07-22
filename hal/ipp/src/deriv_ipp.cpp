// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
// Copyright (C) 2026, Intel Corporation, all rights reserved.

#include "ipp_hal_imgproc.hpp"
#include "precomp_ipp.hpp"

#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>

#if IPP_VERSION_X100 >= 810

#ifdef HAVE_IPP_IW
#include "iw++/iw.hpp"

static inline IppiMaskSize ippiGetMaskSize(int kx, int ky)
{
    return (kx == 1 && ky == 3) ? ippMskSize1x3 :
           (kx == 1 && ky == 5) ? ippMskSize1x5 :
           (kx == 3 && ky == 1) ? ippMskSize3x1 :
           (kx == 3 && ky == 3) ? ippMskSize3x3 :
           (kx == 5 && ky == 1) ? ippMskSize5x1 :
           (kx == 5 && ky == 5) ? ippMskSize5x5 :
           (IppiMaskSize)-1;
}

static inline IwiDerivativeType ippiGetDerivType(int dx, int dy, bool nvert)
{
    return (dx == 1 && dy == 0) ? ((nvert)?iwiDerivNVerFirst:iwiDerivVerFirst) :
           (dx == 0 && dy == 1) ? iwiDerivHorFirst :
           (dx == 2 && dy == 0) ? iwiDerivVerSecond :
           (dx == 0 && dy == 2) ? iwiDerivHorSecond :
           (IwiDerivativeType)-1;
}

// Build an IwiImage from a raw buffer, encoding the in-memory border (the OpenCV
// margins around the ROI) so that BORDER_*-with-physical-pixels works as in core.
// TODO: promote to precomp_ipp.hpp and unify with the raw-pointer ippiGetImage in
//       transforms_ipp.cpp once more filter HALs share it
static inline ::ipp::IwiImage ippiGetImage(int depth, int channels, const uchar* data, size_t step,
                                           int width, int height,
                                           int margin_left, int margin_top, int margin_right, int margin_bottom)
{
    ::ipp::IwiImage image;
    ::ipp::IwiBorderSize inMemBorder((IwSize)margin_left, (IwSize)margin_top, (IwSize)margin_right, (IwSize)margin_bottom);
    image.Init({width, height}, ippiGetDataType(depth), channels, inMemBorder, (void*)data, step);
    return image;
}

// Translate the OpenCV border type into an IPP border, accounting for in-memory pixels.
// Returns (IppiBorderType)0 on unsupported configuration.
// TODO: promote to precomp_ipp.hpp once shared by other filter HALs (box/gaussian/bilateral/
//       sepFilter etc.). Note the shared ippiGetBorderType there does not map BORDER_REFLECT_101;
//       widening it would also affect warp_ipp.cpp, so keep this filter-specific mapping separate
//       (or add a dedicated filter border helper) when consolidating.
static inline IppiBorderType ippiGetBorder(::ipp::IwiImage &image, int ocvBorderType, ::ipp::IwiBorderSize &borderSize)
{
    int            inMemFlags   = 0;
    int            borderTypeNI = ocvBorderType & ~cv::BORDER_ISOLATED;
    IppiBorderType border       = borderTypeNI == cv::BORDER_CONSTANT    ? ippBorderConst  :
                                  borderTypeNI == cv::BORDER_TRANSPARENT ? ippBorderTransp :
                                  borderTypeNI == cv::BORDER_REPLICATE   ? ippBorderRepl   :
                                  borderTypeNI == cv::BORDER_REFLECT_101 ? ippBorderMirror :
                                  (IppiBorderType)-1;
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

    return (IppiBorderType)(border|inMemFlags);
}

// Shared worker for Sobel (useScharr == false) and Scharr (useScharr == true).
static int ipp_Deriv(const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step,
                     int width, int height, int src_depth, int dst_depth, int cn,
                     int margin_left, int margin_top, int margin_right, int margin_bottom,
                     int dx, int dy, int ksize, double scale, double delta, int borderType,
                     bool useScharr)
{
    IppDataType srcType  = ippiGetDataType(src_depth);
    IppDataType dstType  = ippiGetDataType(dst_depth);
    bool        useScale = false;

    if(cn < 1 || cn > 4)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if(src_depth < 0 || src_depth >= CV_DEPTH_MAX || dst_depth < 0 || dst_depth >= CV_DEPTH_MAX)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    // Supported (source depth -> destination depth) x channels combinations.
    // Index order: [src_depth][dst_depth][channel-1]; depths are 8U,8S,16U,16S,32S,32F,64F.
    // IPP iwiFilterSobel/iwiFilterScharr provide only single-channel kernels for
    // 8u->16s, 16s->16s and 32f->32f; 8u->8u is realized as an 8u->16s filter plus a 16s->8u scale.
    // 8u->32f and 16s->32f are intentionally left disabled: IPP would need an extra full-image
    // conversion pass there and is slower than OpenCV's fused sepFilter2D.
                                                  /*               dst: 8U 8S 16U 16S 32S 32F 64F */
#if defined(IPP_CALLS_ENFORCED)
    const char impl[CV_DEPTH_MAX][CV_DEPTH_MAX][4] = {
        /* src 8U  */ {{1,0,0,0},{0,0,0,0},{0,0,0,0},{1,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}},
        /* src 8S  */ {{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}},
        /* src 16U */ {{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}},
        /* src 16S */ {{0,0,0,0},{0,0,0,0},{0,0,0,0},{1,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}},
        /* src 32S */ {{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}},
        /* src 32F */ {{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{1,0,0,0},{0,0,0,0}},
        /* src 64F */ {{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}}};
#else
    const char impl[CV_DEPTH_MAX][CV_DEPTH_MAX][4] = {
        /* src 8U  */ {{1,0,0,0},{0,0,0,0},{0,0,0,0},{1,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}},
        /* src 8S  */ {{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}},
        /* src 16U */ {{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}},
        /* src 16S */ {{0,0,0,0},{0,0,0,0},{0,0,0,0},{1,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}},
        /* src 32S */ {{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}},
        /* src 32F */ {{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{1,0,0,0},{0,0,0,0}},
        /* src 64F */ {{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}}};
#endif // IPP_CALLS_ENFORCED
    if(impl[src_depth][dst_depth][cn-1] == 0)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    if(fabs(delta) > FLT_EPSILON || fabs(scale-1) > FLT_EPSILON)
        useScale = true;

    // cv::Sobel accepts ksize == FILTER_SCHARR (-1, i.e. ksize <= 0) which selects the 3x3
    // Scharr derivative, so the Sobel entry point may legitimately request a Scharr filter.
    if(ksize <= 0)
    {
        ksize     = 3;
        useScharr = true;
    }

    IppiMaskSize maskSize = ippiGetMaskSize(ksize, ksize);
    if((int)maskSize < 0)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

#if IPP_VERSION_X100 <= 201703
    // Bug with mirror wrap
    if(borderType == cv::BORDER_REFLECT_101 && (ksize/2+1 > width || ksize/2+1 > height))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
#endif

    IwiDerivativeType derivType = ippiGetDerivType(dx, dy, (useScharr)?false:true);
    if((int)derivType < 0)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    try
    {
        ::ipp::IwiImage iwSrc      = ippiGetImage(src_depth, cn, src_data, src_step, width, height,
                                                  margin_left, margin_top, margin_right, margin_bottom);
        ::ipp::IwiImage iwDst      = ippiGetImage(dst_depth, cn, dst_data, dst_step, width, height,
                                                  0, 0, 0, 0);
        ::ipp::IwiImage iwSrcProc  = iwSrc;
        ::ipp::IwiImage iwDstProc  = iwDst;
        ::ipp::IwiBorderSize  borderSize(maskSize);
        ::ipp::IwiBorderType  ippBorder(ippiGetBorder(iwSrc, borderType, borderSize));
        if(!ippBorder)
            return CV_HAL_ERROR_NOT_IMPLEMENTED;

        // IPP needs an extra iwiScale pass for 32f output with scale/delta, slower than OpenCV's fused approach
        if(useScale && dstType == ipp32f)
            return CV_HAL_ERROR_NOT_IMPLEMENTED;

        if(srcType == ipp8u && dstType == ipp8u)
        {
            iwDstProc.Alloc(iwDst.m_size, ipp16s, cn);
            useScale = true;
        }

        if(useScharr)
            CV_INSTRUMENT_FUN_IPP(::ipp::iwiFilterScharr, iwSrcProc, iwDstProc, derivType, maskSize, ::ipp::IwDefault(), ippBorder);
        else
            CV_INSTRUMENT_FUN_IPP(::ipp::iwiFilterSobel, iwSrcProc, iwDstProc, derivType, maskSize, ::ipp::IwDefault(), ippBorder);

        if(useScale)
            CV_INSTRUMENT_FUN_IPP(::ipp::iwiScale, iwDstProc, iwDst, scale, delta, ::ipp::IwiScaleParams(ippAlgHintFast));
    }
    catch (const ::ipp::IwException &)
    {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    return CV_HAL_ERROR_OK;
}

int ipp_hal_sobel(const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step,
                  int width, int height, int src_depth, int dst_depth, int cn,
                  int margin_left, int margin_top, int margin_right, int margin_bottom,
                  int dx, int dy, int ksize, double scale, double delta, int border_type)
{
    CV_HAL_CHECK_USE_IPP();
    return ipp_Deriv(src_data, src_step, dst_data, dst_step, width, height, src_depth, dst_depth, cn,
                     margin_left, margin_top, margin_right, margin_bottom,
                     dx, dy, ksize, scale, delta, border_type, false);
}

int ipp_hal_scharr(const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step,
                   int width, int height, int src_depth, int dst_depth, int cn,
                   int margin_left, int margin_top, int margin_right, int margin_bottom,
                   int dx, int dy, double scale, double delta, int border_type)
{
    CV_HAL_CHECK_USE_IPP();
    return ipp_Deriv(src_data, src_step, dst_data, dst_step, width, height, src_depth, dst_depth, cn,
                     margin_left, margin_top, margin_right, margin_bottom,
                     dx, dy, 0, scale, delta, border_type, true);
}

#endif // HAVE_IPP_IW

#endif // IPP_VERSION_X100 >= 810
