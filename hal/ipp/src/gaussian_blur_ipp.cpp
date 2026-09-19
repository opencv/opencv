// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "ipp_hal_imgproc.hpp"

#include <opencv2/core.hpp>
#include "precomp_ipp.hpp"

#include <cfloat>

#if defined(HAVE_IPP_IW) && defined(ENABLE_IPP_GAUSSIAN_BLUR)

#define IPP_DISABLE_GAUSSIAN_BLUR_LARGE_KERNELS_1TH 1
#define IPP_DISABLE_GAUSSIAN_BLUR_16SC4_1TH 1
#define IPP_DISABLE_GAUSSIAN_BLUR_32FC4_1TH 1

// IW 2017u2 has bug which doesn't allow use of partial inMem with tiling
#if IPP_VERSION_X100 < 201900
#define IPP_GAUSSIANBLUR_PARALLEL 0
#else
#define IPP_GAUSSIANBLUR_PARALLEL 1
#endif

// Copied from core/private.hpp (gated by HAVE_IPP, which the plugin lacks).
static inline IppiBorderType gaussianGetIppBorderType(int borderTypeNI)
{
    return borderTypeNI == cv::BORDER_CONSTANT    ? ippBorderConst  :
           borderTypeNI == cv::BORDER_TRANSPARENT ? ippBorderTransp :
           borderTypeNI == cv::BORDER_REPLICATE   ? ippBorderRepl   :
           borderTypeNI == cv::BORDER_REFLECT_101 ? ippBorderMirror :
           (IppiBorderType)-1;
}

static inline IppiBorderType ippiGetBorder(::ipp::IwiImage &image, int ocvBorderType, ::ipp::IwiBorderSize &borderSize)
{
    int            inMemFlags = 0;
    IppiBorderType border     = gaussianGetIppBorderType(ocvBorderType & ~cv::BORDER_ISOLATED);
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

class ipp_gaussianBlurParallel: public cv::ParallelLoopBody
{
public:
    ipp_gaussianBlurParallel(::ipp::IwiImage &src, ::ipp::IwiImage &dst, int kernelSize, float sigma, ::ipp::IwiBorderType &border, bool *pOk):
        m_src(src), m_dst(dst), m_kernelSize(kernelSize), m_sigma(sigma), m_border(border), m_pOk(pOk) {
        *m_pOk = true;
    }
    ~ipp_gaussianBlurParallel()
    {
    }

    virtual void operator() (const cv::Range& range) const CV_OVERRIDE
    {
        if(!*m_pOk)
            return;

        try
        {
            ::ipp::IwiTile tile = ::ipp::IwiRoi(0, range.start, m_dst.m_size.width, range.end - range.start);
            CV_INSTRUMENT_FUN_IPP(::ipp::iwiFilterGaussian, m_src, m_dst, m_kernelSize, m_sigma, ::ipp::IwDefault(), m_border, tile);
        }
        catch(const ::ipp::IwException &)
        {
            *m_pOk = false;
            return;
        }
    }
private:
    ::ipp::IwiImage &m_src;
    ::ipp::IwiImage &m_dst;

    int m_kernelSize;
    float m_sigma;
    ::ipp::IwiBorderType &m_border;

    volatile bool *m_pOk;
    const ipp_gaussianBlurParallel& operator= (const ipp_gaussianBlurParallel&);
};

int ipp_hal_gaussianBlur(const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step,
                         int width, int height, int depth, int cn,
                         size_t margin_left, size_t margin_top, size_t margin_right, size_t margin_bottom,
                         size_t ksize_width, size_t ksize_height, double sigmaX, double sigmaY, int border_type)
{
    CV_HAL_CHECK_USE_IPP();

#if IPP_VERSION_X100 < 201800 && ((defined _MSC_VER && defined _M_IX86) || (defined __GNUC__ && defined __i386__))
    CV_UNUSED(src_data); CV_UNUSED(src_step); CV_UNUSED(dst_data); CV_UNUSED(dst_step);
    CV_UNUSED(width); CV_UNUSED(height); CV_UNUSED(depth); CV_UNUSED(cn);
    CV_UNUSED(margin_left); CV_UNUSED(margin_top); CV_UNUSED(margin_right); CV_UNUSED(margin_bottom);
    CV_UNUSED(ksize_width); CV_UNUSED(ksize_height); CV_UNUSED(sigmaX); CV_UNUSED(sigmaY); CV_UNUSED(border_type);
    return CV_HAL_ERROR_NOT_IMPLEMENTED; // bug on ia32
#else
    if(sigmaX != sigmaY)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    if(sigmaX < FLT_EPSILON)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    if(ksize_width != ksize_height)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    IppDataType ippType = ippiGetDataType(depth);
    if(ippType == (IppDataType)-1)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    const int ks = (int)ksize_width; // == ksize_height

    try
    {
        // raw-pointer equivalent of ippiGetImage: margins are the in-memory border
        ::ipp::IwiBorderSize inMemBorder;
        inMemBorder.left   = (IwSize)margin_left;
        inMemBorder.top    = (IwSize)margin_top;
        inMemBorder.right  = (IwSize)margin_right;
        inMemBorder.bottom = (IwSize)margin_bottom;

        ::ipp::IwiImage iwSrc, iwDst;
        iwSrc.Init(IwiSize{width, height}, ippType, cn, inMemBorder, (void*)src_data, IwSize(src_step));
        iwDst.Init(IwiSize{width, height}, ippType, cn, ::ipp::IwiBorderSize(), dst_data, IwSize(dst_step));

        ::ipp::IwiSize       iwKSize{ks, ks};
        ::ipp::IwiBorderSize borderSize(iwKSize);
        ::ipp::IwiBorderType ippBorder(ippiGetBorder(iwSrc, border_type, borderSize));
        if(!ippBorder)
            return CV_HAL_ERROR_NOT_IMPLEMENTED;

        const int threads = ippiSuggestThreadsNum(iwDst, 2);

        if(IPP_DISABLE_GAUSSIAN_BLUR_LARGE_KERNELS_1TH && (threads == 1 && ks > 25))
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        if(IPP_DISABLE_GAUSSIAN_BLUR_16SC4_1TH && (threads == 1 && depth == CV_16S && cn == 4))
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        if(IPP_DISABLE_GAUSSIAN_BLUR_32FC4_1TH && (threads == 1 && depth == CV_32F && cn == 4))
            return CV_HAL_ERROR_NOT_IMPLEMENTED;

        if(IPP_GAUSSIANBLUR_PARALLEL && threads > 1 && iwSrc.m_size.height/(threads * 4) >= ks/2)
        {
            bool ok;
            ipp_gaussianBlurParallel invoker(iwSrc, iwDst, ks, (float)sigmaX, ippBorder, &ok);

            if(!ok)
                return CV_HAL_ERROR_NOT_IMPLEMENTED;
            const cv::Range range(0, (int)iwDst.m_size.height);
            cv::parallel_for_(range, invoker, threads*4);

            if(!ok)
                return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }
        else
        {
            CV_INSTRUMENT_FUN_IPP(::ipp::iwiFilterGaussian, iwSrc, iwDst, ks, sigmaX, ::ipp::IwDefault(), ippBorder);
        }
    }
    catch (const ::ipp::IwException &)
    {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    return CV_HAL_ERROR_OK;
#endif
}

#endif // defined(HAVE_IPP_IW) && defined(ENABLE_IPP_GAUSSIAN_BLUR)
