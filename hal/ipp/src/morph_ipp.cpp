// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "ipp_hal_imgproc.hpp"
#include "precomp_ipp.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#if defined(HAVE_IPP_IW) && !DISABLE_IPP_MORPH

namespace cv { namespace ipp { unsigned long long getIppTopFeatures(); } }

static inline IwiMorphologyType ippiGetMorphologyType(int morphOp)
{
    return morphOp == cv::MORPH_ERODE ? iwiMorphErode :
        morphOp == cv::MORPH_DILATE   ? iwiMorphDilate :
        morphOp == cv::MORPH_OPEN     ? iwiMorphOpen :
        morphOp == cv::MORPH_CLOSE    ? iwiMorphClose :
        morphOp == cv::MORPH_GRADIENT ? iwiMorphGradient :
        morphOp == cv::MORPH_TOPHAT   ? iwiMorphTophat :
        morphOp == cv::MORPH_BLACKHAT ? iwiMorphBlackhat : (IwiMorphologyType)-1;
}

static inline bool ippiCheckAnchor(int x, int y, int kernelWidth, int kernelHeight)
{
    return (x == (kernelWidth - 1)/2 && y == (kernelHeight - 1)/2);
}

// Copied from core/private.hpp (gated by HAVE_IPP, which the plugin lacks).
static inline IppiBorderType morphGetIppBorderType(int borderTypeNI)
{
    return borderTypeNI == cv::BORDER_CONSTANT    ? ippBorderConst  :
           borderTypeNI == cv::BORDER_TRANSPARENT ? ippBorderTransp :
           borderTypeNI == cv::BORDER_REPLICATE   ? ippBorderRepl   :
           borderTypeNI == cv::BORDER_REFLECT_101 ? ippBorderMirror :
           (IppiBorderType)-1;
}

static inline IppiBorderType ippiGetBorder(::ipp::IwiImage &image, int ocvBorderType, ::ipp::IwiBorderSize &borderSize)
{
    int            inMemFlags   = 0;
    IppiBorderType border       = morphGetIppBorderType(ocvBorderType & ~cv::BORDER_ISOLATED);
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

int ipp_hal_morph_stateless(int operation, const uchar * src_data, size_t src_step, int src_type,
                            uchar * dst_data, size_t dst_step, int dst_type,
                            int width, int height, int src_full_width, int src_full_height, int src_roi_x, int src_roi_y,
                            int dst_full_width, int dst_full_height, int dst_roi_x, int dst_roi_y,
                            const uchar * kernel_data, size_t kernel_step, int kernel_type, int kernel_width, int kernel_height,
                            int anchor_x, int anchor_y, int borderType, const double borderValue[4],
                            int iterations, bool allowSubmatrix, bool allowInplace)
{
    CV_HAL_CHECK_USE_IPP();

    const int op         = operation;
    const int roi_width  = src_full_width,  roi_height  = src_full_height,  roi_x  = src_roi_x,  roi_y  = src_roi_y;
    const int roi_width2 = dst_full_width,  roi_height2 = dst_full_height,  roi_x2 = dst_roi_x,  roi_y2 = dst_roi_y;
    CV_UNUSED(allowSubmatrix); CV_UNUSED(allowInplace);

#if IPP_VERSION_X100 < 201800
    // Problem with SSE42 optimizations performance
    if(cv::ipp::getIppTopFeatures() == ippCPUID_SSE42)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    // Different mask flipping
    if(op == cv::MORPH_GRADIENT)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    // Integer overflow bug
    if(src_step >= IPP_MAX_32S ||
       src_step*height >= IPP_MAX_32S)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
#endif

#if IPP_VERSION_X100 < 201801
    // Problem with AVX512 optimizations performance
    if(cv::ipp::getIppTopFeatures()&ippCPUID_AVX512F)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    // Multiple iterations on small mask is not effective in current integration
    // Inplace imitation for 3x3 kernel is not efficient
    // Advanced morphology for small mask introduces degradations
    if((iterations > 1 || src_data == dst_data || (op != cv::MORPH_ERODE && op != cv::MORPH_DILATE)) && kernel_width*kernel_height < 25)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    // Skip even mask sizes for advanced morphology since they can produce out of spec writes
    if((op != cv::MORPH_ERODE && op != cv::MORPH_DILATE) && (!(kernel_width&1) || !(kernel_height&1)))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
#endif

    ::ipp::IwiBorderSize        iwBorderSize;
    ::ipp::IwiBorderSize        iwBorderSize2;
    ::ipp::IwiBorderType        iwBorderType;
    ::ipp::IwiBorderType        iwBorderType2;
    ::ipp::IwiImage             iwMask;
    ::ipp::IwiImage             iwInter;
    ::ipp::IwiSize              initSize(width, height);
    ::ipp::IwiSize              kernelSize(kernel_width, kernel_height);
    IppDataType                 type        = ippiGetDataType(CV_MAT_DEPTH(src_type));
    int                         channels    = CV_MAT_CN(src_type);
    IwiMorphologyType           morphType   = ippiGetMorphologyType(op);

    if((int)morphType < 0)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    if(iterations > 1 && morphType != iwiMorphErode && morphType != iwiMorphDilate)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    if(src_type != dst_type)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    if(!ippiCheckAnchor(anchor_x, anchor_y, kernel_width, kernel_height))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    try
    {
        ::ipp::IwiImage iwSrc(initSize, type, channels, ::ipp::IwiBorderSize(roi_x, roi_y, roi_width-roi_x-width, roi_height-roi_y-height), (void*)src_data, src_step);
        ::ipp::IwiImage iwDst(initSize, type, channels, ::ipp::IwiBorderSize(roi_x2, roi_y2, roi_width2-roi_x2-width, roi_height2-roi_y2-height), (void*)dst_data, dst_step);

        iwBorderSize = ::ipp::iwiSizeToBorderSize(kernelSize);
        iwBorderType = ippiGetBorder(iwSrc, borderType, iwBorderSize);
        if(!iwBorderType)
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        if(iterations > 1)
        {
            // Check dst border for second and later iterations
            iwBorderSize2 = ::ipp::iwiSizeToBorderSize(kernelSize);
            iwBorderType2 = ippiGetBorder(iwDst, borderType, iwBorderSize2);
            if(!iwBorderType2)
                return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }

        if(morphType != iwiMorphErode && morphType != iwiMorphDilate && morphType != iwiMorphGradient)
        {
            // For now complex morphology support only InMem around all sides. This will be improved later.
            if((iwBorderType&ippBorderInMem) && (iwBorderType&ippBorderInMem) != ippBorderInMem)
                return CV_HAL_ERROR_NOT_IMPLEMENTED;

            if((iwBorderType&ippBorderInMem) == ippBorderInMem)
            {
                iwBorderType &= ~ippBorderInMem;
                iwBorderType &=  ippBorderFirstStageInMem;
            }
        }

        if(iwBorderType.StripFlags() == ippBorderConst)
        {
            if(cv::Vec<double, 4>(borderValue) == cv::morphologyDefaultBorderValue())
                iwBorderType.SetType(ippBorderDefault);
            else
                iwBorderType.m_value = ::ipp::IwValueFloat(borderValue[0], borderValue[1], borderValue[2], borderValue[3]);
        }

        iwMask.Init(ippiSize(kernel_width, kernel_height), ippiGetDataType(CV_MAT_DEPTH(kernel_type)), CV_MAT_CN(kernel_type), 0, (void*)kernel_data, kernel_step);

        ::ipp::IwiImage iwMaskLoc = iwMask;
        if(morphType == iwiMorphDilate)
        {
            iwMaskLoc.Alloc(iwMask.m_size, iwMask.m_dataType, iwMask.m_channels);
            ::ipp::iwiMirror(iwMask, iwMaskLoc, ippAxsBoth);
            iwMask = iwMaskLoc;
        }

        if(iterations > 1)
        {
            // OpenCV uses in mem border from dst for two and more iterations, so we need to keep this border in intermediate image
            iwInter.Alloc(initSize, type, channels, iwBorderSize2);

            ::ipp::IwiImage *pSwap[2] = {&iwInter, &iwDst};
            CV_INSTRUMENT_FUN_IPP(::ipp::iwiFilterMorphology, iwSrc, iwInter, morphType, iwMask, ::ipp::IwDefault(), iwBorderType);

            // Copy border only
            {
                if(iwBorderSize2.top)
                {
                    ::ipp::IwiRoi   borderRoi(-iwBorderSize2.left, -iwBorderSize2.top, iwDst.m_size.width+iwBorderSize2.left+iwBorderSize2.right, iwBorderSize2.top);
                    ::ipp::IwiImage iwInterRoi = iwInter.GetRoiImage(borderRoi);
                    ::ipp::iwiCopy(iwDst.GetRoiImage(borderRoi), iwInterRoi);
                }
                if(iwBorderSize2.bottom)
                {
                    ::ipp::IwiRoi   borderRoi(-iwBorderSize2.left, iwDst.m_size.height, iwDst.m_size.width+iwBorderSize2.left+iwBorderSize2.right, iwBorderSize2.bottom);
                    ::ipp::IwiImage iwInterRoi = iwInter.GetRoiImage(borderRoi);
                    ::ipp::iwiCopy(iwDst.GetRoiImage(borderRoi), iwInterRoi);
                }
                if(iwBorderSize2.left)
                {
                    ::ipp::IwiRoi   borderRoi(-iwBorderSize2.left, 0, iwBorderSize2.left, iwDst.m_size.height);
                    ::ipp::IwiImage iwInterRoi = iwInter.GetRoiImage(borderRoi);
                    ::ipp::iwiCopy(iwDst.GetRoiImage(borderRoi), iwInterRoi);
                }
                if(iwBorderSize2.right)
                {
                    ::ipp::IwiRoi   borderRoi(iwDst.m_size.width, 0, iwBorderSize2.left, iwDst.m_size.height);
                    ::ipp::IwiImage iwInterRoi = iwInter.GetRoiImage(borderRoi);
                    ::ipp::iwiCopy(iwDst.GetRoiImage(borderRoi), iwInterRoi);
                }
            }

            iwBorderType2.SetType(iwBorderType);
            for(int i = 0; i < iterations-1; i++)
                CV_INSTRUMENT_FUN_IPP(::ipp::iwiFilterMorphology, *pSwap[i&0x1], *pSwap[(i+1)&0x1], morphType, iwMask, ::ipp::IwDefault(), iwBorderType2);
            if(iterations&0x1)
                CV_INSTRUMENT_FUN_IPP(::ipp::iwiCopy, iwInter, iwDst);
        }
        else
        {
            if(src_data == dst_data)
            {
                iwInter.Alloc(initSize, type, channels);

                CV_INSTRUMENT_FUN_IPP(::ipp::iwiFilterMorphology, iwSrc, iwInter, morphType, iwMask, ::ipp::IwDefault(), iwBorderType);
                CV_INSTRUMENT_FUN_IPP(::ipp::iwiCopy, iwInter, iwDst);
            }
            else
                CV_INSTRUMENT_FUN_IPP(::ipp::iwiFilterMorphology, iwSrc, iwDst, morphType, iwMask, ::ipp::IwDefault(), iwBorderType);
        }
    }
    catch(const ::ipp::IwException &)
    {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    return CV_HAL_ERROR_OK;
}

#endif // defined(HAVE_IPP_IW) && !DISABLE_IPP_MORPH
