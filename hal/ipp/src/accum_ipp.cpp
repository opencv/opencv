// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "ipp_hal_imgproc.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#if IPP_VERSION_X100 >= 700

int ipp_hal_accumulate(const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step,
                       const uchar* mask_data, size_t mask_step,
                       int width, int height, int src_type, int dst_type)
{
    CV_HAL_CHECK_USE_IPP();

    int sdepth = CV_MAT_DEPTH(src_type), scn = CV_MAT_CN(src_type);
    int ddepth = CV_MAT_DEPTH(dst_type);

    typedef IppStatus (CV_STDCALL * IppiAdd)(const void* pSrc, int srcStep, Ipp32f* pSrcDst, int srcdstStep, IppiSize roiSize);
    typedef IppStatus (CV_STDCALL * IppiAddMask)(const void* pSrc, int srcStep, const Ipp8u* pMask, int maskStep, Ipp32f* pSrcDst,
                                                 int srcDstStep, IppiSize roiSize);
    IppiAdd ippiAdd_I = 0;
    IppiAddMask ippiAdd_IM = 0;

    if (!mask_data)
    {
        CV_SUPPRESS_DEPRECATED_START
        ippiAdd_I = sdepth == CV_8U && ddepth == CV_32F ? (IppiAdd)ippiAdd_8u32f_C1IR :
            sdepth == CV_16U && ddepth == CV_32F ? (IppiAdd)ippiAdd_16u32f_C1IR :
            sdepth == CV_32F && ddepth == CV_32F ? (IppiAdd)ippiAdd_32f_C1IR : 0;
        CV_SUPPRESS_DEPRECATED_END
    }
    else if (scn == 1)
    {
        ippiAdd_IM = sdepth == CV_8U && ddepth == CV_32F ? (IppiAddMask)ippiAdd_8u32f_C1IMR :
            sdepth == CV_16U && ddepth == CV_32F ? (IppiAddMask)ippiAdd_16u32f_C1IMR :
            sdepth == CV_32F && ddepth == CV_32F ? (IppiAddMask)ippiAdd_32f_C1IMR : 0;
    }

    if (!ippiAdd_I && !ippiAdd_IM)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    width *= scn;
    IppiSize roi = { width, height };
    IppStatus status = ippStsErr;

    if (ippiAdd_I)
        status = CV_INSTRUMENT_FUN_IPP(ippiAdd_I, src_data, (int)src_step, (Ipp32f*)dst_data, (int)dst_step, roi);
    else
        status = CV_INSTRUMENT_FUN_IPP(ippiAdd_IM, src_data, (int)src_step, mask_data, (int)mask_step,
                                       (Ipp32f*)dst_data, (int)dst_step, roi);

    return status >= 0 ? CV_HAL_ERROR_OK : CV_HAL_ERROR_NOT_IMPLEMENTED;
}

int ipp_hal_accumulateSquare(const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step,
                             const uchar* mask_data, size_t mask_step,
                             int width, int height, int src_type, int dst_type)
{
    CV_HAL_CHECK_USE_IPP();

    int sdepth = CV_MAT_DEPTH(src_type), scn = CV_MAT_CN(src_type);
    int ddepth = CV_MAT_DEPTH(dst_type);

    typedef IppStatus (CV_STDCALL * IppiAddSquare)(const void* pSrc, int srcStep, Ipp32f* pSrcDst, int srcdstStep, IppiSize roiSize);
    typedef IppStatus (CV_STDCALL * IppiAddSquareMask)(const void* pSrc, int srcStep, const Ipp8u* pMask, int maskStep, Ipp32f* pSrcDst,
                                                       int srcDstStep, IppiSize roiSize);
    IppiAddSquare ippiAddSquare_I = 0;
    IppiAddSquareMask ippiAddSquare_IM = 0;

    if (!mask_data)
    {
        ippiAddSquare_I = sdepth == CV_8U && ddepth == CV_32F ? (IppiAddSquare)ippiAddSquare_8u32f_C1IR :
            sdepth == CV_16U && ddepth == CV_32F ? (IppiAddSquare)ippiAddSquare_16u32f_C1IR :
            sdepth == CV_32F && ddepth == CV_32F ? (IppiAddSquare)ippiAddSquare_32f_C1IR : 0;
    }
    else if (scn == 1)
    {
        ippiAddSquare_IM = sdepth == CV_8U && ddepth == CV_32F ? (IppiAddSquareMask)ippiAddSquare_8u32f_C1IMR :
            sdepth == CV_16U && ddepth == CV_32F ? (IppiAddSquareMask)ippiAddSquare_16u32f_C1IMR :
            sdepth == CV_32F && ddepth == CV_32F ? (IppiAddSquareMask)ippiAddSquare_32f_C1IMR : 0;
    }

    if (!ippiAddSquare_I && !ippiAddSquare_IM)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    width *= scn;
    IppiSize roi = { width, height };
    IppStatus status = ippStsErr;

    if (ippiAddSquare_I)
        status = CV_INSTRUMENT_FUN_IPP(ippiAddSquare_I, src_data, (int)src_step, (Ipp32f*)dst_data, (int)dst_step, roi);
    else
        status = CV_INSTRUMENT_FUN_IPP(ippiAddSquare_IM, src_data, (int)src_step, mask_data, (int)mask_step,
                                       (Ipp32f*)dst_data, (int)dst_step, roi);

    return status >= 0 ? CV_HAL_ERROR_OK : CV_HAL_ERROR_NOT_IMPLEMENTED;
}

int ipp_hal_accumulateProduct(const uchar* src1_data, size_t src1_step,
                              const uchar* src2_data, size_t src2_step,
                              uchar* dst_data, size_t dst_step,
                              const uchar* mask_data, size_t mask_step,
                              int width, int height, int src_type, int dst_type)
{
    CV_HAL_CHECK_USE_IPP();

    int sdepth = CV_MAT_DEPTH(src_type), scn = CV_MAT_CN(src_type);
    int ddepth = CV_MAT_DEPTH(dst_type);

    typedef IppStatus (CV_STDCALL * IppiAddProduct)(const void* pSrc1, int src1Step, const void* pSrc2,
                                                    int src2Step, Ipp32f* pSrcDst, int srcDstStep, IppiSize roiSize);
    typedef IppStatus (CV_STDCALL * IppiAddProductMask)(const void* pSrc1, int src1Step, const void* pSrc2, int src2Step,
                                                        const Ipp8u* pMask, int maskStep, Ipp32f* pSrcDst, int srcDstStep, IppiSize roiSize);
    IppiAddProduct ippiAddProduct_I = 0;
    IppiAddProductMask ippiAddProduct_IM = 0;

    if (!mask_data)
    {
        ippiAddProduct_I = sdepth == CV_8U && ddepth == CV_32F ? (IppiAddProduct)ippiAddProduct_8u32f_C1IR :
            sdepth == CV_16U && ddepth == CV_32F ? (IppiAddProduct)ippiAddProduct_16u32f_C1IR :
            sdepth == CV_32F && ddepth == CV_32F ? (IppiAddProduct)ippiAddProduct_32f_C1IR : 0;
    }
    else if (scn == 1)
    {
        ippiAddProduct_IM = sdepth == CV_8U && ddepth == CV_32F ? (IppiAddProductMask)ippiAddProduct_8u32f_C1IMR :
            sdepth == CV_16U && ddepth == CV_32F ? (IppiAddProductMask)ippiAddProduct_16u32f_C1IMR :
            sdepth == CV_32F && ddepth == CV_32F ? (IppiAddProductMask)ippiAddProduct_32f_C1IMR : 0;
    }

    if (!ippiAddProduct_I && !ippiAddProduct_IM)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    width *= scn;
    IppiSize roi = { width, height };
    IppStatus status = ippStsErr;

    if (ippiAddProduct_I)
        status = CV_INSTRUMENT_FUN_IPP(ippiAddProduct_I, src1_data, (int)src1_step, src2_data, (int)src2_step,
                                       (Ipp32f*)dst_data, (int)dst_step, roi);
    else
        status = CV_INSTRUMENT_FUN_IPP(ippiAddProduct_IM, src1_data, (int)src1_step, src2_data, (int)src2_step,
                                       mask_data, (int)mask_step, (Ipp32f*)dst_data, (int)dst_step, roi);

    return status >= 0 ? CV_HAL_ERROR_OK : CV_HAL_ERROR_NOT_IMPLEMENTED;
}

int ipp_hal_accumulateWeighted(const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step,
                               const uchar* mask_data, size_t mask_step,
                               int width, int height, int src_type, int dst_type, double alpha)
{
    CV_HAL_CHECK_USE_IPP();

    int sdepth = CV_MAT_DEPTH(src_type), scn = CV_MAT_CN(src_type);
    int ddepth = CV_MAT_DEPTH(dst_type);

    typedef IppStatus (CV_STDCALL * IppiAddWeighted)(const void* pSrc, int srcStep, Ipp32f* pSrcDst, int srcdstStep,
                                                     IppiSize roiSize, Ipp32f alpha);
    typedef IppStatus (CV_STDCALL * IppiAddWeightedMask)(const void* pSrc, int srcStep, const Ipp8u* pMask,
                                                         int maskStep, Ipp32f* pSrcDst,
                                                         int srcDstStep, IppiSize roiSize, Ipp32f alpha);
    IppiAddWeighted ippiAddWeighted_I = 0;
    IppiAddWeightedMask ippiAddWeighted_IM = 0;

    if (!mask_data)
    {
        ippiAddWeighted_I = sdepth == CV_8U && ddepth == CV_32F ? (IppiAddWeighted)ippiAddWeighted_8u32f_C1IR :
            sdepth == CV_16U && ddepth == CV_32F ? (IppiAddWeighted)ippiAddWeighted_16u32f_C1IR :
            sdepth == CV_32F && ddepth == CV_32F ? (IppiAddWeighted)ippiAddWeighted_32f_C1IR : 0;
    }
    else if (scn == 1)
    {
        ippiAddWeighted_IM = sdepth == CV_8U && ddepth == CV_32F ? (IppiAddWeightedMask)ippiAddWeighted_8u32f_C1IMR :
            sdepth == CV_16U && ddepth == CV_32F ? (IppiAddWeightedMask)ippiAddWeighted_16u32f_C1IMR :
            sdepth == CV_32F && ddepth == CV_32F ? (IppiAddWeightedMask)ippiAddWeighted_32f_C1IMR : 0;
    }

    if (!ippiAddWeighted_I && !ippiAddWeighted_IM)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    width *= scn;
    IppiSize roi = { width, height };
    IppStatus status = ippStsErr;

    if (ippiAddWeighted_I)
        status = CV_INSTRUMENT_FUN_IPP(ippiAddWeighted_I, src_data, (int)src_step, (Ipp32f*)dst_data, (int)dst_step, roi, (Ipp32f)alpha);
    else
        status = CV_INSTRUMENT_FUN_IPP(ippiAddWeighted_IM, src_data, (int)src_step, mask_data, (int)mask_step,
                                       (Ipp32f*)dst_data, (int)dst_step, roi, (Ipp32f)alpha);

    return status >= 0 ? CV_HAL_ERROR_OK : CV_HAL_ERROR_NOT_IMPLEMENTED;
}

#endif // IPP_VERSION_X100 >= 700
