// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "ipp_hal_core.hpp"

#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>

#if IPP_VERSION_X100 >= 700

int ipp_hal_norm(const uchar* src, size_t src_step, const uchar* mask, size_t mask_step,
                 int width, int height, int type, int norm_type, double* result)
{
    if( mask )
    {
        IppiSize sz = { width, height };

        typedef IppStatus (CV_STDCALL* ippiMaskNormFuncC1)(const void *, int, const void *, int, IppiSize, Ipp64f *);
        ippiMaskNormFuncC1 ippiNorm_C1MR =
        norm_type == cv::NORM_INF ?
        (type == CV_8UC1 ? (ippiMaskNormFuncC1)ippiNorm_Inf_8u_C1MR :
        type == CV_16UC1 ? (ippiMaskNormFuncC1)ippiNorm_Inf_16u_C1MR :
        type == CV_32FC1 ? (ippiMaskNormFuncC1)ippiNorm_Inf_32f_C1MR :
        0) :
        norm_type == cv::NORM_L1 ?
        (type == CV_8UC1 ? (ippiMaskNormFuncC1)ippiNorm_L1_8u_C1MR :
        type == CV_16UC1 ? (ippiMaskNormFuncC1)ippiNorm_L1_16u_C1MR :
        type == CV_32FC1 ? (ippiMaskNormFuncC1)ippiNorm_L1_32f_C1MR :
        0) :
        norm_type == cv::NORM_L2 || norm_type == cv::NORM_L2SQR ?
        (type == CV_8UC1 ? (ippiMaskNormFuncC1)ippiNorm_L2_8u_C1MR :
        type == CV_16UC1 ? (ippiMaskNormFuncC1)ippiNorm_L2_16u_C1MR :
        type == CV_32FC1 ? (ippiMaskNormFuncC1)ippiNorm_L2_32f_C1MR :
        0) : 0;
        if( ippiNorm_C1MR )
        {
            Ipp64f norm;
            if( CV_INSTRUMENT_FUN_IPP(ippiNorm_C1MR, src, (int)src_step, mask, (int)mask_step, sz, &norm) >= 0 )
            {
                *result = (norm_type == cv::NORM_L2SQR ? (double)(norm * norm) : (double)norm);
                return CV_HAL_ERROR_OK;
            }
        }
        typedef IppStatus (CV_STDCALL* ippiMaskNormFuncC3)(const void *, int, const void *, int, IppiSize, int, Ipp64f *);
        ippiMaskNormFuncC3 ippiNorm_C3CMR =
        norm_type == cv::NORM_INF ?
        (type == CV_8UC3 ? (ippiMaskNormFuncC3)ippiNorm_Inf_8u_C3CMR :
        type == CV_16UC3 ? (ippiMaskNormFuncC3)ippiNorm_Inf_16u_C3CMR :
        type == CV_32FC3 ? (ippiMaskNormFuncC3)ippiNorm_Inf_32f_C3CMR :
        0) :
        norm_type == cv::NORM_L1 ?
        (type == CV_8UC3 ? (ippiMaskNormFuncC3)ippiNorm_L1_8u_C3CMR :
        type == CV_16UC3 ? (ippiMaskNormFuncC3)ippiNorm_L1_16u_C3CMR :
        type == CV_32FC3 ? (ippiMaskNormFuncC3)ippiNorm_L1_32f_C3CMR :
        0) :
        norm_type == cv::NORM_L2 || norm_type == cv::NORM_L2SQR ?
        (type == CV_8UC3 ? (ippiMaskNormFuncC3)ippiNorm_L2_8u_C3CMR :
        type == CV_16UC3 ? (ippiMaskNormFuncC3)ippiNorm_L2_16u_C3CMR :
        type == CV_32FC3 ? (ippiMaskNormFuncC3)ippiNorm_L2_32f_C3CMR :
        0) : 0;
        if( ippiNorm_C3CMR )
        {
            Ipp64f norm1, norm2, norm3;
            if( CV_INSTRUMENT_FUN_IPP(ippiNorm_C3CMR, src, (int)src_step, mask, (int)mask_step, sz, 1, &norm1) >= 0 &&
                CV_INSTRUMENT_FUN_IPP(ippiNorm_C3CMR, src, (int)src_step, mask, (int)mask_step, sz, 2, &norm2) >= 0 &&
                CV_INSTRUMENT_FUN_IPP(ippiNorm_C3CMR, src, (int)src_step, mask, (int)mask_step, sz, 3, &norm3) >= 0)
            {
                Ipp64f norm =
                norm_type == cv::NORM_INF ? std::max(std::max(norm1, norm2), norm3) :
                norm_type == cv::NORM_L1 ? norm1 + norm2 + norm3 :
                norm_type == cv::NORM_L2 || norm_type == cv::NORM_L2SQR ? std::sqrt(norm1 * norm1 + norm2 * norm2 + norm3 * norm3) :
                0;
                *result = (norm_type == cv::NORM_L2SQR ? (double)(norm * norm) : (double)norm);
                return CV_HAL_ERROR_OK;
            }
        }
    }
    else
    {
        int cn = CV_MAT_CN(type);
        IppiSize sz = { width*cn, height };

        typedef IppStatus (CV_STDCALL* ippiNormFuncHint)(const void *, int, IppiSize, Ipp64f *, IppHintAlgorithm hint);
        typedef IppStatus (CV_STDCALL* ippiNormFuncNoHint)(const void *, int, IppiSize, Ipp64f *);
        ippiNormFuncHint ippiNormHint =
        norm_type == cv::NORM_L1 ?
        (type == CV_32FC1 ? (ippiNormFuncHint)ippiNorm_L1_32f_C1R :
        0) :
        norm_type == cv::NORM_L2 || norm_type == cv::NORM_L2SQR ?
        (type == CV_32FC1 ? (ippiNormFuncHint)ippiNorm_L2_32f_C1R :
        0) : 0;
        ippiNormFuncNoHint ippiNorm =
        norm_type == cv::NORM_INF ?
        (type == CV_8UC1 ? (ippiNormFuncNoHint)ippiNorm_Inf_8u_C1R :
        type == CV_16UC1 ? (ippiNormFuncNoHint)ippiNorm_Inf_16u_C1R :
        type == CV_16SC1 ? (ippiNormFuncNoHint)ippiNorm_Inf_16s_C1R :
        type == CV_32FC1 ? (ippiNormFuncNoHint)ippiNorm_Inf_32f_C1R :
        0) :
        norm_type == cv::NORM_L1 ?
        (type == CV_8UC1 ? (ippiNormFuncNoHint)ippiNorm_L1_8u_C1R :
        type == CV_16UC1 ? (ippiNormFuncNoHint)ippiNorm_L1_16u_C1R :
        type == CV_16SC1 ? (ippiNormFuncNoHint)ippiNorm_L1_16s_C1R :
        0) :
        norm_type == cv::NORM_L2 || norm_type == cv::NORM_L2SQR ?
        (
            #if (!IPP_DISABLE_NORM_8U)
            type == CV_8UC1 ? (ippiNormFuncNoHint)ippiNorm_L2_8u_C1R :
            #endif
            type == CV_16UC1 ? (ippiNormFuncNoHint)ippiNorm_L2_16u_C1R :
            type == CV_16SC1 ? (ippiNormFuncNoHint)ippiNorm_L2_16s_C1R :
            0) : 0;
            if( ippiNormHint || ippiNorm )
            {
                Ipp64f norm;
                IppStatus ret = ippiNormHint ? CV_INSTRUMENT_FUN_IPP(ippiNormHint, src, (int)src_step, sz, &norm, ippAlgHintAccurate) :
                CV_INSTRUMENT_FUN_IPP(ippiNorm, src, (int)src_step, sz, &norm);
                if( ret >= 0 )
                {
                    *result = (norm_type == cv::NORM_L2SQR) ? norm * norm : norm;
                    return CV_HAL_ERROR_OK;
                }
            }
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

int ipp_hal_normDiff(const uchar* src1, size_t src1_step, const uchar* src2, size_t src2_step, const uchar* mask,
                     size_t mask_step, int width, int height, int type, int norm_type, double* result)
{
    if( norm_type & cv::NORM_RELATIVE )
    {
        norm_type &= cv::NORM_TYPE_MASK;

        if( mask )
        {
            IppiSize sz = { width, height };

            typedef IppStatus (CV_STDCALL* ippiMaskNormDiffFuncC1)(const void *, int, const void *, int, const void *, int, IppiSize, Ipp64f *);
            ippiMaskNormDiffFuncC1 ippiNormRel_C1MR =
            norm_type == cv::NORM_INF ?
            (type == CV_8UC1 ? (ippiMaskNormDiffFuncC1)ippiNormRel_Inf_8u_C1MR :
            type == CV_16UC1 ? (ippiMaskNormDiffFuncC1)ippiNormRel_Inf_16u_C1MR :
            type == CV_32FC1 ? (ippiMaskNormDiffFuncC1)ippiNormRel_Inf_32f_C1MR :
            0) :
            norm_type == cv::NORM_L1 ?
            (type == CV_8UC1 ? (ippiMaskNormDiffFuncC1)ippiNormRel_L1_8u_C1MR :
            type == CV_16UC1 ? (ippiMaskNormDiffFuncC1)ippiNormRel_L1_16u_C1MR :
            type == CV_32FC1 ? (ippiMaskNormDiffFuncC1)ippiNormRel_L1_32f_C1MR :
            0) :
            norm_type == cv::NORM_L2 || norm_type == cv::NORM_L2SQR ?
            (type == CV_8UC1 ? (ippiMaskNormDiffFuncC1)ippiNormRel_L2_8u_C1MR :
            type == CV_16UC1 ? (ippiMaskNormDiffFuncC1)ippiNormRel_L2_16u_C1MR :
            type == CV_32FC1 ? (ippiMaskNormDiffFuncC1)ippiNormRel_L2_32f_C1MR :
            0) : 0;
            if( ippiNormRel_C1MR )
            {
                Ipp64f norm;
                if( CV_INSTRUMENT_FUN_IPP(ippiNormRel_C1MR, src1, (int)src1_step, src2, (int)src2_step, mask, (int)mask_step, sz, &norm) >= 0 )
                {
                    *result = (norm_type == cv::NORM_L2SQR ? (double)(norm * norm) : (double)norm);
                    return CV_HAL_ERROR_OK;
                }
            }
        }
        else
        {
            int cn = CV_MAT_CN(type);
            type = CV_MAT_DEPTH(type);
            IppiSize sz = { width*cn, height };

            typedef IppStatus (CV_STDCALL* ippiNormRelFuncHint)(const void *, int, const void *, int, IppiSize, Ipp64f *, IppHintAlgorithm hint);
            typedef IppStatus (CV_STDCALL* ippiNormRelFuncNoHint)(const void *, int, const void *, int, IppiSize, Ipp64f *);
            ippiNormRelFuncHint ippiNormRelHint =
            norm_type == cv::NORM_L1 ?
            (type == CV_32F ? (ippiNormRelFuncHint)ippiNormRel_L1_32f_C1R :
            0) :
            norm_type == cv::NORM_L2 || norm_type == cv::NORM_L2SQR ?
            (type == CV_32F ? (ippiNormRelFuncHint)ippiNormRel_L2_32f_C1R :
            0) : 0;
            ippiNormRelFuncNoHint ippiNormRel =
            norm_type == cv::NORM_INF ?
            (
                #if (!IPP_DISABLE_NORM_8U)
                type == CV_8U ? (ippiNormRelFuncNoHint)ippiNormRel_Inf_8u_C1R :
                #endif
                type == CV_16U ? (ippiNormRelFuncNoHint)ippiNormRel_Inf_16u_C1R :
                type == CV_16S ? (ippiNormRelFuncNoHint)ippiNormRel_Inf_16s_C1R :
                type == CV_32F ? (ippiNormRelFuncNoHint)ippiNormRel_Inf_32f_C1R :
                0) :
                norm_type == cv::NORM_L1 ?
                (
                    #if (!IPP_DISABLE_NORM_8U)
                    type == CV_8U ? (ippiNormRelFuncNoHint)ippiNormRel_L1_8u_C1R :
                    #endif
                    type == CV_16U ? (ippiNormRelFuncNoHint)ippiNormRel_L1_16u_C1R :
                    type == CV_16S ? (ippiNormRelFuncNoHint)ippiNormRel_L1_16s_C1R :
                    0) :
                    norm_type == cv::NORM_L2 || norm_type == cv::NORM_L2SQR ?
                    (
                        #if (!IPP_DISABLE_NORM_8U)
                        type == CV_8U ? (ippiNormRelFuncNoHint)ippiNormRel_L2_8u_C1R :
                        #endif
                        type == CV_16U ? (ippiNormRelFuncNoHint)ippiNormRel_L2_16u_C1R :
                        type == CV_16S ? (ippiNormRelFuncNoHint)ippiNormRel_L2_16s_C1R :
                        0) : 0;
                        if( ippiNormRelHint || ippiNormRel )
                        {
                            Ipp64f norm;
                            IppStatus ret = ippiNormRelHint ? CV_INSTRUMENT_FUN_IPP(ippiNormRelHint, src1, (int)src1_step, src2, (int)src2_step, sz, &norm, ippAlgHintAccurate) :
                            CV_INSTRUMENT_FUN_IPP(ippiNormRel, src1, (int)src1_step, src2, (int)src2_step, sz, &norm);
                            if( ret >= 0 )
                            {
                                *result = (norm_type == cv::NORM_L2SQR) ? norm * norm : norm;
                                return CV_HAL_ERROR_OK;
                            }
                        }
        }
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    norm_type &= cv::NORM_TYPE_MASK;

    if( mask )
    {
        IppiSize sz = { width, height };

        typedef IppStatus (CV_STDCALL* ippiMaskNormDiffFuncC1)(const void *, int, const void *, int, const void *, int, IppiSize, Ipp64f *);
        ippiMaskNormDiffFuncC1 ippiNormDiff_C1MR =
        norm_type == cv::NORM_INF ?
        (type == CV_8UC1 ? (ippiMaskNormDiffFuncC1)ippiNormDiff_Inf_8u_C1MR :
        type == CV_16UC1 ? (ippiMaskNormDiffFuncC1)ippiNormDiff_Inf_16u_C1MR :
        type == CV_32FC1 ? (ippiMaskNormDiffFuncC1)ippiNormDiff_Inf_32f_C1MR :
        0) :
        norm_type == cv::NORM_L1 ?
        (type == CV_8UC1 ? (ippiMaskNormDiffFuncC1)ippiNormDiff_L1_8u_C1MR :
        type == CV_16UC1 ? (ippiMaskNormDiffFuncC1)ippiNormDiff_L1_16u_C1MR :
        type == CV_32FC1 ? (ippiMaskNormDiffFuncC1)ippiNormDiff_L1_32f_C1MR :
        0) :
        norm_type == cv::NORM_L2 || norm_type == cv::NORM_L2SQR ?
        (type == CV_8UC1 ? (ippiMaskNormDiffFuncC1)ippiNormDiff_L2_8u_C1MR :
        type == CV_16UC1 ? (ippiMaskNormDiffFuncC1)ippiNormDiff_L2_16u_C1MR :
        type == CV_32FC1 ? (ippiMaskNormDiffFuncC1)ippiNormDiff_L2_32f_C1MR :
        0) : 0;
        if( ippiNormDiff_C1MR )
        {
            Ipp64f norm;
            if( CV_INSTRUMENT_FUN_IPP(ippiNormDiff_C1MR, src1, (int)src1_step, src2, (int)src2_step, mask, (int)mask_step, sz, &norm) >= 0 )
            {
                *result = (norm_type == cv::NORM_L2SQR ? (double)(norm * norm) : (double)norm);
                return CV_HAL_ERROR_OK;
            }
        }
        typedef IppStatus (CV_STDCALL* ippiMaskNormDiffFuncC3)(const void *, int, const void *, int, const void *, int, IppiSize, int, Ipp64f *);
        ippiMaskNormDiffFuncC3 ippiNormDiff_C3CMR =
        norm_type == cv::NORM_INF ?
        (type == CV_8UC3 ? (ippiMaskNormDiffFuncC3)ippiNormDiff_Inf_8u_C3CMR :
        type == CV_16UC3 ? (ippiMaskNormDiffFuncC3)ippiNormDiff_Inf_16u_C3CMR :
        type == CV_32FC3 ? (ippiMaskNormDiffFuncC3)ippiNormDiff_Inf_32f_C3CMR :
        0) :
        norm_type == cv::NORM_L1 ?
        (type == CV_8UC3 ? (ippiMaskNormDiffFuncC3)ippiNormDiff_L1_8u_C3CMR :
        type == CV_16UC3 ? (ippiMaskNormDiffFuncC3)ippiNormDiff_L1_16u_C3CMR :
        type == CV_32FC3 ? (ippiMaskNormDiffFuncC3)ippiNormDiff_L1_32f_C3CMR :
        0) :
        norm_type == cv::NORM_L2 || norm_type == cv::NORM_L2SQR ?
        (type == CV_8UC3 ? (ippiMaskNormDiffFuncC3)ippiNormDiff_L2_8u_C3CMR :
        type == CV_16UC3 ? (ippiMaskNormDiffFuncC3)ippiNormDiff_L2_16u_C3CMR :
        type == CV_32FC3 ? (ippiMaskNormDiffFuncC3)ippiNormDiff_L2_32f_C3CMR :
        0) : 0;
        if( ippiNormDiff_C3CMR )
        {
            Ipp64f norm1, norm2, norm3;
            if( CV_INSTRUMENT_FUN_IPP(ippiNormDiff_C3CMR, src1, (int)src1_step, src2, (int)src2_step, mask, (int)mask_step, sz, 1, &norm1) >= 0 &&
                CV_INSTRUMENT_FUN_IPP(ippiNormDiff_C3CMR, src1, (int)src1_step, src2, (int)src2_step, mask, (int)mask_step, sz, 2, &norm2) >= 0 &&
                CV_INSTRUMENT_FUN_IPP(ippiNormDiff_C3CMR, src1, (int)src1_step, src2, (int)src2_step, mask, (int)mask_step, sz, 3, &norm3) >= 0)
            {
                Ipp64f norm =
                norm_type == cv::NORM_INF ? std::max(std::max(norm1, norm2), norm3) :
                norm_type == cv::NORM_L1 ? norm1 + norm2 + norm3 :
                norm_type == cv::NORM_L2 || norm_type == cv::NORM_L2SQR ? std::sqrt(norm1 * norm1 + norm2 * norm2 + norm3 * norm3) :
                0;
                *result = (norm_type == cv::NORM_L2SQR ? (double)(norm * norm) : (double)norm);
                return CV_HAL_ERROR_OK;
            }
        }
    }
    else
    {
        int cn = CV_MAT_CN(type);
        type = CV_MAT_DEPTH(type);
        IppiSize sz = { width*cn, height };

        typedef IppStatus (CV_STDCALL* ippiNormDiffFuncHint)(const void *, int, const void *, int, IppiSize, Ipp64f *, IppHintAlgorithm hint);
        typedef IppStatus (CV_STDCALL* ippiNormDiffFuncNoHint)(const void *, int, const void *, int, IppiSize, Ipp64f *);
        ippiNormDiffFuncHint ippiNormDiffHint =
        norm_type == cv::NORM_L1 ?
        (type == CV_32F ? (ippiNormDiffFuncHint)ippiNormDiff_L1_32f_C1R :
        0) :
        norm_type == cv::NORM_L2 || norm_type == cv::NORM_L2SQR ?
        (type == CV_32F ? (ippiNormDiffFuncHint)ippiNormDiff_L2_32f_C1R :
        0) : 0;
        ippiNormDiffFuncNoHint ippiNormDiff =
        norm_type == cv::NORM_INF ?
        (
            #if (!IPP_DISABLE_NORM_8U)
            type == CV_8U ? (ippiNormDiffFuncNoHint)ippiNormDiff_Inf_8u_C1R :
            #endif
            type == CV_16U ? (ippiNormDiffFuncNoHint)ippiNormDiff_Inf_16u_C1R :
            type == CV_16S ? (ippiNormDiffFuncNoHint)ippiNormDiff_Inf_16s_C1R :
            type == CV_32F ? (ippiNormDiffFuncNoHint)ippiNormDiff_Inf_32f_C1R :
            0) :
            norm_type == cv::NORM_L1 ?
            (
                #if (!IPP_DISABLE_NORM_8U)
                type == CV_8U ? (ippiNormDiffFuncNoHint)ippiNormDiff_L1_8u_C1R :
                #endif
                type == CV_16U ? (ippiNormDiffFuncNoHint)ippiNormDiff_L1_16u_C1R :
                type == CV_16S ? (ippiNormDiffFuncNoHint)ippiNormDiff_L1_16s_C1R :
                0) :
                norm_type == cv::NORM_L2 || norm_type == cv::NORM_L2SQR ?
                (
                    #if (!IPP_DISABLE_NORM_8U)
                    type == CV_8U ? (ippiNormDiffFuncNoHint)ippiNormDiff_L2_8u_C1R :
                    #endif
                    type == CV_16U ? (ippiNormDiffFuncNoHint)ippiNormDiff_L2_16u_C1R :
                    type == CV_16S ? (ippiNormDiffFuncNoHint)ippiNormDiff_L2_16s_C1R :
                    0) : 0;
                    if( ippiNormDiffHint || ippiNormDiff )
                    {
                        Ipp64f norm;
                        IppStatus ret = ippiNormDiffHint ? CV_INSTRUMENT_FUN_IPP(ippiNormDiffHint, src1, (int)src1_step, src2, (int)src2_step, sz, &norm, ippAlgHintAccurate) :
                        CV_INSTRUMENT_FUN_IPP(ippiNormDiff, src1, (int)src1_step, src2, (int)src2_step, sz, &norm);
                        if( ret >= 0 )
                        {
                            *result = (norm_type == cv::NORM_L2SQR) ? norm * norm : norm;
                            return CV_HAL_ERROR_OK;
                        }
                    }
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}


#endif
