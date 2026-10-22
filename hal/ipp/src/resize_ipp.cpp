// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
// Copyright (C) 2026, Intel Corporation, all rights reserved.

#include "ipp_hal_imgproc.hpp"

#ifdef HAVE_IPP_IW

#include "resize_ipp.hpp"

#include <cfloat>

#define IPP_RESIZE_PARALLEL 1

int ipp_resize_prepare(int src_type, int src_width, int src_height, int dst_width, int dst_height,
                       double inv_scale_x, double inv_scale_y, int interpolation, IppiInterpolationType& inter, bool& affine)
{
    int depth = CV_MAT_DEPTH(src_type), channels = CV_MAT_CN(src_type);

    IppiInterpolationType ippInter    = ippiGetInterpolation(interpolation);
    int                   interpIdx   = interpolation & cv::InterpolationFlags::INTER_MAX;
    if((int)ippInter < 0 || interpIdx > 4 || channels > 4)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

#if defined(IPP_CALLS_ENFORCED)

    const char impl[CV_DEPTH_MAX][4][5] = {      /* N  L  C  S  Z */
        /* 8U  */ {{1, 1, 1, 1, 1},{0, 0, 0, 0, 0},{1, 1, 1, 1, 1},{1, 1, 1, 1, 1}},
        /* 8S  */ {{0, 0, 0, 0, 0},{0, 0, 0, 0, 0},{0, 0, 0, 0, 0},{0, 0, 0, 0, 0}},
        /* 16U */ {{1, 1, 1, 1, 1},{0, 0, 0, 0, 0},{1, 1, 1, 1, 1},{1, 1, 1, 1, 1}},
        /* 16S */ {{1, 1, 1, 1, 1},{0, 0, 0, 0, 0},{1, 1, 1, 1, 1},{1, 1, 1, 1, 1}},
        /* 32S */ {{0, 0, 0, 0, 0},{0, 0, 0, 0, 0},{0, 0, 0, 0, 0},{0, 0, 0, 0, 0}},
        /* 32F */ {{1, 1, 1, 1, 1},{0, 0, 0, 0, 0},{1, 1, 1, 1, 1},{1, 1, 1, 1, 1}},
        /* 64F */ {{0, 1, 0, 0, 0},{0, 0, 0, 0, 0},{0, 1, 0, 0, 0},{0, 1, 0, 0, 0}},
        /* 16F */ {{0, 0, 0, 0, 0},{0, 0, 0, 0, 0},{0, 0, 0, 0, 0},{0, 0, 0, 0, 0}}};
#else // IPP_CALLS_ENFORCED is not defined, results are strictly aligned to OpenCV implementation

    const char impl[CV_DEPTH_MAX][4][5] = {      /* N  L  C  S  Z */
        /* 8U  */ {{0, 0, 1, 0, 1},{0, 0, 0, 0, 0},{0, 0, 1, 0, 1},{0, 0, 1, 0, 1}},
        /* 8S  */ {{0, 0, 0, 0, 0},{0, 0, 0, 0, 0},{0, 0, 0, 0, 0},{0, 0, 0, 0, 0}},
        /* 16U */ {{0, 1, 1, 0, 1},{0, 0, 0, 0, 0},{0, 1, 1, 0, 1},{0, 1, 1, 0, 1}},
        /* 16S */ {{0, 1, 1, 0, 1},{0, 0, 0, 0, 0},{0, 1, 1, 0, 1},{0, 1, 1, 0, 1}},
        /* 32S */ {{0, 0, 0, 0, 0},{0, 0, 0, 0, 0},{0, 0, 0, 0, 0},{0, 0, 0, 0, 0}},
        /* 32F */ {{0, 1, 1, 0, 1},{0, 0, 0, 0, 0},{0, 1, 1, 0, 1},{0, 1, 1, 0, 1}},
        /* 64F */ {{0, 1, 0, 0, 0},{0, 0, 0, 0, 0},{0, 1, 0, 0, 0},{0, 1, 0, 0, 0}},
        /* 16F */ {{0, 0, 0, 0, 0},{0, 0, 0, 0, 0},{0, 0, 0, 0, 0},{0, 0, 0, 0, 0}}};
#endif

    if (impl[depth][channels - 1][interpIdx] == 0)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

#if IPP_VERSION_X100 < 201801 && !defined(IPP_CALLS_ENFORCED)
    // Degradations on int^2 linear downscale
    if (depth != CV_64F && ippInter == ippLinear && inv_scale_x < 1 && inv_scale_y < 1) // if downscale
    {
        int scale_x = (int)(1 / inv_scale_x);
        int scale_y = (int)(1 / inv_scale_y);
        if (1 / inv_scale_x - scale_x < DBL_EPSILON && 1 / inv_scale_y - scale_y < DBL_EPSILON) // if integer
        {
            if (!(scale_x&(scale_x - 1)) && !(scale_y&(scale_y - 1))) // if power of 2
                return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }
    }
#endif

    affine = false;
    const double IPP_RESIZE_EPS = (depth == CV_64F) ? 0 : 1e-10;
    double ex = fabs((double)dst_width / src_width  - inv_scale_x) / inv_scale_x;
    double ey = fabs((double)dst_height / src_height - inv_scale_y) / inv_scale_y;

    // Use affine transform resize to allow sub-pixel accuracy
    if(ex > IPP_RESIZE_EPS || ey > IPP_RESIZE_EPS)
        affine = true;

    // Affine doesn't support Lanczos and Super interpolations
    if(affine && (ippInter == ippLanczos || ippInter == ippSuper))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    inter = ippInter;
    return CV_HAL_ERROR_OK;
}

int ipp_hal_resize(int src_type, const uchar *src_data, size_t src_step, int src_width, int src_height,
                   uchar *dst_data, size_t dst_step, int dst_width, int dst_height,
                   double inv_scale_x, double inv_scale_y, int interpolation)
{
    CV_HAL_CHECK_USE_IPP();

    IppiInterpolationType ippInter;
    bool affine;
    if (ipp_resize_prepare(src_type, src_width, src_height, dst_width, dst_height,
                           inv_scale_x, inv_scale_y, interpolation, ippInter, affine) != CV_HAL_ERROR_OK)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    IppDataType ippDataType = ippiGetDataType(CV_MAT_DEPTH(src_type));
    int         channels    = CV_MAT_CN(src_type);

    try
    {
        ::ipp::IwiImage iwSrc(::ipp::IwiSize(src_width, src_height), ippDataType, channels, 0, (void*)src_data, src_step);
        ::ipp::IwiImage iwDst(::ipp::IwiSize(dst_width, dst_height), ippDataType, channels, 0, (void*)dst_data, dst_step);

        std::atomic_bool ok{true};
        int   threads = ippiSuggestThreadsNum(iwDst, 1+((double)(src_width*src_height)/(dst_width*dst_height)));
        cv::Range range(0, dst_height);
        ipp_resizeParallel       invokerGeneral(iwSrc, iwDst, ok);
        ipp_resizeAffineParallel invokerAffine(iwSrc, iwDst, ok);
        cv::ParallelLoopBody    *pInvoker = NULL;
        if(affine)
        {
            pInvoker = &invokerAffine;
            invokerAffine.Init(ippInter, inv_scale_x, inv_scale_y);
        }
        else
        {
            pInvoker = &invokerGeneral;
            invokerGeneral.Init(ippInter);
        }

        if(IPP_RESIZE_PARALLEL && threads > 1)
            cv::parallel_for_(range, *pInvoker, threads*4);
        else
            pInvoker->operator()(range);

        if(!ok)
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }
    catch(const ::ipp::IwException &)
    {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    return CV_HAL_ERROR_OK;
}

#endif // HAVE_IPP_IW
