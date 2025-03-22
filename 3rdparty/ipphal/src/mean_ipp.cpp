#include "ipp_hal_core.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/core/base.hpp>

#if IPP_VERSION_X100 >= 700

static int ipp_mean(const uchar* src_data, size_t src_step, int width, int height,
                    int src_type, double* mean_val, uchar* mask, size_t mask_step)
{
    int cn = CV_MAT_CN(src_type);
    if (cn > 4)
    {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    if((src_step == 1 || src_step == static_cast<size_t>(width)) && (mask_step == 1 || mask_step == static_cast<size_t>(width)))
    {
        IppiSize sz = { width, height };
        if( mask )
        {
            typedef IppStatus (CV_STDCALL* ippiMaskMeanFuncC1)(const void *, int, const void *, int, IppiSize, Ipp64f *);
            ippiMaskMeanFuncC1 ippiMean_C1MR =
            src_type == CV_8UC1 ? (ippiMaskMeanFuncC1)ippiMean_8u_C1MR :
            src_type == CV_16UC1 ? (ippiMaskMeanFuncC1)ippiMean_16u_C1MR :
            src_type == CV_32FC1 ? (ippiMaskMeanFuncC1)ippiMean_32f_C1MR :
            0;
            if( ippiMean_C1MR )
            {
                if( CV_INSTRUMENT_FUN_IPP(ippiMean_C1MR, src_data, (int)src_step, mask, (int)mask_step, sz, mean_val) >= 0 )
                {
                    return CV_HAL_ERROR_OK;
                }
            }
            typedef IppStatus (CV_STDCALL* ippiMaskMeanFuncC3)(const void *, int, const void *, int, IppiSize, int, Ipp64f *);
            ippiMaskMeanFuncC3 ippiMean_C3MR =
            src_type == CV_8UC3 ? (ippiMaskMeanFuncC3)ippiMean_8u_C3CMR :
            src_type == CV_16UC3 ? (ippiMaskMeanFuncC3)ippiMean_16u_C3CMR :
            src_type == CV_32FC3 ? (ippiMaskMeanFuncC3)ippiMean_32f_C3CMR :
            0;
            if( ippiMean_C3MR )
            {
                if( CV_INSTRUMENT_FUN_IPP(ippiMean_C3MR, src_data, (int)src_step, mask, (int)mask_step, sz, 1, &mean_val[0]) >= 0 &&
                    CV_INSTRUMENT_FUN_IPP(ippiMean_C3MR, src_data, (int)src_step, mask, (int)mask_step, sz, 2, &mean_val[1]) >= 0 &&
                    CV_INSTRUMENT_FUN_IPP(ippiMean_C3MR, src_data, (int)src_step, mask, (int)mask_step, sz, 3, &mean_val[2]) >= 0 )
                {
                    return CV_HAL_ERROR_OK;
                }
            }
        }
        else
        {
            typedef IppStatus (CV_STDCALL* ippiMeanFuncHint)(const void*, int, IppiSize, double *, IppHintAlgorithm);
            typedef IppStatus (CV_STDCALL* ippiMeanFuncNoHint)(const void*, int, IppiSize, double *);
            ippiMeanFuncHint ippiMeanHint =
            src_type == CV_32FC1 ? (ippiMeanFuncHint)ippiMean_32f_C1R :
            src_type == CV_32FC3 ? (ippiMeanFuncHint)ippiMean_32f_C3R :
            src_type == CV_32FC4 ? (ippiMeanFuncHint)ippiMean_32f_C4R :
            0;
            ippiMeanFuncNoHint ippiMean =
            src_type == CV_8UC1 ? (ippiMeanFuncNoHint)ippiMean_8u_C1R :
            src_type == CV_8UC3 ? (ippiMeanFuncNoHint)ippiMean_8u_C3R :
            src_type == CV_8UC4 ? (ippiMeanFuncNoHint)ippiMean_8u_C4R :
            src_type == CV_16UC1 ? (ippiMeanFuncNoHint)ippiMean_16u_C1R :
            src_type == CV_16UC3 ? (ippiMeanFuncNoHint)ippiMean_16u_C3R :
            src_type == CV_16UC4 ? (ippiMeanFuncNoHint)ippiMean_16u_C4R :
            src_type == CV_16SC1 ? (ippiMeanFuncNoHint)ippiMean_16s_C1R :
            src_type == CV_16SC3 ? (ippiMeanFuncNoHint)ippiMean_16s_C3R :
            src_type == CV_16SC4 ? (ippiMeanFuncNoHint)ippiMean_16s_C4R :
            0;

            // Make sure only zero or one version of the function pointer is valid
            CV_Assert(!ippiMeanHint || !ippiMean);
            if( ippiMeanHint || ippiMean )
            {
                IppStatus status = ippiMeanHint ? CV_INSTRUMENT_FUN_IPP(ippiMeanHint, src_data, (int)src_step, sz, mean_val, ippAlgHintAccurate) :
                CV_INSTRUMENT_FUN_IPP(ippiMean, src_data, (int)src_step, sz, mean_val);
                if( status >= 0 )
                {
                    return CV_HAL_ERROR_OK;
                }
            }
        }
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}



static int ipp_meanStdDev(const uchar* src_data, size_t src_step, int width, int height,
                          int src_type, double* mean_val, double* stddev_val, uchar* mask, size_t mask_step)
{
    int cn = CV_MAT_CN(src_type);

    if((src_step == 1 || src_step == static_cast<size_t>(width)) && (mask_step == 1 || mask_step == static_cast<size_t>(width)))
    {
        Ipp64f mean_temp[3];
        Ipp64f stddev_temp[3];
        Ipp64f *pmean = &mean_temp[0];
        Ipp64f *pstddev = &stddev_temp[0];
        int dcn_mean = -1;
        if( mean_val )
        {
            dcn_mean = cn;
            pmean = mean_val;
        }
        int dcn_stddev = -1;
        if( stddev_val )
        {
            dcn_stddev = cn;
            pstddev = stddev_val;
        }

        for( int c = cn; c < dcn_mean; c++ )
            pmean[c] = 0;
        for( int c = cn; c < dcn_stddev; c++ )
            pstddev[c] = 0;

        IppiSize sz = { width, height };
        if( !mask )
        {
            typedef IppStatus (CV_STDCALL* ippiMaskMeanStdDevFuncC1)(const void *, int, const void *, int, IppiSize, Ipp64f *, Ipp64f *);
            ippiMaskMeanStdDevFuncC1 ippiMean_StdDev_C1MR =
            src_type == CV_8UC1 ? (ippiMaskMeanStdDevFuncC1)ippiMean_StdDev_8u_C1MR :
            src_type == CV_16UC1 ? (ippiMaskMeanStdDevFuncC1)ippiMean_StdDev_16u_C1MR :
            src_type == CV_32FC1 ? (ippiMaskMeanStdDevFuncC1)ippiMean_StdDev_32f_C1MR :
            nullptr;
            if( ippiMean_StdDev_C1MR )
            {
                if( CV_INSTRUMENT_FUN_IPP(ippiMean_StdDev_C1MR, src_data, (int)src_step, mask, (int)mask_step, sz, pmean, pstddev) >= 0 )
                {
                    return CV_HAL_ERROR_OK;
                }
            }

            typedef IppStatus (CV_STDCALL* ippiMaskMeanStdDevFuncC3)(const void *, int, const void *, int, IppiSize, int, Ipp64f *, Ipp64f *);
            ippiMaskMeanStdDevFuncC3 ippiMean_StdDev_C3CMR =
            src_type == CV_8UC3 ? (ippiMaskMeanStdDevFuncC3)ippiMean_StdDev_8u_C3CMR :
            src_type == CV_16UC3 ? (ippiMaskMeanStdDevFuncC3)ippiMean_StdDev_16u_C3CMR :
            src_type == CV_32FC3 ? (ippiMaskMeanStdDevFuncC3)ippiMean_StdDev_32f_C3CMR :
            nullptr;
            if( ippiMean_StdDev_C3CMR )
            {
                if( CV_INSTRUMENT_FUN_IPP(ippiMean_StdDev_C3CMR, src_data, (int)src_step, mask, (int)mask_step, sz, 1, &pmean[0], &pstddev[0]) >= 0 &&
                    CV_INSTRUMENT_FUN_IPP(ippiMean_StdDev_C3CMR, src_data, (int)src_step, mask, (int)mask_step, sz, 2, &pmean[1], &pstddev[1]) >= 0 &&
                    CV_INSTRUMENT_FUN_IPP(ippiMean_StdDev_C3CMR, src_data, (int)src_step, mask, (int)mask_step, sz, 3, &pmean[2], &pstddev[2]) >= 0 )
                {
                    return CV_HAL_ERROR_OK;
                }
            }
        }
        else
        {
            typedef IppStatus (CV_STDCALL* ippiMeanStdDevFuncC1)(const void *, int, IppiSize, Ipp64f *, Ipp64f *);
            ippiMeanStdDevFuncC1 ippiMean_StdDev_C1R =
                src_type == CV_8UC1 ? (ippiMeanStdDevFuncC1)ippiMean_StdDev_8u_C1R :
                src_type == CV_16UC1 ? (ippiMeanStdDevFuncC1)ippiMean_StdDev_16u_C1R :
            #if (IPP_VERSION_X100 >= 810)
                src_type == CV_32FC1 ? (ippiMeanStdDevFuncC1)ippiMean_StdDev_32f_C1R ://Aug 2013: bug in IPP 7.1, 8.0
            #endif
                nullptr;
            if( ippiMean_StdDev_C1R )
            {
                if( CV_INSTRUMENT_FUN_IPP(ippiMean_StdDev_C1R, src_data, (int)src_step, sz, pmean, pstddev) >= 0 )
                {
                    return CV_HAL_ERROR_OK;
                }
            }

            typedef IppStatus (CV_STDCALL* ippiMeanStdDevFuncC3)(const void *, int, IppiSize, int, Ipp64f *, Ipp64f *);
                ippiMeanStdDevFuncC3 ippiMean_StdDev_C3CR =
                src_type == CV_8UC3 ? (ippiMeanStdDevFuncC3)ippiMean_StdDev_8u_C3CR :
                src_type == CV_16UC3 ? (ippiMeanStdDevFuncC3)ippiMean_StdDev_16u_C3CR :
                src_type == CV_32FC3 ? (ippiMeanStdDevFuncC3)ippiMean_StdDev_32f_C3CR :
                nullptr;
            if( ippiMean_StdDev_C3CR )
            {
                if( CV_INSTRUMENT_FUN_IPP(ippiMean_StdDev_C3CR, src_data, (int)src_step, sz, 1, &pmean[0], &pstddev[0]) >= 0 &&
                    CV_INSTRUMENT_FUN_IPP(ippiMean_StdDev_C3CR, src_data, (int)src_step, sz, 2, &pmean[1], &pstddev[1]) >= 0 &&
                    CV_INSTRUMENT_FUN_IPP(ippiMean_StdDev_C3CR, src_data, (int)src_step, sz, 3, &pmean[2], &pstddev[2]) >= 0 )
                {
                    return CV_HAL_ERROR_OK;
                }
            }
        }
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

int ipp_hal_meanStdDev(const uchar* src_data, size_t src_step, int width, int height, int src_type,
                       double* mean_val, double* stddev_val, uchar* mask, size_t mask_step)
{
    if (stddev_val)
    {
        return ipp_meanStdDev(src_data, src_step, width, height, src_type, mean_val, stddev_val, mask, mask_step);
    }
    else
    {
        return ipp_mean(src_data, src_step, width, height, src_type, mean_val, mask, mask_step);
    }
}


#endif // IPP_VERSION_X100 >= 700
