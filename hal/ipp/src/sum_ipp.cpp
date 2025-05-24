#include "ipp_hal_core.hpp"

#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>

#if IPP_VERSION_X100 >= 700

int ipp_hal_sum(const uchar *src_data, size_t src_step, int src_type, int width, int height, double *result)
{
    int cn = CV_MAT_CN(src_type);
    if (cn > 4)
    {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    IppiSize sz = { width, height };

    typedef IppStatus (CV_STDCALL* ippiSumFuncHint)(const void*, int, IppiSize, double *, IppHintAlgorithm);
    typedef IppStatus (CV_STDCALL* ippiSumFuncNoHint)(const void*, int, IppiSize, double *);
    ippiSumFuncHint ippiSumHint =
        src_type == CV_32FC1 ? (ippiSumFuncHint)ippiSum_32f_C1R :
        src_type == CV_32FC3 ? (ippiSumFuncHint)ippiSum_32f_C3R :
        src_type == CV_32FC4 ? (ippiSumFuncHint)ippiSum_32f_C4R :
        0;
    ippiSumFuncNoHint ippiSum =
        src_type == CV_8UC1 ? (ippiSumFuncNoHint)ippiSum_8u_C1R :
        src_type == CV_8UC3 ? (ippiSumFuncNoHint)ippiSum_8u_C3R :
        src_type == CV_8UC4 ? (ippiSumFuncNoHint)ippiSum_8u_C4R :
        src_type == CV_16UC1 ? (ippiSumFuncNoHint)ippiSum_16u_C1R :
        src_type == CV_16UC3 ? (ippiSumFuncNoHint)ippiSum_16u_C3R :
        src_type == CV_16UC4 ? (ippiSumFuncNoHint)ippiSum_16u_C4R :
        src_type == CV_16SC1 ? (ippiSumFuncNoHint)ippiSum_16s_C1R :
        src_type == CV_16SC3 ? (ippiSumFuncNoHint)ippiSum_16s_C3R :
        src_type == CV_16SC4 ? (ippiSumFuncNoHint)ippiSum_16s_C4R :
        0;

    if( ippiSumHint || ippiSum )
    {
        IppStatus ret = ippiSumHint ?
        CV_INSTRUMENT_FUN_IPP(ippiSumHint, src_data, (int)src_step, sz, result, ippAlgHintAccurate) :
        CV_INSTRUMENT_FUN_IPP(ippiSum, src_data, (int)src_step, sz, result);
        if( ret >= 0 )
        {
            return CV_HAL_ERROR_OK;
        }
        else
        {
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

#endif
