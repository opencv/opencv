// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
// Copyright (C) 2026, Intel Corporation, all rights reserved.

#include "ipp_hal_imgproc.hpp"

#include <opencv2/core.hpp>
#include "precomp_ipp.hpp"

#include <climits>

#if IPP_VERSION_X100 >= 202610

static bool ipp_pyr_checkArgs(int depth, int cn, size_t src_step, size_t dst_step, int border_type)
{
    return (border_type & ~cv::BORDER_ISOLATED) == cv::BORDER_DEFAULT &&
           depth >= 0 && depth < CV_DEPTH_MAX && cn >= 1 && cn <= 4 &&
           src_step <= (size_t)INT_MAX && dst_step <= (size_t)INT_MAX;
}

int ipp_hal_pyrdown(const uchar* src_data, size_t src_step, int src_width, int src_height,
                    uchar* dst_data, size_t dst_step, int dst_width, int dst_height,
                    int depth, int cn, int border_type)
{
    CV_HAL_CHECK_USE_IPP();

    if (!ipp_pyr_checkArgs(depth, cn, src_step, dst_step, border_type))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (dst_width != (src_width + 1)/2 || dst_height != (src_height + 1)/2)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    // It is slower than OpenCV's fused, multi-threaded implementation for all types,
    // so it is used only with IPP_CALLS_ENFORCED.
                                      /* C1 C2 C3 C4 */
#if defined(IPP_CALLS_ENFORCED)
    const char impl[CV_DEPTH_MAX][4] = {{1, 0, 1, 0},   //8U
                                        {0, 0, 0, 0},   //8S
                                        {1, 0, 1, 0},   //16U
                                        {0, 0, 0, 0},   //16S
                                        {0, 0, 0, 0},   //32S
                                        {1, 0, 1, 0},   //32F
                                        {0, 0, 0, 0}};  //64F
#else
    const char impl[CV_DEPTH_MAX][4] = {{0, 0, 0, 0},   //8U
                                        {0, 0, 0, 0},   //8S
                                        {0, 0, 0, 0},   //16U
                                        {0, 0, 0, 0},   //16S
                                        {0, 0, 0, 0},   //32S
                                        {0, 0, 0, 0},   //32F
                                        {0, 0, 0, 0}};  //64F
#endif // IPP_CALLS_ENFORCED

    if (impl[depth][cn-1] == 0)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    typedef IppStatus (CV_STDCALL * ippiPyrDownGetSize)(IppiSize srcRoi, Ipp32f rate, int kernelSize, int* pStateSize, int* pBufSize);
    typedef IppStatus (CV_STDCALL * ippiPyrDownInit)(void** state, IppiSize srcRoi, Ipp32f rate, void* pKernel, int kernelSize, int mode, Ipp8u* stateBuf, Ipp8u* buffer);
    typedef IppStatus (CV_STDCALL * ippiPyrDown)(const void* pSrc, int srcStep, IppiSize srcRoiSize, void* pDst, int dstStep, IppiSize dstRoiSize, void* pState);

    int type = CV_MAKETYPE(depth, cn);
    ippiPyrDownGetSize pyrDownGetSizeFunc = type == CV_8UC1  ? (ippiPyrDownGetSize) ippiPyramidLayerDownGetSize_8u_C1R  :
                                            type == CV_8UC3  ? (ippiPyrDownGetSize) ippiPyramidLayerDownGetSize_8u_C3R  :
                                            type == CV_16UC1 ? (ippiPyrDownGetSize) ippiPyramidLayerDownGetSize_16u_C1R :
                                            type == CV_16UC3 ? (ippiPyrDownGetSize) ippiPyramidLayerDownGetSize_16u_C3R :
                                            type == CV_32FC1 ? (ippiPyrDownGetSize) ippiPyramidLayerDownGetSize_32f_C1R :
                                            type == CV_32FC3 ? (ippiPyrDownGetSize) ippiPyramidLayerDownGetSize_32f_C3R : 0;
    ippiPyrDownInit pyrDownInitFunc = type == CV_8UC1  ? (ippiPyrDownInit) ippiPyramidLayerDownInit_8u_C1R  :
                                      type == CV_8UC3  ? (ippiPyrDownInit) ippiPyramidLayerDownInit_8u_C3R  :
                                      type == CV_16UC1 ? (ippiPyrDownInit) ippiPyramidLayerDownInit_16u_C1R :
                                      type == CV_16UC3 ? (ippiPyrDownInit) ippiPyramidLayerDownInit_16u_C3R :
                                      type == CV_32FC1 ? (ippiPyrDownInit) ippiPyramidLayerDownInit_32f_C1R :
                                      type == CV_32FC3 ? (ippiPyrDownInit) ippiPyramidLayerDownInit_32f_C3R : 0;
    ippiPyrDown pyrDownFunc = type == CV_8UC1  ? (ippiPyrDown) ippiPyramidLayerDown_8u_C1R  :
                              type == CV_8UC3  ? (ippiPyrDown) ippiPyramidLayerDown_8u_C3R  :
                              type == CV_16UC1 ? (ippiPyrDown) ippiPyramidLayerDown_16u_C1R :
                              type == CV_16UC3 ? (ippiPyrDown) ippiPyramidLayerDown_16u_C3R :
                              type == CV_32FC1 ? (ippiPyrDown) ippiPyramidLayerDown_32f_C1R :
                              type == CV_32FC3 ? (ippiPyrDown) ippiPyramidLayerDown_32f_C3R : 0;

    if (!pyrDownGetSizeFunc || !pyrDownInitFunc || !pyrDownFunc)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    int stateSize = 0, bufSize = 0;
    IppiSize srcRoi = { src_width, src_height };
    IppiSize dstRoi = { dst_width, dst_height };
    const Ipp32f rate = 2.0f;
    const int kernelSize = 5;
    Ipp32f kernel32f[kernelSize] = { 1.f/16.f, 4.f/16.f, 6.f/16.f, 4.f/16.f, 1.f/16.f }; // normalized kernel improves performance for small sizes
    Ipp16s kernel16s[kernelSize] = { 1, 4, 6, 4, 1 }; // with rate 2 enables the dedicated single-pass 8u C1 kernel
    void* kernel = (depth == CV_32F) ? (void*)kernel32f : (void*)kernel16s;

    IppStatus ok = pyrDownGetSizeFunc(srcRoi, rate, kernelSize, &stateSize, &bufSize);
    if (ok < 0)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    Ipp8u* pStateBuf = stateSize > 0 ? (Ipp8u*)ippMalloc_L(stateSize) : NULL;
    Ipp8u* pBuffer = bufSize > 0 ? (Ipp8u*)ippMalloc_L(bufSize) : NULL;
    if ((stateSize > 0 && !pStateBuf) || (bufSize > 0 && !pBuffer))
    {
        if (pStateBuf) ippFree(pStateBuf);
        if (pBuffer) ippFree(pBuffer);
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    void* state = NULL;
    ok = pyrDownInitFunc(&state, srcRoi, rate, kernel, kernelSize, IPPI_INTER_LINEAR | IPPI_OCV_ALIGNMENT, pStateBuf, pBuffer);
    if (ok >= 0)
        ok = CV_INSTRUMENT_FUN_IPP(pyrDownFunc, src_data, (int)src_step, srcRoi, dst_data, (int)dst_step, dstRoi, state);

    if (pStateBuf) ippFree(pStateBuf);
    if (pBuffer) ippFree(pBuffer);
    if (ok < 0)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    return CV_HAL_ERROR_OK;
}

int ipp_hal_pyrup(const uchar* src_data, size_t src_step, int src_width, int src_height,
                  uchar* dst_data, size_t dst_step, int dst_width, int dst_height,
                  int depth, int cn, int border_type)
{
    CV_HAL_CHECK_USE_IPP();

    if (!ipp_pyr_checkArgs(depth, cn, src_step, dst_step, border_type))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (dst_width != src_width*2 || dst_height != src_height*2)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    // 16U is used only with IPP_CALLS_ENFORCED.
                                      /* C1 C2 C3 C4 */
#if defined(IPP_CALLS_ENFORCED)
    const char impl[CV_DEPTH_MAX][4] = {{1, 0, 1, 0},   //8U
                                        {0, 0, 0, 0},   //8S
                                        {1, 0, 1, 0},   //16U
                                        {0, 0, 0, 0},   //16S
                                        {0, 0, 0, 0},   //32S
                                        {1, 0, 1, 0},   //32F
                                        {0, 0, 0, 0}};  //64F
#else
    const char impl[CV_DEPTH_MAX][4] = {{1, 0, 1, 0},   //8U
                                        {0, 0, 0, 0},   //8S
                                        {0, 0, 0, 0},   //16U
                                        {0, 0, 0, 0},   //16S
                                        {0, 0, 0, 0},   //32S
                                        {1, 0, 1, 0},   //32F
                                        {0, 0, 0, 0}};  //64F
#endif // IPP_CALLS_ENFORCED

    if (impl[depth][cn-1] == 0)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    typedef IppStatus (CV_STDCALL * ippiPyrUpGetSize)(IppiSize dstRoi, Ipp32f rate, int kernelSize, int* pStateSize);
    typedef IppStatus (CV_STDCALL * ippiPyrUpInit)(void** state, IppiSize dstRoi, Ipp32f rate, void* pKernel, int kernelSize, int mode, Ipp8u* stateBuf);
    typedef IppStatus (CV_STDCALL * ippiPyrUp)(const void* pSrc, int srcStep, IppiSize srcRoiSize, void* pDst, int dstStep, IppiSize dstRoiSize, void* pState);

    int type = CV_MAKETYPE(depth, cn);
    ippiPyrUpGetSize pyrUpGetSizeFunc = type == CV_8UC1  ? (ippiPyrUpGetSize) ippiPyramidLayerUpGetSize_8u_C1R  :
                                        type == CV_8UC3  ? (ippiPyrUpGetSize) ippiPyramidLayerUpGetSize_8u_C3R  :
                                        type == CV_16UC1 ? (ippiPyrUpGetSize) ippiPyramidLayerUpGetSize_16u_C1R :
                                        type == CV_16UC3 ? (ippiPyrUpGetSize) ippiPyramidLayerUpGetSize_16u_C3R :
                                        type == CV_32FC1 ? (ippiPyrUpGetSize) ippiPyramidLayerUpGetSize_32f_C1R :
                                        type == CV_32FC3 ? (ippiPyrUpGetSize) ippiPyramidLayerUpGetSize_32f_C3R : 0;
    ippiPyrUpInit pyrUpInitFunc = type == CV_8UC1  ? (ippiPyrUpInit) ippiPyramidLayerUpInit_8u_C1R  :
                                  type == CV_8UC3  ? (ippiPyrUpInit) ippiPyramidLayerUpInit_8u_C3R  :
                                  type == CV_16UC1 ? (ippiPyrUpInit) ippiPyramidLayerUpInit_16u_C1R :
                                  type == CV_16UC3 ? (ippiPyrUpInit) ippiPyramidLayerUpInit_16u_C3R :
                                  type == CV_32FC1 ? (ippiPyrUpInit) ippiPyramidLayerUpInit_32f_C1R :
                                  type == CV_32FC3 ? (ippiPyrUpInit) ippiPyramidLayerUpInit_32f_C3R : 0;
    ippiPyrUp pyrUpFunc = type == CV_8UC1  ? (ippiPyrUp) ippiPyramidLayerUp_8u_C1R  :
                          type == CV_8UC3  ? (ippiPyrUp) ippiPyramidLayerUp_8u_C3R  :
                          type == CV_16UC1 ? (ippiPyrUp) ippiPyramidLayerUp_16u_C1R :
                          type == CV_16UC3 ? (ippiPyrUp) ippiPyramidLayerUp_16u_C3R :
                          type == CV_32FC1 ? (ippiPyrUp) ippiPyramidLayerUp_32f_C1R :
                          type == CV_32FC3 ? (ippiPyrUp) ippiPyramidLayerUp_32f_C3R : 0;

    if (!pyrUpGetSizeFunc || !pyrUpInitFunc || !pyrUpFunc)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    int stateSize = 0;
    const Ipp32f rate = 2.0f;
    const int kernelSize = 5;
    Ipp32f kernel32f[kernelSize] = { 1.f, 4.f, 6.f, 4.f, 1.f };
    Ipp16s kernel16s[kernelSize] = { 1, 4, 6, 4, 1 };
    void* kernel = (depth == CV_32F) ? (void*)kernel32f : (void*)kernel16s;

    IppiSize srcRoi = { src_width, src_height };
    IppiSize dstRoi = { dst_width, dst_height };

    const int pyrUpMode = IPPI_INTER_LINEAR | IPPI_OCV_ALIGNMENT;

    IppStatus ok = pyrUpGetSizeFunc(dstRoi, rate, kernelSize, &stateSize);
    if (ok < 0)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    Ipp8u* pStateBuf = stateSize > 0 ? (Ipp8u*)ippMalloc_L(stateSize) : NULL;
    if (stateSize > 0 && !pStateBuf)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    void* state = NULL;
    ok = pyrUpInitFunc(&state, dstRoi, rate, kernel, kernelSize, pyrUpMode, pStateBuf);
    if (ok >= 0)
        ok = CV_INSTRUMENT_FUN_IPP(pyrUpFunc, src_data, (int)src_step, srcRoi, dst_data, (int)dst_step, dstRoi, state);

    if (pStateBuf) ippFree(pStateBuf);
    if (ok < 0)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    return CV_HAL_ERROR_OK;
}

#endif // IPP_VERSION_X100 >= 202610
