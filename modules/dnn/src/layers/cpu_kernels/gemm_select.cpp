// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../../precomp.hpp"
#include "gemm_select.hpp"
#include "mlas_gemm.hpp"
#include <opencv2/core/hal/hal.hpp>
#include <opencv2/core/private.hpp>
#include <opencv2/core/hal/intrin.hpp>
#include <opencv2/core/utils/configuration.private.hpp>
#include <opencv2/core/utils/logger.hpp>

namespace cv { namespace dnn {

GemmPath gemmPath()
{
    static const GemmPath path = []() {
        const std::string s = utils::getConfigurationParameterString("OPENCV_DNN_GEMM_PATH", "auto");
        if (s == "engine")
            return GEMM_PATH_ENGINE;
        if (s == "mlas")
            return GEMM_PATH_MLAS;
        if (s != "auto")
            CV_LOG_WARNING(NULL, "OPENCV_DNN_GEMM_PATH=" << s << " is not engine|mlas|auto, using auto");
        return GEMM_PATH_AUTO;
    }();
    return path;
}

bool gemmUseEngine(int M, int N, int K, bool constB)
{
#if CV_NEON && CV_NEON_AARCH64
    CV_UNUSED(N); CV_UNUSED(K);
    const GemmPath path = gemmPath();
    if (path != GEMM_PATH_AUTO)
        return path == GEMM_PATH_ENGINE;
    // Defaults measured on Apple M5 against MLAS, with SME on and with SME masked.
    if (!mlasAvailable())
        return true;                        // the engine beats the dnn kernels everywhere
    static const bool sme = hal::gemm32fUsesSME();
    if (sme)
        return constB || M > 1;             // MLAS keeps only the unpacked GEMV
    return constB || (M >= 2 && M <= 32);   // MLAS keeps the unpacked GEMV and large unpacked products
#else
    CV_UNUSED(M); CV_UNUSED(N); CV_UNUSED(K); CV_UNUSED(constB);
    return false;
#endif
}

bool gemmEngineEnabled()
{
#if CV_NEON && CV_NEON_AARCH64
    return gemmPath() != GEMM_PATH_MLAS;
#else
    return false;
#endif
}

void EngineGemmB::pack(const float* B, size_t B_step, int n_, int k_, bool transB, int count_, size_t sliceElems)
{
    const int flags = transB ? CV_HAL_GEMM_2_T : 0;
    stride = alignSize(hal::gemm32fPackBSize(n_, k_, flags), 64);
    CV_Assert(stride * count_ <= (size_t)INT_MAX);
    buf.create(1, (int)(stride * count_), CV_8U);
    for (int i = 0; i < count_; i++)
        hal::gemm32fPackB(B + i * sliceElems, B_step, n_, k_, flags, buf.data + i * stride);
    n = n_;
    k = k_;
    count = count_;
}

void EngineGemmB::release()
{
    buf.release();
    n = k = count = 0;
    stride = 0;
}

void EngineGemmB::gemm(int i, const float* A, size_t A_step, bool transA, float* C, size_t C_step, int M,
                       float alpha, float beta, bool multiThread) const
{
    CV_DbgAssert(0 <= i && i < count);
    hal::gemm32fPacked(A, A_step, buf.data + i * stride, C, C_step, M, n, k, alpha, beta,
                       (transA ? CV_HAL_GEMM_1_T : 0) | (multiThread ? 0 : CV_HAL_GEMM_SINGLE_THREAD));
}

}} // cv::dnn
