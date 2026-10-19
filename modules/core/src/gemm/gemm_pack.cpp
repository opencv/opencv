// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "gemm_engine.hpp"
#include "gemm_kernels.simd.hpp"

namespace cv { namespace gemm_engine {

#if CV_GEMM_ENGINE_NEON
// Panel width of packed B: 2*VL when the SME kernels will read it, otherwise the NEON kernel's 12.
static int packedPanelWidth()
{
    const int vl = smeVectorLength();
    return vl > 0 ? 2 * vl : (int)GEMM_NR;
}
#endif

size_t packedPayloadSize(int n, int k)
{
#if CV_GEMM_ENGINE_NEON
    return (size_t)roundUp(n, packedPanelWidth()) * k * sizeof(float);
#else
    return (size_t)n * k * sizeof(float);
#endif
}

void packB(int n, int k, const float* B, size_t ldb0, size_t ldb1, void* packedB, bool multiThread)
{
    PackedBHeader* hdr = (PackedBHeader*)packedB;
    float* dst = (float*)((uchar*)packedB + PACKED_HEADER_SIZE);
#if CV_GEMM_ENGINE_NEON
    const int W = packedPanelWidth(), nPanels = (n + W - 1) / W;
    auto pack = [&](const Range& r) {
        const int j0 = r.start * W;
        CV_CPU_OPTIMIZATION_NAMESPACE::packBPanels(std::min(r.end * W, n) - j0, k, B + j0 * ldb1, ldb0, ldb1, W,
                                                   dst + (size_t)j0 * k);
    };
    if (multiThread)
        parallel_for_(Range(0, nPanels), pack, (double)n * k * (1.0 / 65536));
    else
        pack(Range(0, nPanels));
    hdr->format = PACKED_PANELS;
    hdr->width = W;
    hdr->sme = W != GEMM_NR;
#else
    CV_UNUSED(multiThread);
    for (int p = 0; p < k; p++)
        for (int j = 0; j < n; j++)
            dst[(size_t)p * n + j] = B[p * ldb0 + j * ldb1];
    hdr->format = PACKED_PLAIN;
    hdr->width = 0;
    hdr->sme = 0;
#endif
    hdr->magic = PACKED_MAGIC;
    hdr->n = n;
    hdr->k = k;
}

const PackedBHeader& checkPackedB(const void* packedB, int n, int k)
{
    CV_Assert(packedB);
    const PackedBHeader& hdr = *(const PackedBHeader*)packedB;
    CV_CheckEQ((int)hdr.magic, (int)PACKED_MAGIC, "packedB was not filled by cv::hal::gemm32fPackB");
    CV_CheckEQ(hdr.n, n, "packedB was packed for a different n");
    CV_CheckEQ(hdr.k, k, "packedB was packed for a different k");
    return hdr;
}

}} // cv::gemm_engine
