// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "perf_precomp.hpp"
#include "opencv2/core/hal/hal.hpp"

namespace opencv_test
{
using namespace perf;

// m, n, k; B is packed once, outside the timed loop.
typedef perf::TestBaseWithParam<tuple<int, int, int> > GemmPackedTest;

PERF_TEST_P(GemmPackedTest, gemm32fPacked,
            testing::Values(
                make_tuple(1, 1000, 1024),
                make_tuple(8, 768, 768),
                make_tuple(197, 768, 768),
                make_tuple(196, 256, 1152),
                make_tuple(1024, 1024, 1024)
            ))
{
    const int m = get<0>(GetParam()), n = get<1>(GetParam()), k = get<2>(GetParam());
    Mat A(m, k, CV_32F), B(k, n, CV_32F), C(m, n, CV_32F);
    declare.in(A, B, WARMUP_RNG).out(C);

    std::vector<uchar> packed(hal::gemm32fPackBSize(n, k, 0));
    hal::gemm32fPackB(B.ptr<float>(), B.step, n, k, 0, packed.data());

    TEST_CYCLE() hal::gemm32fPacked(A.ptr<float>(), A.step, packed.data(), C.ptr<float>(), C.step,
                                    m, n, k, 1.f, 0.f, 0);

    SANITY_CHECK_NOTHING();
}

// batch, m, n, k, and whether B is transposed. Mostly attention shapes, plus a batch of one.
typedef perf::TestBaseWithParam<tuple<tuple<int, int, int, int>, bool> > GemmBatchTest;

PERF_TEST_P(GemmBatchTest, gemm32fBatch,
            testing::Combine(
                testing::Values(
                    make_tuple(12, 197, 197, 64),
                    make_tuple(12, 197, 64, 197),
                    make_tuple(16, 50, 50, 64),
                    make_tuple(32, 1, 64, 128),
                    make_tuple(1, 512, 512, 512)
                ),
                testing::Bool()
            ))
{
    const int batch = get<0>(get<0>(GetParam())), m = get<1>(get<0>(GetParam()));
    const int n = get<2>(get<0>(GetParam())), k = get<3>(get<0>(GetParam()));
    const bool transB = get<1>(GetParam());
    const int bRows = transB ? n : k, bCols = transB ? k : n;
    Mat A(batch * m, k, CV_32F), B(batch * bRows, bCols, CV_32F), C(batch * m, n, CV_32F);
    declare.in(A, B, WARMUP_RNG).out(C);

    std::vector<size_t> aOfs(batch), bOfs(batch), cOfs(batch);
    for (int i = 0; i < batch; i++)
    {
        aOfs[i] = (size_t)i * m * k;
        bOfs[i] = (size_t)i * bRows * bCols;
        cOfs[i] = (size_t)i * m * n;
    }
    const int flags = transB ? CV_HAL_GEMM_2_T : 0;

    TEST_CYCLE() hal::gemm32fBatch(batch, A.ptr<float>(), aOfs.data(), A.step, B.ptr<float>(), bOfs.data(), B.step,
                                   C.ptr<float>(), cOfs.data(), C.step, m, n, k, 1.f, 0.f, flags);

    SANITY_CHECK_NOTHING();
}

} // namespace
