#include "perf_precomp.hpp"

namespace opencv_test
{
using namespace perf;

CV_FLAGS(GemmFlag, 0, GEMM_1_T, GEMM_2_T, GEMM_3_T)

typedef tuple<tuple<int, int, int>, MatType, GemmFlag> GemmTestParams_t;
class GemmTest : public perf::TestBaseWithParam<GemmTestParams_t>
{
    public:
    void runGemmTest(const GemmTestParams_t& params)
    {
        int M = get<0>(get<0>(params));
        int N = get<1>(get<0>(params));
        int K = get<2>(get<0>(params));
        int type = get<1>(params);
        int flags = get<2>(params);

        Size aSize = (flags & GEMM_1_T) ? Size(M, K) : Size(K, M);
        Size bSize = (flags & GEMM_2_T) ? Size(K, N) : Size(N, K);

        Mat src1(aSize, type), src2(bSize, type), src3(M, N, type), dst(M, N, type);
        declare.in(src1, src2, src3, WARMUP_RNG).out(dst);

        TEST_CYCLE() cv::gemm(src1, src2, 1.0, src3, 1.0, dst, flags);

        if (dst.total() * dst.channels() < 26)
            SANITY_CHECK_NOTHING();
        else
            SANITY_CHECK(dst, (CV_MAT_DEPTH(type) == CV_32F) ? 1e-4 : 1e-6, ERROR_RELATIVE);
    };
};

// Sparse coverage: exercise tiny/small/rectangular shapes and m=1/n=1 edge cases.
// Large square sizes (640+) are covered by opencl/perf_gemm.cpp on the CPU perf tool.
PERF_TEST_P(GemmTest, gemmTiny,
            testing::Combine(
                testing::Values(
                    make_tuple(2, 2, 2),
                    make_tuple(3, 3, 3)
                ),
                testing::Values(CV_32FC1, CV_64FC1),
                testing::Values(0, (int)GEMM_1_T, (int)GEMM_2_T,
                                (int)(GEMM_1_T | GEMM_2_T))
            ))
{
    GemmTest::runGemmTest(GetParam());
}

PERF_TEST_P(GemmTest, gemmSmall,
            testing::Combine(
                testing::Values(
                    make_tuple(8, 8, 8),
                    make_tuple(32, 32, 32),
                    make_tuple(64, 64, 64),
                    make_tuple(8, 16, 32)
                ),
                testing::Values(CV_32FC1),
                testing::Values(0, (int)GEMM_2_T)
            ))
{
    GemmTest::runGemmTest(GetParam());
}

PERF_TEST_P(GemmTest, gemmSquare,
            testing::Combine(
                testing::Values(
                    make_tuple(256, 256, 256),
                    make_tuple(512, 512, 512)
                ),
                testing::Values(CV_32FC1),
                testing::Values(0)
            ))
{
    GemmTest::runGemmTest(GetParam());
}

PERF_TEST_P(GemmTest, gemmRect,
            testing::Combine(
                testing::Values(
                    make_tuple(1024, 64, 256),
                    make_tuple(256, 1024, 512)
                ),
                testing::Values(CV_32FC1),
                testing::Values(0)
            ))
{
    GemmTest::runGemmTest(GetParam());
}

PERF_TEST_P(GemmTest, gemmM1,
            testing::Combine(
                testing::Values(
                    make_tuple(1, 64, 2500)
                ),
                testing::Values(CV_32FC1),
                testing::Values(0, (int)GEMM_1_T)
            ))
{
    GemmTest::runGemmTest(GetParam());
}

PERF_TEST_P(GemmTest, gemmN1,
            testing::Combine(
                testing::Values(
                    make_tuple(256, 1, 256)
                ),
                testing::Values(CV_32FC1),
                testing::Values(0)
            ))
{
    GemmTest::runGemmTest(GetParam());
}

} // namespace
