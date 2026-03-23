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

        SANITY_CHECK(dst, 1e-3);
    };
};

PERF_TEST_P(GemmTest, gemmTiny,
            testing::Combine(
                testing::Values(
                    make_tuple(2, 2, 2),
                    make_tuple(3, 3, 3),
                    make_tuple(4, 4, 4)
                ),
                testing::Values(CV_32FC1, CV_64FC1, CV_32FC2, CV_64FC2),
                testing::Values(0, (int)GEMM_1_T, (int)GEMM_2_T,
                                (int)(GEMM_1_T | GEMM_2_T))
            ))
{
    GemmTest::runGemmTest(GetParam());
}

PERF_TEST_P(GemmTest, gemmSmall,
            testing::Combine(
                testing::Values(
                    make_tuple(4, 8, 16),
                    make_tuple(8, 8, 8),
                    make_tuple(8, 16, 32),
                    make_tuple(16, 16, 16),
                    make_tuple(32, 32, 32),
                    make_tuple(64, 64, 64),
                    make_tuple(128, 128, 128)
                ),
                testing::Values(CV_32FC1, CV_64FC1, CV_32FC2, CV_64FC2),
                testing::Values(0, (int)GEMM_1_T, (int)GEMM_2_T,
                                (int)(GEMM_1_T | GEMM_2_T))
            ))
{
    GemmTest::runGemmTest(GetParam());
}

PERF_TEST_P(GemmTest, gemmSquare,
            testing::Combine(
                testing::Values(
                    make_tuple(256, 256, 256),
                    make_tuple(512, 512, 512),
                    make_tuple(1024, 1024, 1024)
                ),
                testing::Values(CV_32FC1, CV_64FC1, CV_32FC2, CV_64FC2),
                testing::Values(0, (int)GEMM_1_T, (int)GEMM_2_T,
                                (int)(GEMM_1_T | GEMM_2_T))
            ))
{
    GemmTest::runGemmTest(GetParam());
}

PERF_TEST_P(GemmTest, gemmRect,
            testing::Combine(
                testing::Values(
                    // Tall output (M >> N)
                    make_tuple(1024, 64, 256),
                    make_tuple(1024, 256, 512),
                    make_tuple(512, 32, 512),
                    make_tuple(512, 128, 256),
                    // Wide output (N >> M)
                    make_tuple(256, 1024, 512),
                    make_tuple(128, 512, 256),
                    make_tuple(64, 1024, 256),
                    make_tuple(32, 512, 512)
                ),
                testing::Values(CV_32FC1, CV_64FC1, CV_32FC2, CV_64FC2),
                testing::Values(0, (int)GEMM_1_T, (int)GEMM_2_T,
                                (int)(GEMM_1_T | GEMM_2_T))
            ))
{
    GemmTest::runGemmTest(GetParam());
}

PERF_TEST_P(GemmTest, gemmM1,
            testing::Combine(
                testing::Values(
                    make_tuple(1, 20, 2500),
                    make_tuple(1, 64, 2500),
                    make_tuple(1, 80, 10000)
                ),
                testing::Values(CV_32FC1, CV_64FC1, CV_32FC2, CV_64FC2),
                testing::Values(0, (int)GEMM_1_T, (int)GEMM_2_T,
                                (int)(GEMM_1_T | GEMM_2_T))
            ))
{
    GemmTest::runGemmTest(GetParam());
}

PERF_TEST_P(GemmTest, gemmN1,
            testing::Combine(
                testing::Values(
                    make_tuple(256, 1, 256),
                    make_tuple(1024, 1, 1024)
                ),
                testing::Values(CV_32FC1, CV_64FC1, CV_32FC2, CV_64FC2),
                testing::Values(0, (int)GEMM_1_T, (int)GEMM_2_T,
                                (int)(GEMM_1_T | GEMM_2_T))
            ))
{
    GemmTest::runGemmTest(GetParam());
}

} // namespace
