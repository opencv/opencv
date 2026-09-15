#include "perf_precomp.hpp"

namespace opencv_test
{
using namespace perf;

typedef tuple<Size, MatType, MatType, int, double> Size_DepthSrc_DepthDst_Channels_alpha_t;
typedef perf::TestBaseWithParam<Size_DepthSrc_DepthDst_Channels_alpha_t> Size_DepthSrc_DepthDst_Channels_alpha;

PERF_TEST_P( Size_DepthSrc_DepthDst_Channels_alpha, convertTo,
             testing::Combine
             (
                 testing::Values(szVGA, sz1080p),
                 testing::Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F),
                 testing::Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F),
                 testing::Values(1, 4),
                 testing::Values(1.0, 1./255)
             )
           )
{
    Size sz = get<0>(GetParam());
    int depthSrc = get<1>(GetParam());
    int depthDst = get<2>(GetParam());
    int channels = get<3>(GetParam());
    double alpha = get<4>(GetParam());

    int maxValue = 255;

    Mat src(sz, CV_MAKETYPE(depthSrc, channels));
    randu(src, 0, maxValue);
    Mat dst(sz, CV_MAKETYPE(depthDst, channels));

    int runs = (sz.width <= 640) ? 8 : 1;
    TEST_CYCLE_MULTIRUN(runs) src.convertTo(dst, depthDst, alpha);

    double eps = depthSrc <= CV_32S && (depthDst <= CV_32S || depthDst == CV_64F) ? 1e-12 : (FLT_EPSILON * maxValue);
    eps = eps * std::max(1.0, fabs(alpha));
    SANITY_CHECK(dst, eps);
}

// alpha param covers both the identity and scale FP8 kernels.
// Two distributions tracked: well-scaled (fast path) vs near-zero (fallback path).
typedef tuple<Size, MatType, double> Size_Fp8Depth_Alpha_t;
typedef perf::TestBaseWithParam<Size_Fp8Depth_Alpha_t> Size_Fp8Depth_Alpha;

PERF_TEST_P( Size_Fp8Depth_Alpha, convertToFp8_wellScaled,
             testing::Combine
             (
                 testing::Values(szVGA, sz1080p),
                 testing::Values(CV_8F_E4M3FN, CV_8F_E4M3FNUZ),
                 testing::Values(1.0, 0.1)
             )
           )
{
    Size sz = get<0>(GetParam());
    int fp8depth = get<1>(GetParam());
    double alpha = get<2>(GetParam());

    Mat src(sz, CV_32FC1);
    randu(src, -8.0, 8.0);   // mostly normal-range -> mostly the fast path
    Mat dst(sz, fp8depth);

    TEST_CYCLE() src.convertTo(dst, fp8depth, alpha, 0.0);
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P( Size_Fp8Depth_Alpha, convertToFp8_nearZero,
             testing::Combine
             (
                 testing::Values(szVGA, sz1080p),
                 testing::Values(CV_8F_E4M3FN, CV_8F_E4M3FNUZ),
                 testing::Values(1.0, 0.1)
             )
           )
{
    Size sz = get<0>(GetParam());
    int fp8depth = get<1>(GetParam());
    double alpha = get<2>(GetParam());

    Mat src(sz, CV_32FC1);
    randu(src, -0.006, 0.006);   // below E4M3's smallest normal (2^-6) -> mostly the fallback path
    Mat dst(sz, fp8depth);

    TEST_CYCLE() src.convertTo(dst, fp8depth, alpha, 0.0);
    SANITY_CHECK_NOTHING();
}

} // namespace
