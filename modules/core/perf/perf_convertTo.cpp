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

// Scale + shift (matches IPP perf coverage: convertTo with a nonzero beta/shift)
PERF_TEST_P( Size_DepthSrc_DepthDst_Channels_alpha, convertTo_scale_shift,
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
    double beta = 5.0;

    Mat src(sz, CV_MAKETYPE(depthSrc, channels));
    randu(src, 0, 255);
    Mat dst(sz, CV_MAKETYPE(depthDst, channels));

    int maxValue = 255;
    int runs = (sz.width <= 640) ? 8 : 1;
    TEST_CYCLE_MULTIRUN(runs) src.convertTo(dst, depthDst, alpha, beta);

    double eps = depthSrc <= CV_32S && (depthDst <= CV_32S || depthDst == CV_64F) ? 1e-12 : (FLT_EPSILON * maxValue);
    eps = eps * std::max(1.0, fabs(alpha)) + fabs(beta);
    SANITY_CHECK_NOTHING();
}

// In-place convertTo with scale+shift (matches IPP ippConvertTo_ScaleInplace:
// dst.convertTo(dst, sameType, scale, shift)). In-place requires src type==dst type.
typedef tuple<Size, MatType, double> Size_Depth_alpha_t;
typedef perf::TestBaseWithParam<Size_Depth_alpha_t> Size_Depth_alpha;

PERF_TEST_P( Size_Depth_alpha, convertTo_scale_shift_inplace,
             testing::Combine
             (
                 testing::Values(szVGA, sz1080p),
                 testing::Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F),
                 testing::Values(1.0, 1./255)
             )
           )
{
    Size sz = get<0>(GetParam());
    int depth = get<1>(GetParam());
    double alpha = get<2>(GetParam());
    double beta = 5.0;

    Mat src(sz, CV_MAKETYPE(depth, 1));
    randu(src, 0, 255);
    Mat dst(sz, CV_MAKETYPE(depth, 1));

    int runs = (sz.width <= 640) ? 8 : 1;
    while (next())
    {
        src.copyTo(dst);
        startTimer();
        for (int i = 0; i < runs; ++i)
            dst.convertTo(dst, depth, alpha, beta);
        stopTimer();
    }

    SANITY_CHECK_NOTHING();
}

} // namespace
