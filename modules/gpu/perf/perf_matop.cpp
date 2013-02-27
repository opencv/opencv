#include "perf_precomp.hpp"

using namespace std;
using namespace testing;

//////////////////////////////////////////////////////////////////////
// SetTo

PERF_TEST_P(Sz_Depth_Cn, MatOp_SetTo,
            Combine(GPU_TYPICAL_MAT_SIZES,
                    Values(CV_8U, CV_16U, CV_32F, CV_64F),
                    GPU_CHANNELS_1_3_4))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);
    const int channels = GET_PARAM(2);

    const int type = CV_MAKE_TYPE(depth, channels);

    const cv::Scalar val(1, 2, 3, 4);

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat dst(size, type);

        TEST_CYCLE() dst.setTo(val);

        GPU_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst(size, type);

        TEST_CYCLE() dst.setTo(val);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// SetToMasked

PERF_TEST_P(Sz_Depth_Cn, MatOp_SetToMasked,
            Combine(GPU_TYPICAL_MAT_SIZES,
                    Values(CV_8U, CV_16U, CV_32F, CV_64F),
                    GPU_CHANNELS_1_3_4))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);
    const int channels = GET_PARAM(2);

    const int type = CV_MAKE_TYPE(depth, channels);

    cv::Mat src(size, type);
    cv::Mat mask(size, CV_8UC1);
    declare.in(src, mask, WARMUP_RNG);

    const cv::Scalar val(1, 2, 3, 4);

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat dst(src);
        const cv::gpu::GpuMat d_mask(mask);

        TEST_CYCLE() dst.setTo(val, d_mask);

        GPU_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst = src;

        TEST_CYCLE() dst.setTo(val, mask);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// CopyToMasked

PERF_TEST_P(Sz_Depth_Cn, MatOp_CopyToMasked,
            Combine(GPU_TYPICAL_MAT_SIZES,
                    Values(CV_8U, CV_16U, CV_32F, CV_64F),
                    GPU_CHANNELS_1_3_4))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);
    const int channels = GET_PARAM(2);

    const int type = CV_MAKE_TYPE(depth, channels);

    cv::Mat src(size, type);
    cv::Mat mask(size, CV_8UC1);
    declare.in(src, mask, WARMUP_RNG);

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_src(src);
        const cv::gpu::GpuMat d_mask(mask);
        cv::gpu::GpuMat dst(d_src.size(), d_src.type(), cv::Scalar::all(0));

        TEST_CYCLE() d_src.copyTo(dst, d_mask);

        GPU_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst(src.size(), src.type(), cv::Scalar::all(0));

        TEST_CYCLE() src.copyTo(dst, mask);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// ConvertTo

DEF_PARAM_TEST(Sz_2Depth, cv::Size, MatDepth, MatDepth);

PERF_TEST_P(Sz_2Depth, MatOp_ConvertTo,
            Combine(GPU_TYPICAL_MAT_SIZES,
                    Values(CV_8U, CV_16U, CV_32F, CV_64F),
                    Values(CV_8U, CV_16U, CV_32F, CV_64F)))
{
    const cv::Size size = GET_PARAM(0);
    const int depth1 = GET_PARAM(1);
    const int depth2 = GET_PARAM(2);

    cv::Mat src(size, depth1);
    declare.in(src, WARMUP_RNG);

    const double a = 0.5;
    const double b = 1.0;

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat dst;

        TEST_CYCLE() d_src.convertTo(dst, depth2, a, b);

        GPU_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() src.convertTo(dst, depth2, a, b);

        CPU_SANITY_CHECK(dst);
    }
}
