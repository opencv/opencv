#include "perf_precomp.hpp"

using namespace std;
using namespace testing;

#define GPU_DENOISING_IMAGE_SIZES testing::Values(perf::szVGA, perf::sz720p)

//////////////////////////////////////////////////////////////////////
// BilateralFilter

DEF_PARAM_TEST(Sz_Depth_Cn_KernelSz, cv::Size, MatDepth, MatCn, int);

PERF_TEST_P(Sz_Depth_Cn_KernelSz, Denoising_BilateralFilter,
            Combine(GPU_DENOISING_IMAGE_SIZES,
                    Values(CV_8U, CV_32F),
                    GPU_CHANNELS_1_3,
                    Values(3, 5, 9)))
{
    declare.time(60.0);

    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);
    const int channels = GET_PARAM(2);
    const int kernel_size = GET_PARAM(3);

    const float sigma_color = 7;
    const float sigma_spatial = 5;
    const int borderMode = cv::BORDER_REFLECT101;

    const int type = CV_MAKE_TYPE(depth, channels);

    cv::Mat src(size, type);
    declare.in(src, WARMUP_RNG);

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat dst;

        TEST_CYCLE() cv::gpu::bilateralFilter(d_src, dst, kernel_size, sigma_color, sigma_spatial, borderMode);

        GPU_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::bilateralFilter(src, dst, kernel_size, sigma_color, sigma_spatial, borderMode);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// nonLocalMeans

DEF_PARAM_TEST(Sz_Depth_Cn_WinSz_BlockSz, cv::Size, MatDepth, MatCn, int, int);

PERF_TEST_P(Sz_Depth_Cn_WinSz_BlockSz, Denoising_NonLocalMeans,
            Combine(GPU_DENOISING_IMAGE_SIZES,
                    Values<MatDepth>(CV_8U),
                    GPU_CHANNELS_1_3,
                    Values(21),
                    Values(5)))
{
    declare.time(60.0);

    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);
    const int channels = GET_PARAM(2);
    const int search_widow_size = GET_PARAM(3);
    const int block_size = GET_PARAM(4);

    const float h = 10;
    const int borderMode = cv::BORDER_REFLECT101;

    const int type = CV_MAKE_TYPE(depth, channels);

    cv::Mat src(size, type);
    declare.in(src, WARMUP_RNG);

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat dst;

        TEST_CYCLE() cv::gpu::nonLocalMeans(d_src, dst, h, search_widow_size, block_size, borderMode);

        GPU_SANITY_CHECK(dst);
    }
    else
    {
        FAIL_NO_CPU();
    }
}


//////////////////////////////////////////////////////////////////////
// fastNonLocalMeans

DEF_PARAM_TEST(Sz_Depth_Cn_WinSz_BlockSz, cv::Size, MatDepth, MatCn, int, int);

PERF_TEST_P(Sz_Depth_Cn_WinSz_BlockSz, Denoising_FastNonLocalMeans,
            Combine(GPU_DENOISING_IMAGE_SIZES,
                    Values<MatDepth>(CV_8U),
                    GPU_CHANNELS_1_3,
                    Values(21),
                    Values(7)))
{
    declare.time(60.0);

    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);
    const int search_widow_size = GET_PARAM(2);
    const int block_size = GET_PARAM(3);

    const float h = 10;
    const int type = CV_MAKE_TYPE(depth, 1);

    cv::Mat src(size, type);
    declare.in(src, WARMUP_RNG);

    if (PERF_RUN_GPU())
    {
        cv::gpu::FastNonLocalMeansDenoising fnlmd;

        const cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat dst;

        TEST_CYCLE() fnlmd.simpleMethod(d_src, dst, h, search_widow_size, block_size);

        GPU_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::fastNlMeansDenoising(src, dst, h, block_size, search_widow_size);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// fastNonLocalMeans (colored)

DEF_PARAM_TEST(Sz_Depth_WinSz_BlockSz, cv::Size, MatDepth, int, int);

PERF_TEST_P(Sz_Depth_WinSz_BlockSz, Denoising_FastNonLocalMeansColored,
            Combine(GPU_DENOISING_IMAGE_SIZES,
                    Values<MatDepth>(CV_8U),
                    Values(21),
                    Values(7)))
{
    declare.time(60.0);

    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);
    const int search_widow_size = GET_PARAM(2);
    const int block_size = GET_PARAM(3);

    const float h = 10;
    const int type = CV_MAKE_TYPE(depth, 3);

    cv::Mat src(size, type);
    declare.in(src, WARMUP_RNG);

    if (PERF_RUN_GPU())
    {
        cv::gpu::FastNonLocalMeansDenoising fnlmd;

        const cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat dst;

        TEST_CYCLE() fnlmd.labMethod(d_src, dst, h, h, search_widow_size, block_size);

        GPU_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::fastNlMeansDenoisingColored(src, dst, h, h, block_size, search_widow_size);

        CPU_SANITY_CHECK(dst);
    }
}
