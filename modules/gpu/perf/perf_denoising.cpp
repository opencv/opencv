#include "perf_precomp.hpp"

using namespace std;
using namespace testing;


//////////////////////////////////////////////////////////////////////
// BilateralFilter

DEF_PARAM_TEST(Sz_Depth_Cn_KernelSz, cv::Size, MatDepth , int, int);

PERF_TEST_P(Sz_Depth_Cn_KernelSz, Denoising_BilateralFilter, 
            Combine(GPU_TYPICAL_MAT_SIZES, Values(CV_8U, CV_16U, CV_32F), GPU_CHANNELS_1_3_4, Values(3, 5, 9)))
{
    declare.time(30.0);

    cv::Size size = GET_PARAM(0);
    int depth = GET_PARAM(1);
    int channels = GET_PARAM(2);
    int kernel_size = GET_PARAM(3);

    float sigma_color = 7;
    float sigma_spatial = 5;
    int borderMode = cv::BORDER_REFLECT101;

    int type = CV_MAKE_TYPE(depth, channels);

    cv::Mat src(size, type);
    fillRandom(src);

     if (runOnGpu)
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        cv::gpu::bilateralFilter(d_src, d_dst, kernel_size, sigma_color, sigma_spatial, borderMode);

        TEST_CYCLE()
        {
            cv::gpu::bilateralFilter(d_src, d_dst, kernel_size, sigma_color, sigma_spatial, borderMode);
        }
    }
    else
    {
        cv::Mat dst;

        cv::bilateralFilter(src, dst, kernel_size, sigma_color, sigma_spatial, borderMode);

        TEST_CYCLE()
        {
            cv::bilateralFilter(src, dst, kernel_size, sigma_color, sigma_spatial, borderMode);
        }
    }
}


//////////////////////////////////////////////////////////////////////
// nonLocalMeans

DEF_PARAM_TEST(Sz_Depth_Cn_WinSz_BlockSz, cv::Size, MatDepth , int, int, int);

PERF_TEST_P(Sz_Depth_Cn_WinSz_BlockSz, Denoising_NonLocalMeans, 
            Combine(GPU_TYPICAL_MAT_SIZES, Values<MatDepth>(CV_8U), Values(1), Values(21), Values(5, 7)))
{
    declare.time(30.0);

    cv::Size size = GET_PARAM(0);
    int depth = GET_PARAM(1);
    int channels = GET_PARAM(2);
    
    int search_widow_size = GET_PARAM(3);
    int block_size = GET_PARAM(4);

    float h = 10;
    int borderMode = cv::BORDER_REFLECT101;
    
    int type = CV_MAKE_TYPE(depth, channels);

    cv::Mat src(size, type);
    fillRandom(src);

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;

        cv::gpu::nonLocalMeans(d_src, d_dst, h, search_widow_size, block_size, borderMode);

        TEST_CYCLE()
        {
            cv::gpu::nonLocalMeans(d_src, d_dst, h, search_widow_size, block_size, borderMode);
        }
    }
    else
    {
        FAIL();
    }
}