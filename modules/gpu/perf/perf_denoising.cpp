#include "perf_precomp.hpp"

using namespace std;
using namespace testing;

#define GPU_DENOISING_IMAGE_SIZES testing::Values(perf::szVGA, perf::szXGA, perf::sz720p, perf::sz1080p)


//////////////////////////////////////////////////////////////////////
// BilateralFilter

DEF_PARAM_TEST(Sz_Depth_Cn_KernelSz, cv::Size, MatDepth, MatCn, int);

PERF_TEST_P(Sz_Depth_Cn_KernelSz, Denoising_BilateralFilter, 
            Combine(GPU_DENOISING_IMAGE_SIZES, Values(CV_8U, CV_32F), GPU_CHANNELS_1_3, Values(3, 5, 9)))
{
    declare.time(60.0);

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

DEF_PARAM_TEST(Sz_Depth_Cn_WinSz_BlockSz, cv::Size, MatDepth, MatCn, int, int);

PERF_TEST_P(Sz_Depth_Cn_WinSz_BlockSz, Denoising_NonLocalMeans, 
            Combine(GPU_DENOISING_IMAGE_SIZES, Values<MatDepth>(CV_8U), GPU_CHANNELS_1_3, Values(21), Values(5, 7)))
{
    declare.time(60.0);

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


//////////////////////////////////////////////////////////////////////
// fastNonLocalMeans

DEF_PARAM_TEST(Sz_Depth_Cn_WinSz_BlockSz, cv::Size, MatDepth, MatCn, int, int);

PERF_TEST_P(Sz_Depth_Cn_WinSz_BlockSz, Denoising_FastNonLocalMeans, 
            Combine(GPU_DENOISING_IMAGE_SIZES, Values<MatDepth>(CV_8U), GPU_CHANNELS_1_3, Values(21), Values(7)))
{
    declare.time(150.0);
    
    cv::Size size = GET_PARAM(0);
    int depth = GET_PARAM(1);    
    
    int search_widow_size = GET_PARAM(2);
    int block_size = GET_PARAM(3);
    
    float h = 10;           
    int type = CV_MAKE_TYPE(depth, 1);    

    cv::Mat src(size, type);
    fillRandom(src);

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;
        cv::gpu::FastNonLocalMeansDenoising fnlmd;

        fnlmd.simpleMethod(d_src, d_dst, h, search_widow_size, block_size); 

        TEST_CYCLE()
        {
            fnlmd.simpleMethod(d_src, d_dst, h, search_widow_size, block_size); 
        }
    }
    else
    {
        cv::Mat dst;
        cv::fastNlMeansDenoising(src, dst, h, block_size, search_widow_size);

        TEST_CYCLE()
        {
            cv::fastNlMeansDenoising(src, dst, h, block_size, search_widow_size);
        }
    }
}

//////////////////////////////////////////////////////////////////////
// fastNonLocalMeans (colored)

DEF_PARAM_TEST(Sz_Depth_WinSz_BlockSz, cv::Size, MatDepth, int, int);

PERF_TEST_P(Sz_Depth_WinSz_BlockSz, Denoising_FastNonLocalMeansColored, 
            Combine(GPU_DENOISING_IMAGE_SIZES, Values<MatDepth>(CV_8U), Values(21), Values(7)))
{
    declare.time(350.0);
    
    cv::Size size = GET_PARAM(0);
    int depth = GET_PARAM(1);
    
    int search_widow_size = GET_PARAM(2);
    int block_size = GET_PARAM(3);

    float h = 10;       
    int type = CV_MAKE_TYPE(depth, 3);

    cv::Mat src(size, type);
    fillRandom(src);

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_dst;
        cv::gpu::FastNonLocalMeansDenoising fnlmd;

        fnlmd.labMethod(d_src, d_dst, h, h, search_widow_size, block_size); 

        TEST_CYCLE()
        {
            fnlmd.labMethod(d_src, d_dst, h, h, search_widow_size, block_size); 
        }
    }
    else
    {
        cv::Mat dst;
        cv::fastNlMeansDenoisingColored(src, dst, h, h, block_size, search_widow_size);

        TEST_CYCLE()
        {
            cv::fastNlMeansDenoisingColored(src, dst, h, h, block_size, search_widow_size);
        }
    }
}