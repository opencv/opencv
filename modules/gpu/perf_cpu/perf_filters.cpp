#include "perf_cpu_precomp.hpp"

#ifdef HAVE_CUDA

//////////////////////////////////////////////////////////////////////
// BoxFilter

GPU_PERF_TEST(BoxFilter, cv::gpu::DeviceInfo, cv::Size, perf::MatType, int)
{
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);
    int ksize = GET_PARAM(3);

    cv::Mat src(size, type);

    declare.in(src, WARMUP_RNG);

    cv::Mat dst(src.size(), src.type());

    cv::Ptr<cv::FilterEngine> filter = cv::createBoxFilter(type, type, cv::Size(ksize, ksize));

    TEST_CYCLE()
    {
        filter->apply(src, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Filter, BoxFilter, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES,
                        testing::Values(CV_8UC1, CV_8UC4),
                        testing::Values(3, 5)));

//////////////////////////////////////////////////////////////////////
// MorphologyFilter

GPU_PERF_TEST(MorphologyFilter, cv::gpu::DeviceInfo, cv::Size, perf::MatType, MorphOp, int)
{
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);
    int op = GET_PARAM(3);
    int ksize = GET_PARAM(4);

    cv::Mat src(size, type);

    declare.in(src, WARMUP_RNG);

    cv::Mat dst(src.size(), src.type());

    cv::Ptr<cv::FilterEngine> filter = cv::createMorphologyFilter(op, type, cv::Mat::ones(ksize, ksize, CV_8U));

    TEST_CYCLE()
    {
        filter->apply(src, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Filter, MorphologyFilter, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES,
                        testing::Values(CV_8UC1, CV_8UC4),
                        testing::Values((int) cv::MORPH_ERODE, (int) cv::MORPH_DILATE),
                        testing::Values(3, 5)));

//////////////////////////////////////////////////////////////////////
// LinearFilter

GPU_PERF_TEST(LinearFilter, cv::gpu::DeviceInfo, cv::Size, perf::MatType, int)
{
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);
    int ksize = GET_PARAM(3);

    cv::Mat src(size, type);

    declare.in(src, WARMUP_RNG);

    cv::Mat dst(src.size(), src.type());

    cv::Ptr<cv::FilterEngine> filter = cv::createLinearFilter(type, type, cv::Mat::ones(ksize, ksize, CV_8U));

    declare.time(1.0);

    TEST_CYCLE()
    {
        filter->apply(src, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Filter, LinearFilter, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES,
                        testing::Values(CV_8UC1, CV_8UC4, CV_32FC1),
                        testing::Values(3, 5, 7, 9)));

//////////////////////////////////////////////////////////////////////
// SeparableLinearFilter

GPU_PERF_TEST(SeparableLinearFilter, cv::gpu::DeviceInfo, cv::Size, perf::MatType, int)
{
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);
    int ksize = GET_PARAM(3);

    cv::Mat src(size, type);

    declare.in(src, WARMUP_RNG);

    cv::Mat dst(src.size(), src.type());

    cv::Mat kernel = cv::getGaussianKernel(ksize, 0.5, CV_32F);
    cv::Ptr<cv::FilterEngine> filter = cv::createSeparableLinearFilter(type, type, kernel, kernel);

    declare.time(1.0);

    TEST_CYCLE()
    {
        filter->apply(src, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Filter, SeparableLinearFilter, testing::Combine(
                        ALL_DEVICES,
                        GPU_TYPICAL_MAT_SIZES,
                        testing::Values(CV_8UC1, CV_8UC4, CV_32FC1),
                        testing::Values(3, 5, 7, 9, 11, 13, 15)));

#endif
