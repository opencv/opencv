#include "perf_precomp.hpp"

#ifdef HAVE_CUDA

//////////////////////////////////////////////////////////////////////
// Merge

GPU_PERF_TEST(Merge, cv::gpu::DeviceInfo, cv::Size, perf::MatType)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::gpu::setDevice(devInfo.deviceID());

    const int num_channels = 4;

    std::vector<cv::gpu::GpuMat> src(num_channels);
    for (int i = 0; i < num_channels; ++i)
        src[i] = cv::gpu::GpuMat(size, type, cv::Scalar::all(i)); 

    cv::gpu::GpuMat dst;

    TEST_CYCLE(100)
    {
        cv::gpu::merge(src, dst);
    }
}

INSTANTIATE_TEST_CASE_P(MatOp, Merge, testing::Combine(
                        ALL_DEVICES, 
                        GPU_TYPICAL_MAT_SIZES, 
                        testing::Values(CV_8UC1, CV_16UC1, CV_32FC1)));

//////////////////////////////////////////////////////////////////////
// Split

GPU_PERF_TEST(Split, cv::gpu::DeviceInfo, cv::Size, perf::MatType)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::gpu::setDevice(devInfo.deviceID());

    const int num_channels = 4;

    cv::gpu::GpuMat src(size, CV_MAKETYPE(type, num_channels), cv::Scalar(1, 2, 3, 4));

    std::vector<cv::gpu::GpuMat> dst(num_channels);
    for (int i = 0; i < num_channels; ++i)
        dst[i] = cv::gpu::GpuMat(size, type); 

    TEST_CYCLE(100)
    {
        cv::gpu::split(src, dst);
    }
}

INSTANTIATE_TEST_CASE_P(MatOp, Split, testing::Combine(
                        ALL_DEVICES, 
                        GPU_TYPICAL_MAT_SIZES, 
                        testing::Values(CV_8UC1, CV_16UC1, CV_32FC1)));

//////////////////////////////////////////////////////////////////////
// SetTo

GPU_PERF_TEST(SetTo, cv::gpu::DeviceInfo, cv::Size, perf::MatType)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::gpu::GpuMat src(size, type);
    cv::Scalar val(1, 2, 3, 4);

    TEST_CYCLE(100)
    {
        src.setTo(val);
    }
}

INSTANTIATE_TEST_CASE_P(MatOp, SetTo, testing::Combine(
                        ALL_DEVICES, 
                        GPU_TYPICAL_MAT_SIZES, 
                        testing::Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_16UC3, CV_16UC4, CV_32FC1, CV_32FC3, CV_32FC4)));

//////////////////////////////////////////////////////////////////////
// SetToMasked

GPU_PERF_TEST(SetToMasked, cv::gpu::DeviceInfo, cv::Size, perf::MatType)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat src_host(size, type);
    cv::Mat mask_host(size, CV_8UC1);

    declare.in(src_host, WARMUP_RNG);
    fill(mask_host, 0, 2);

    cv::gpu::GpuMat src(src_host);
    cv::Scalar val(1, 2, 3, 4);
    cv::gpu::GpuMat mask(mask_host);
    
    TEST_CYCLE(100)
    {
        src.setTo(val, mask);
    }
}

INSTANTIATE_TEST_CASE_P(MatOp, SetToMasked, testing::Combine(
                        ALL_DEVICES, 
                        GPU_TYPICAL_MAT_SIZES, 
                        testing::Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_16UC3, CV_16UC4, CV_32FC1, CV_32FC3, CV_32FC4)));

//////////////////////////////////////////////////////////////////////
// CopyToMasked

GPU_PERF_TEST(CopyToMasked, cv::gpu::DeviceInfo, cv::Size, perf::MatType)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat src_host(size, type);
    cv::Mat mask_host(size, CV_8UC1);

    declare.in(src_host, WARMUP_RNG);
    fill(mask_host, 0, 2);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat mask(mask_host);
    cv::gpu::GpuMat dst;
    
    TEST_CYCLE(100)
    {
        src.copyTo(dst, mask);
    }
}

INSTANTIATE_TEST_CASE_P(MatOp, CopyToMasked, testing::Combine(
                        ALL_DEVICES, 
                        GPU_TYPICAL_MAT_SIZES, 
                        testing::Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_16UC3, CV_16UC4, CV_32FC1, CV_32FC3, CV_32FC4)));

//////////////////////////////////////////////////////////////////////
// ConvertTo

GPU_PERF_TEST(ConvertTo, cv::gpu::DeviceInfo, cv::Size, perf::MatType, perf::MatType)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::Size size = GET_PARAM(1);
    int type1 = GET_PARAM(2);
    int type2 = GET_PARAM(3);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat src_host(size, type1);

    declare.in(src_host, WARMUP_RNG);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat dst;
    
    TEST_CYCLE(100)
    {
        src.convertTo(dst, type2, 0.5, 1.0);
    }
}

INSTANTIATE_TEST_CASE_P(MatOp, ConvertTo, testing::Combine(
                        ALL_DEVICES, 
                        GPU_TYPICAL_MAT_SIZES, 
                        testing::Values(CV_8UC1, CV_16UC1, CV_32FC1), 
                        testing::Values(CV_8UC1, CV_16UC1, CV_32FC1)));

#endif
