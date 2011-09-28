#include "perf_precomp.hpp"

PERF_TEST_P(DevInfo_Size_MatType_KernelSize, boxFilter, testing::Combine(testing::ValuesIn(devices()), 
                                                              testing::Values(GPU_TYPICAL_MAT_SIZES), 
                                                              testing::Values(CV_8UC1, CV_8UC4),
                                                              testing::Values(3, 5)))
{
    DeviceInfo devInfo = std::tr1::get<0>(GetParam());
    Size size = std::tr1::get<1>(GetParam());
    int type = std::tr1::get<2>(GetParam());
    int ksize = std::tr1::get<3>(GetParam());

    setDevice(devInfo.deviceID());

    Mat src_host(size, type);

    declare.in(src_host, WARMUP_RNG);

    GpuMat src(src_host);
    GpuMat dst(size, type);

    Ptr<FilterEngine_GPU> filter = createBoxFilter_GPU(type, type, Size(ksize, ksize));

    declare.time(0.5).iterations(100);

    SIMPLE_TEST_CYCLE()
    {
        filter->apply(src, dst);
    }

    Mat dst_host = dst;

    SANITY_CHECK(dst_host);
}

PERF_TEST_P(DevInfo_Size_MatType_MorphOp_KernelSize, morphologyFilter, testing::Combine(testing::ValuesIn(devices()), 
                                                                                        testing::Values(GPU_TYPICAL_MAT_SIZES), 
                                                                                        testing::Values(CV_8UC1, CV_8UC4),
                                                                                        testing::Values((int)MORPH_ERODE, (int)MORPH_DILATE),
                                                                                        testing::Values(3, 5)))
{
    DeviceInfo devInfo = std::tr1::get<0>(GetParam());
    Size size = std::tr1::get<1>(GetParam());
    int type = std::tr1::get<2>(GetParam());
    int op = std::tr1::get<3>(GetParam());
    int ksize = std::tr1::get<4>(GetParam());

    setDevice(devInfo.deviceID());

    Mat src_host(size, type);

    declare.in(src_host, WARMUP_RNG);

    GpuMat src(src_host);
    GpuMat dst(size, type);

    Ptr<FilterEngine_GPU> filter = createMorphologyFilter_GPU(op, type, Mat::ones(ksize, ksize, CV_8U));

    declare.time(0.5).iterations(100);

    SIMPLE_TEST_CYCLE()
    {
        filter->apply(src, dst);
    }

    Mat dst_host = dst;

    SANITY_CHECK(dst_host);
}

PERF_TEST_P(DevInfo_Size_MatType_KernelSize, linearFilter, testing::Combine(testing::ValuesIn(devices()), 
                                                                            testing::Values(GPU_TYPICAL_MAT_SIZES), 
                                                                            testing::Values(CV_8UC1, CV_8UC4),
                                                                            testing::Values(3, 5)))
{
    DeviceInfo devInfo = std::tr1::get<0>(GetParam());
    Size size = std::tr1::get<1>(GetParam());
    int type = std::tr1::get<2>(GetParam());
    int ksize = std::tr1::get<3>(GetParam());

    setDevice(devInfo.deviceID());

    Mat src_host(size, type);

    declare.in(src_host, WARMUP_RNG);

    GpuMat src(src_host);
    GpuMat dst(size, type);

    Ptr<FilterEngine_GPU> filter = createLinearFilter_GPU(type, type, Mat::ones(ksize, ksize, CV_8U));

    declare.time(1.0).iterations(100);

    SIMPLE_TEST_CYCLE()
    {
        filter->apply(src, dst);
    }

    Mat dst_host = dst;

    SANITY_CHECK(dst_host);
}

PERF_TEST_P(DevInfo_Size_MatType_KernelSize, separableLinearFilter, testing::Combine(testing::ValuesIn(devices()), 
                                                                                     testing::Values(GPU_TYPICAL_MAT_SIZES), 
                                                                                     testing::Values(CV_8UC1, CV_8UC4, CV_32FC1),
                                                                                     testing::Values(3, 5)))
{
    DeviceInfo devInfo = std::tr1::get<0>(GetParam());
    Size size = std::tr1::get<1>(GetParam());
    int type = std::tr1::get<2>(GetParam());
    int ksize = std::tr1::get<3>(GetParam());

    setDevice(devInfo.deviceID());

    Mat src_host(size, type);

    declare.in(src_host, WARMUP_RNG);

    GpuMat src(src_host);
    GpuMat dst(size, type);

    Mat kernel = getGaussianKernel(ksize, 0.5, CV_32F);
    Ptr<FilterEngine_GPU> filter = createSeparableLinearFilter_GPU(type, type, kernel, kernel, Point(-1,-1));

    declare.time(1.0).iterations(100);

    SIMPLE_TEST_CYCLE()
    {
        filter->apply(src, dst, Rect(0, 0, src.cols, src.rows));
    }

    Mat dst_host = dst;

    SANITY_CHECK(dst_host);
}
