#include "perf_precomp.hpp"

PERF_TEST_P(DevInfo_Size_MatType, merge, testing::Combine(testing::ValuesIn(devices()), 
                                                          testing::Values(GPU_TYPICAL_MAT_SIZES), 
                                                          testing::Values(CV_8UC1, CV_16UC1, CV_32FC1)))
{
    DeviceInfo devInfo = std::tr1::get<0>(GetParam());
    Size size = std::tr1::get<1>(GetParam());
    int type = std::tr1::get<2>(GetParam());

    setDevice(devInfo.deviceID());

    const int num_channels = 4;

    vector<GpuMat> src(num_channels);
    for (int i = 0; i < num_channels; ++i)
        src[i] = GpuMat(size, type, cv::Scalar::all(i)); 

    GpuMat dst(size, CV_MAKETYPE(type, num_channels));

    declare.time(0.5).iterations(100);

    SIMPLE_TEST_CYCLE()
    {
        merge(src, dst);
    }

    Mat dst_host(dst);

    SANITY_CHECK(dst_host);
}

PERF_TEST_P(DevInfo_Size_MatType, split, testing::Combine(testing::ValuesIn(devices()), 
                                                          testing::Values(GPU_TYPICAL_MAT_SIZES), 
                                                          testing::Values(CV_8UC1, CV_16UC1, CV_32FC1)))
{
    DeviceInfo devInfo = std::tr1::get<0>(GetParam());
    Size size = std::tr1::get<1>(GetParam());
    int type = std::tr1::get<2>(GetParam());

    setDevice(devInfo.deviceID());

    const int num_channels = 4;

    GpuMat src(size, CV_MAKETYPE(type, num_channels), cv::Scalar(1, 2, 3, 4));

    vector<GpuMat> dst(num_channels);
    for (int i = 0; i < num_channels; ++i)
        dst[i] = GpuMat(size, type); 

    declare.time(0.5).iterations(100);

    SIMPLE_TEST_CYCLE()
    {
        split(src, dst);
    }

    vector<Mat> dst_host(dst.size());
    for (size_t i = 0; i < dst.size(); ++i)
        dst[i].download(dst_host[i]);

    SANITY_CHECK(dst_host);
}

PERF_TEST_P(DevInfo_Size_MatType, setTo, testing::Combine(testing::ValuesIn(devices()), 
                                                          testing::Values(GPU_TYPICAL_MAT_SIZES), 
                                                          testing::Values(CV_8UC1, CV_8UC4, CV_32FC1)))
{
    DeviceInfo devInfo = std::tr1::get<0>(GetParam());
    Size size = std::tr1::get<1>(GetParam());
    int type = std::tr1::get<2>(GetParam());

    setDevice(devInfo.deviceID());

    GpuMat src(size, type);
    Scalar val(1, 2, 3, 4);

    declare.time(0.5).iterations(100);

    SIMPLE_TEST_CYCLE()
    {
        src.setTo(val);
    }

    Mat src_host(src);

    SANITY_CHECK(src_host);
}

PERF_TEST_P(DevInfo_Size_MatType, setToMasked, testing::Combine(testing::ValuesIn(devices()), 
                                                                testing::Values(GPU_TYPICAL_MAT_SIZES), 
                                                                testing::Values(CV_8UC1, CV_8UC4, CV_32FC1)))
{
    DeviceInfo devInfo = std::tr1::get<0>(GetParam());
    Size size = std::tr1::get<1>(GetParam());
    int type = std::tr1::get<2>(GetParam());

    setDevice(devInfo.deviceID());

    Mat src_host(size, type);

    declare.in(src_host, WARMUP_RNG);

    GpuMat src(src_host);
    Scalar val(1, 2, 3, 4);

    Mat mask_host(size, CV_8UC1);
    randu(mask_host, 0.0, 2.0);
    GpuMat mask(mask_host);
    
    declare.time(0.5).iterations(100);

    SIMPLE_TEST_CYCLE()
    {
        src.setTo(val, mask);
    }

    src.download(src_host);

    SANITY_CHECK(src_host);
}

PERF_TEST_P(DevInfo_Size_MatType, copyToMasked, testing::Combine(testing::ValuesIn(devices()), 
                                                                 testing::Values(GPU_TYPICAL_MAT_SIZES), 
                                                                 testing::Values(CV_8UC1, CV_8UC4, CV_32FC1)))
{
    DeviceInfo devInfo = std::tr1::get<0>(GetParam());
    Size size = std::tr1::get<1>(GetParam());
    int type = std::tr1::get<2>(GetParam());

    setDevice(devInfo.deviceID());

    Mat src_host(size, type);

    declare.in(src_host, WARMUP_RNG);

    GpuMat src(src_host);
    GpuMat dst(size, type);

    Mat mask_host(size, CV_8UC1);
    randu(mask_host, 0.0, 2.0);
    GpuMat mask(mask_host);
    
    declare.time(0.5).iterations(100);

    SIMPLE_TEST_CYCLE()
    {
        src.copyTo(dst, mask);
    }

    Mat dst_host(dst);

    SANITY_CHECK(dst_host);
}

PERF_TEST_P(DevInfo_Size_MatType_MatType, convertTo, testing::Combine(testing::ValuesIn(devices()), 
                                                                      testing::Values(GPU_TYPICAL_MAT_SIZES), 
                                                                      testing::Values(CV_8UC1, CV_16UC1, CV_32FC1),
                                                                      testing::Values(CV_8UC1, CV_16UC1, CV_32FC1)))
{
    DeviceInfo devInfo = std::tr1::get<0>(GetParam());
    Size size = std::tr1::get<1>(GetParam());
    int type1 = std::tr1::get<2>(GetParam());
    int type2 = std::tr1::get<3>(GetParam());

    setDevice(devInfo.deviceID());

    Mat src_host(size, type1);

    declare.in(src_host, WARMUP_RNG);

    GpuMat src(src_host);
    GpuMat dst(size, type2);

    double a = 0.5;
    double b = 1.0;
    
    declare.time(0.5).iterations(100);

    SIMPLE_TEST_CYCLE() 
    {
        src.convertTo(dst, type2, a, b);
    }

    Mat dst_host(dst);

    SANITY_CHECK(dst_host);
}
