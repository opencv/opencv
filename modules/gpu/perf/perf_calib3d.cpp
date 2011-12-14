#include "perf_precomp.hpp"

PERF_TEST_P(DevInfo, transformPoints, testing::ValuesIn(devices()))
{
    DeviceInfo devInfo = GetParam();

    setDevice(devInfo.deviceID());

    Mat src_host(1, 10000, CV_32FC3);

    declare.in(src_host, WARMUP_RNG);

    GpuMat src(src_host);
    GpuMat dst;

    TEST_CYCLE(100)
    {
        transformPoints(src, Mat::ones(1, 3, CV_32FC1), Mat::ones(1, 3, CV_32FC1), dst);
    }

    Mat dst_host(dst);

    SANITY_CHECK(dst_host);
}

PERF_TEST_P(DevInfo, projectPoints, testing::ValuesIn(devices()))
{
    DeviceInfo devInfo = GetParam();

    setDevice(devInfo.deviceID());

    Mat src_host(1, 10000, CV_32FC3);

    declare.in(src_host, WARMUP_RNG);

    GpuMat src(src_host);
    GpuMat dst;

    TEST_CYCLE(100)
    {
        projectPoints(src, Mat::ones(1, 3, CV_32FC1), Mat::ones(1, 3, CV_32FC1), Mat::ones(3, 3, CV_32FC1), Mat(), dst);
    }

    Mat dst_host(dst);

    SANITY_CHECK(dst_host);
}

PERF_TEST_P(DevInfo, solvePnPRansac, testing::ValuesIn(devices()))
{
    DeviceInfo devInfo = GetParam();

    setDevice(devInfo.deviceID());

    Mat object(1, 10000, CV_32FC3);
    Mat image(1, 10000, CV_32FC2);

    declare.in(object, image, WARMUP_RNG);

    Mat rvec, tvec;

    declare.time(3.0);

    TEST_CYCLE(100)
    {
        solvePnPRansac(object, image, Mat::ones(3, 3, CV_32FC1), Mat(1, 8, CV_32F, Scalar::all(0)), rvec, tvec);
    }

    SANITY_CHECK(rvec);
    SANITY_CHECK(tvec);
}

PERF_TEST_P(DevInfo, StereoBM, testing::ValuesIn(devices()))
{
    DeviceInfo devInfo = GetParam();

    setDevice(devInfo.deviceID());

    Mat img_l_host = readImage("gpu/perf/aloe.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    Mat img_r_host = readImage("gpu/perf/aloeR.jpg", CV_LOAD_IMAGE_GRAYSCALE);

    ASSERT_FALSE(img_l_host.empty());
    ASSERT_FALSE(img_r_host.empty());

    GpuMat img_l(img_l_host);
    GpuMat img_r(img_r_host);

    GpuMat dst;

    StereoBM_GPU bm(0, 256);

    declare.time(5.0);

    TEST_CYCLE(100)
    {
        bm(img_l, img_r, dst);
    }

    Mat dst_host(dst);
    
    SANITY_CHECK(dst_host);
}

PERF_TEST_P(DevInfo, StereoBeliefPropagation, testing::ValuesIn(devices()))
{
    DeviceInfo devInfo = GetParam();

    setDevice(devInfo.deviceID());

    Mat img_l_host = readImage("gpu/stereobp/aloe-L.png", CV_LOAD_IMAGE_GRAYSCALE);
    Mat img_r_host = readImage("gpu/stereobp/aloe-R.png", CV_LOAD_IMAGE_GRAYSCALE);

    ASSERT_FALSE(img_l_host.empty());
    ASSERT_FALSE(img_r_host.empty());

    GpuMat img_l(img_l_host);
    GpuMat img_r(img_r_host);

    GpuMat dst;

    StereoBeliefPropagation bp(128);

    declare.time(10.0);

    TEST_CYCLE(100)
    {
        bp(img_l, img_r, dst);
    }

    Mat dst_host(dst);
    
    SANITY_CHECK(dst_host);
}

PERF_TEST_P(DevInfo, StereoConstantSpaceBP, testing::ValuesIn(devices()))
{
    DeviceInfo devInfo = GetParam();

    setDevice(devInfo.deviceID());

    Mat img_l_host = readImage("gpu/stereocsbp/aloe.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    Mat img_r_host = readImage("gpu/stereocsbp/aloeR.jpg", CV_LOAD_IMAGE_GRAYSCALE);

    ASSERT_FALSE(img_l_host.empty());
    ASSERT_FALSE(img_r_host.empty());

    GpuMat img_l(img_l_host);
    GpuMat img_r(img_r_host);

    GpuMat dst;

    StereoConstantSpaceBP bp(128);

    declare.time(10.0);

    TEST_CYCLE(100)
    {
        bp(img_l, img_r, dst);
    }

    Mat dst_host(dst);
    
    SANITY_CHECK(dst_host);
}

PERF_TEST_P(DevInfo, DisparityBilateralFilter, testing::ValuesIn(devices()))
{
    DeviceInfo devInfo = GetParam();

    setDevice(devInfo.deviceID());

    Mat img_host = readImage("gpu/stereobm/aloe-L.png", CV_LOAD_IMAGE_GRAYSCALE);
    Mat disp_host = readImage("gpu/stereobm/aloe-disp.png", CV_LOAD_IMAGE_GRAYSCALE);

    ASSERT_FALSE(img_host.empty());
    ASSERT_FALSE(disp_host.empty());

    GpuMat img(img_host);
    GpuMat disp(disp_host);

    GpuMat dst;

    DisparityBilateralFilter f(128);

    TEST_CYCLE(100)
    {
        f(disp, img, dst);
    }

    Mat dst_host(dst);
    
    SANITY_CHECK(dst_host);
}
