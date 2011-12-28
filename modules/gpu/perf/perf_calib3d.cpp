#include "perf_precomp.hpp"

#ifdef HAVE_CUDA

//////////////////////////////////////////////////////////////////////
// TransformPoints

GPU_PERF_TEST_1(TransformPoints, cv::gpu::DeviceInfo)
{
    cv::gpu::DeviceInfo devInfo = GetParam();

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat src_host(1, 10000, CV_32FC3);

    declare.in(src_host, WARMUP_RNG);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat dst;

    TEST_CYCLE(100)
    {
        cv::gpu::transformPoints(src, cv::Mat::ones(1, 3, CV_32FC1), cv::Mat::ones(1, 3, CV_32FC1), dst);
    }
}

INSTANTIATE_TEST_CASE_P(Calib3D, TransformPoints, ALL_DEVICES);

//////////////////////////////////////////////////////////////////////
// ProjectPoints

GPU_PERF_TEST_1(ProjectPoints, cv::gpu::DeviceInfo)
{
    cv::gpu::DeviceInfo devInfo = GetParam();

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat src_host(1, 10000, CV_32FC3);

    declare.in(src_host, WARMUP_RNG);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat dst;

    TEST_CYCLE(100)
    {
        cv::gpu::projectPoints(src, cv::Mat::ones(1, 3, CV_32FC1), cv::Mat::ones(1, 3, CV_32FC1), cv::Mat::ones(3, 3, CV_32FC1), cv::Mat(), dst);
    }
}

INSTANTIATE_TEST_CASE_P(Calib3D, ProjectPoints, ALL_DEVICES);

//////////////////////////////////////////////////////////////////////
// SolvePnPRansac

GPU_PERF_TEST_1(SolvePnPRansac, cv::gpu::DeviceInfo)
{
    cv::gpu::DeviceInfo devInfo = GetParam();

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat object(1, 10000, CV_32FC3);
    cv::Mat image(1, 10000, CV_32FC2);

    declare.in(object, image, WARMUP_RNG);

    cv::Mat rvec, tvec;

    declare.time(3.0);

    TEST_CYCLE(100)
    {
        cv::gpu::solvePnPRansac(object, image, cv::Mat::ones(3, 3, CV_32FC1), cv::Mat(1, 8, CV_32F, cv::Scalar::all(0)), rvec, tvec);
    }
}

INSTANTIATE_TEST_CASE_P(Calib3D, SolvePnPRansac, ALL_DEVICES);

//////////////////////////////////////////////////////////////////////
// StereoBM

GPU_PERF_TEST_1(StereoBM, cv::gpu::DeviceInfo)
{
    cv::gpu::DeviceInfo devInfo = GetParam();

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat img_l_host = readImage("gpu/perf/aloe.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat img_r_host = readImage("gpu/perf/aloeR.jpg", cv::IMREAD_GRAYSCALE);

    ASSERT_FALSE(img_l_host.empty());
    ASSERT_FALSE(img_r_host.empty());

    cv::gpu::GpuMat img_l(img_l_host);
    cv::gpu::GpuMat img_r(img_r_host);
    cv::gpu::GpuMat dst;

    cv::gpu::StereoBM_GPU bm(0, 256);

    declare.time(5.0);

    TEST_CYCLE(100)
    {
        bm(img_l, img_r, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Calib3D, StereoBM, ALL_DEVICES);

//////////////////////////////////////////////////////////////////////
// StereoBeliefPropagation

GPU_PERF_TEST_1(StereoBeliefPropagation, cv::gpu::DeviceInfo)
{
    cv::gpu::DeviceInfo devInfo = GetParam();

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat img_l_host = readImage("gpu/stereobp/aloe-L.png");
    cv::Mat img_r_host = readImage("gpu/stereobp/aloe-R.png");

    ASSERT_FALSE(img_l_host.empty());
    ASSERT_FALSE(img_r_host.empty());

    cv::gpu::GpuMat img_l(img_l_host);
    cv::gpu::GpuMat img_r(img_r_host);
    cv::gpu::GpuMat dst;

    cv::gpu::StereoBeliefPropagation bp(64);

    declare.time(10.0);

    TEST_CYCLE(100)
    {
        bp(img_l, img_r, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Calib3D, StereoBeliefPropagation, ALL_DEVICES);

//////////////////////////////////////////////////////////////////////
// StereoConstantSpaceBP

GPU_PERF_TEST_1(StereoConstantSpaceBP, cv::gpu::DeviceInfo)
{
    cv::gpu::DeviceInfo devInfo = GetParam();

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat img_l_host = readImage("gpu/stereobm/aloe-L.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img_r_host = readImage("gpu/stereobm/aloe-R.png", cv::IMREAD_GRAYSCALE);

    ASSERT_FALSE(img_l_host.empty());
    ASSERT_FALSE(img_r_host.empty());

    cv::gpu::GpuMat img_l(img_l_host);
    cv::gpu::GpuMat img_r(img_r_host);
    cv::gpu::GpuMat dst;

    cv::gpu::StereoConstantSpaceBP bp(128);

    declare.time(10.0);

    TEST_CYCLE(100)
    {
        bp(img_l, img_r, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Calib3D, StereoConstantSpaceBP, ALL_DEVICES);

//////////////////////////////////////////////////////////////////////
// DisparityBilateralFilter

GPU_PERF_TEST_1(DisparityBilateralFilter, cv::gpu::DeviceInfo)
{
    cv::gpu::DeviceInfo devInfo = GetParam();

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat img_host = readImage("gpu/stereobm/aloe-L.png", cv::IMREAD_GRAYSCALE);
    cv::Mat disp_host = readImage("gpu/stereobm/aloe-disp.png", cv::IMREAD_GRAYSCALE);

    ASSERT_FALSE(img_host.empty());
    ASSERT_FALSE(disp_host.empty());

    cv::gpu::GpuMat img(img_host);
    cv::gpu::GpuMat disp(disp_host);
    cv::gpu::GpuMat dst;

    cv::gpu::DisparityBilateralFilter f(128);

    TEST_CYCLE(100)
    {
        f(disp, img, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Calib3D, DisparityBilateralFilter, ALL_DEVICES);

#endif

