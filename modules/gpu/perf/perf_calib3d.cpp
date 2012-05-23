#include "perf_precomp.hpp"

#ifdef HAVE_CUDA

//////////////////////////////////////////////////////////////////////
// StereoBM

GPU_PERF_TEST_1(StereoBM, cv::gpu::DeviceInfo)
{
    cv::gpu::DeviceInfo devInfo = GetParam();
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat img_l_host = readImage("gpu/perf/aloe.jpg", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(img_l_host.empty());

    cv::Mat img_r_host = readImage("gpu/perf/aloeR.jpg", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(img_r_host.empty());

    cv::gpu::StereoBM_GPU bm(0, 256);
    cv::gpu::GpuMat img_l(img_l_host);
    cv::gpu::GpuMat img_r(img_r_host);
    cv::gpu::GpuMat dst;

    bm(img_l, img_r, dst);

    declare.time(5.0);

    TEST_CYCLE()
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
    ASSERT_FALSE(img_l_host.empty());

    cv::Mat img_r_host = readImage("gpu/stereobp/aloe-R.png");
    ASSERT_FALSE(img_r_host.empty());

    cv::gpu::StereoBeliefPropagation bp(64);
    cv::gpu::GpuMat img_l(img_l_host);
    cv::gpu::GpuMat img_r(img_r_host);
    cv::gpu::GpuMat dst;

    bp(img_l, img_r, dst);

    declare.time(10.0);

    TEST_CYCLE()
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
    ASSERT_FALSE(img_l_host.empty());

    cv::Mat img_r_host = readImage("gpu/stereobm/aloe-R.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(img_r_host.empty());

    cv::gpu::StereoConstantSpaceBP csbp(128);
    cv::gpu::GpuMat img_l(img_l_host);
    cv::gpu::GpuMat img_r(img_r_host);
    cv::gpu::GpuMat dst;

    csbp(img_l, img_r, dst);

    declare.time(10.0);

    TEST_CYCLE()
    {
        csbp(img_l, img_r, dst);
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
    ASSERT_FALSE(img_host.empty());

    cv::Mat disp_host = readImage("gpu/stereobm/aloe-disp.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(disp_host.empty());

    cv::gpu::DisparityBilateralFilter f(128);
    cv::gpu::GpuMat img(img_host);
    cv::gpu::GpuMat disp(disp_host);
    cv::gpu::GpuMat dst;

    f(disp, img, dst);

    TEST_CYCLE()
    {
        f(disp, img, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Calib3D, DisparityBilateralFilter, ALL_DEVICES);

//////////////////////////////////////////////////////////////////////
// TransformPoints

IMPLEMENT_PARAM_CLASS(Count, int)

GPU_PERF_TEST(TransformPoints, cv::gpu::DeviceInfo, Count)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    int count = GET_PARAM(1);

    cv::Mat src_host(1, count, CV_32FC3);
    fill(src_host, -100, 100);

    cv::gpu::GpuMat src(src_host);
    cv::Mat rvec = cv::Mat::ones(1, 3, CV_32FC1);
    cv::Mat tvec = cv::Mat::ones(1, 3, CV_32FC1);
    cv::gpu::GpuMat dst;

    cv::gpu::transformPoints(src, rvec, tvec, dst);

    TEST_CYCLE()
    {
        cv::gpu::transformPoints(src, rvec, tvec, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Calib3D, TransformPoints, testing::Combine(
    ALL_DEVICES,
    testing::Values<Count>(5000, 10000, 20000)));

//////////////////////////////////////////////////////////////////////
// ProjectPoints

GPU_PERF_TEST(ProjectPoints, cv::gpu::DeviceInfo, Count)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    int count = GET_PARAM(1);

    cv::Mat src_host(1, count, CV_32FC3);
    fill(src_host, -100, 100);

    cv::gpu::GpuMat src(src_host);
    cv::Mat rvec = cv::Mat::ones(1, 3, CV_32FC1);
    cv::Mat tvec = cv::Mat::ones(1, 3, CV_32FC1);
    cv::Mat camera_mat = cv::Mat::ones(3, 3, CV_32FC1);
    cv::gpu::GpuMat dst;

    cv::gpu::projectPoints(src, rvec, tvec, camera_mat, cv::Mat(), dst);

    TEST_CYCLE()
    {
        cv::gpu::projectPoints(src, rvec, tvec, camera_mat, cv::Mat(), dst);
    }
}

INSTANTIATE_TEST_CASE_P(Calib3D, ProjectPoints, testing::Combine(
    ALL_DEVICES,
    testing::Values<Count>(5000, 10000, 20000)));

//////////////////////////////////////////////////////////////////////
// SolvePnPRansac

GPU_PERF_TEST(SolvePnPRansac, cv::gpu::DeviceInfo, Count)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    int count = GET_PARAM(1);

    cv::Mat object(1, count, CV_32FC3);
    fill(object, -100, 100);

    cv::Mat camera_mat(3, 3, CV_32FC1);
    fill(camera_mat, 0.5, 1);
    camera_mat.at<float>(0, 1) = 0.f;
    camera_mat.at<float>(1, 0) = 0.f;
    camera_mat.at<float>(2, 0) = 0.f;
    camera_mat.at<float>(2, 1) = 0.f;

    cv::Mat dist_coef(1, 8, CV_32F, cv::Scalar::all(0));

    std::vector<cv::Point2f> image_vec;
    cv::Mat rvec_gold(1, 3, CV_32FC1);
    fill(rvec_gold, 0, 1);
    cv::Mat tvec_gold(1, 3, CV_32FC1);
    fill(tvec_gold, 0, 1);
    cv::projectPoints(object, rvec_gold, tvec_gold, camera_mat, dist_coef, image_vec);

    cv::Mat image(1, count, CV_32FC2, &image_vec[0]);

    cv::Mat rvec;
    cv::Mat tvec;

    cv::gpu::solvePnPRansac(object, image, camera_mat, dist_coef, rvec, tvec);

    declare.time(3.0);

    TEST_CYCLE()
    {
        cv::gpu::solvePnPRansac(object, image, camera_mat, dist_coef, rvec, tvec);
    }
}

INSTANTIATE_TEST_CASE_P(Calib3D, SolvePnPRansac, testing::Combine(
    ALL_DEVICES,
    testing::Values<Count>(5000, 10000, 20000)));

//////////////////////////////////////////////////////////////////////
// ReprojectImageTo3D

GPU_PERF_TEST(ReprojectImageTo3D, cv::gpu::DeviceInfo, cv::Size, MatDepth)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src_host(size, depth);
    fill(src_host, 5.0, 30.0);

    cv::Mat Q(4, 4, CV_32FC1);
    fill(Q, 0.1, 1.0);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat dst;

    cv::gpu::reprojectImageTo3D(src, dst, Q);

    TEST_CYCLE()
    {
        cv::gpu::reprojectImageTo3D(src, dst, Q);
    }
}

INSTANTIATE_TEST_CASE_P(Calib3D, ReprojectImageTo3D, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values<MatDepth>(CV_8U, CV_16S)));

//////////////////////////////////////////////////////////////////////
// DrawColorDisp

GPU_PERF_TEST(DrawColorDisp, cv::gpu::DeviceInfo, cv::Size, MatDepth)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Size size = GET_PARAM(1);
    int type = GET_PARAM(2);

    cv::Mat src_host(size, type);
    fill(src_host, 0, 255);

    cv::gpu::GpuMat src(src_host);
    cv::gpu::GpuMat dst;

    cv::gpu::drawColorDisp(src, dst, 255);

    TEST_CYCLE()
    {
        cv::gpu::drawColorDisp(src, dst, 255);
    }
}

INSTANTIATE_TEST_CASE_P(Calib3D, DrawColorDisp, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values(MatDepth(CV_8U), MatDepth(CV_16S))));

#endif

