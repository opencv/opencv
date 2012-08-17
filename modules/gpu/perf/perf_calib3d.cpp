#include "perf_precomp.hpp"

using namespace std;
using namespace testing;

namespace {

//////////////////////////////////////////////////////////////////////
// StereoBM

typedef pair<string, string> pair_string;
DEF_PARAM_TEST_1(ImagePair, pair_string);

PERF_TEST_P(ImagePair, Calib3D_StereoBM, Values(make_pair<string, string>("gpu/perf/aloe.jpg", "gpu/perf/aloeR.jpg")))
{
    declare.time(5.0);

    cv::Mat imgLeft = readImage(GetParam().first, cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(imgLeft.empty());

    cv::Mat imgRight = readImage(GetParam().second, cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(imgRight.empty());

    cv::gpu::StereoBM_GPU d_bm(0, 256);
    cv::gpu::GpuMat d_imgLeft(imgLeft);
    cv::gpu::GpuMat d_imgRight(imgRight);
    cv::gpu::GpuMat d_dst;

    d_bm(d_imgLeft, d_imgRight, d_dst);

    TEST_CYCLE()
    {
        d_bm(d_imgLeft, d_imgRight, d_dst);
    }
}

//////////////////////////////////////////////////////////////////////
// StereoBeliefPropagation

PERF_TEST_P(ImagePair, Calib3D_StereoBeliefPropagation, Values(make_pair<string, string>("gpu/stereobp/aloe-L.png", "gpu/stereobp/aloe-R.png")))
{
    declare.time(10.0);

    cv::Mat imgLeft = readImage(GetParam().first);
    ASSERT_FALSE(imgLeft.empty());

    cv::Mat imgRight = readImage(GetParam().second);
    ASSERT_FALSE(imgRight.empty());

    cv::gpu::StereoBeliefPropagation d_bp(64);
    cv::gpu::GpuMat d_imgLeft(imgLeft);
    cv::gpu::GpuMat d_imgRight(imgRight);
    cv::gpu::GpuMat d_dst;

    d_bp(d_imgLeft, d_imgRight, d_dst);

    TEST_CYCLE()
    {
        d_bp(d_imgLeft, d_imgRight, d_dst);
    }
}

//////////////////////////////////////////////////////////////////////
// StereoConstantSpaceBP

PERF_TEST_P(ImagePair, Calib3D_StereoConstantSpaceBP, Values(make_pair<string, string>("gpu/stereobm/aloe-L.png", "gpu/stereobm/aloe-R.png")))
{
    declare.time(10.0);

    cv::Mat imgLeft = readImage(GetParam().first, cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(imgLeft.empty());

    cv::Mat imgRight = readImage(GetParam().second, cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(imgRight.empty());

    cv::gpu::StereoConstantSpaceBP d_csbp(128);
    cv::gpu::GpuMat d_imgLeft(imgLeft);
    cv::gpu::GpuMat d_imgRight(imgRight);
    cv::gpu::GpuMat d_dst;

    d_csbp(d_imgLeft, d_imgRight, d_dst);

    TEST_CYCLE()
    {
        d_csbp(d_imgLeft, d_imgRight, d_dst);
    }
}

//////////////////////////////////////////////////////////////////////
// DisparityBilateralFilter

PERF_TEST_P(ImagePair, Calib3D_DisparityBilateralFilter, Values(make_pair<string, string>("gpu/stereobm/aloe-L.png", "gpu/stereobm/aloe-disp.png")))
{
    cv::Mat img = readImage(GetParam().first, cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(img.empty());

    cv::Mat disp = readImage(GetParam().second, cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(disp.empty());

    cv::gpu::DisparityBilateralFilter d_filter(128);
    cv::gpu::GpuMat d_img(img);
    cv::gpu::GpuMat d_disp(disp);
    cv::gpu::GpuMat d_dst;

    d_filter(d_disp, d_img, d_dst);

    TEST_CYCLE()
    {
        d_filter(d_disp, d_img, d_dst);
    }
}

//////////////////////////////////////////////////////////////////////
// TransformPoints

DEF_PARAM_TEST_1(Count, int);

PERF_TEST_P(Count, Calib3D_TransformPoints, Values(5000, 10000, 20000))
{
    int count = GetParam();

    cv::Mat src(1, count, CV_32FC3);
    fillRandom(src, -100, 100);

    cv::Mat rvec = cv::Mat::ones(1, 3, CV_32FC1);
    cv::Mat tvec = cv::Mat::ones(1, 3, CV_32FC1);

    cv::gpu::GpuMat d_src(src);
    cv::gpu::GpuMat d_dst;

    cv::gpu::transformPoints(d_src, rvec, tvec, d_dst);

    TEST_CYCLE()
    {
        cv::gpu::transformPoints(d_src, rvec, tvec, d_dst);
    }
}

//////////////////////////////////////////////////////////////////////
// ProjectPoints

PERF_TEST_P(Count, Calib3D_ProjectPoints, Values(5000, 10000, 20000))
{
    int count = GetParam();

    cv::Mat src(1, count, CV_32FC3);
    fillRandom(src, -100, 100);

    cv::Mat rvec = cv::Mat::ones(1, 3, CV_32FC1);
    cv::Mat tvec = cv::Mat::ones(1, 3, CV_32FC1);
    cv::Mat camera_mat = cv::Mat::ones(3, 3, CV_32FC1);

    cv::gpu::GpuMat d_src(src);
    cv::gpu::GpuMat d_dst;

    cv::gpu::projectPoints(d_src, rvec, tvec, camera_mat, cv::Mat(), d_dst);

    TEST_CYCLE()
    {
        cv::gpu::projectPoints(d_src, rvec, tvec, camera_mat, cv::Mat(), d_dst);
    }
}

//////////////////////////////////////////////////////////////////////
// SolvePnPRansac

PERF_TEST_P(Count, Calib3D_SolvePnPRansac, Values(5000, 10000, 20000))
{
    declare.time(3.0);

    int count = GetParam();

    cv::Mat object(1, count, CV_32FC3);
    fillRandom(object, -100, 100);

    cv::Mat camera_mat(3, 3, CV_32FC1);
    fillRandom(camera_mat, 0.5, 1);
    camera_mat.at<float>(0, 1) = 0.f;
    camera_mat.at<float>(1, 0) = 0.f;
    camera_mat.at<float>(2, 0) = 0.f;
    camera_mat.at<float>(2, 1) = 0.f;

    cv::Mat dist_coef(1, 8, CV_32F, cv::Scalar::all(0));

    std::vector<cv::Point2f> image_vec;
    cv::Mat rvec_gold(1, 3, CV_32FC1);
    fillRandom(rvec_gold, 0, 1);
    cv::Mat tvec_gold(1, 3, CV_32FC1);
    fillRandom(tvec_gold, 0, 1);
    cv::projectPoints(object, rvec_gold, tvec_gold, camera_mat, dist_coef, image_vec);

    cv::Mat image(1, count, CV_32FC2, &image_vec[0]);

    cv::Mat rvec;
    cv::Mat tvec;

    cv::gpu::solvePnPRansac(object, image, camera_mat, dist_coef, rvec, tvec);

    TEST_CYCLE()
    {
        cv::gpu::solvePnPRansac(object, image, camera_mat, dist_coef, rvec, tvec);
    }
}

//////////////////////////////////////////////////////////////////////
// ReprojectImageTo3D

PERF_TEST_P(Sz_Depth, Calib3D_ReprojectImageTo3D, Combine(GPU_TYPICAL_MAT_SIZES, Values(CV_8U, CV_16S)))
{
    cv::Size size = GET_PARAM(0);
    int depth = GET_PARAM(1);

    cv::Mat src(size, depth);
    fillRandom(src, 5.0, 30.0);

    cv::Mat Q(4, 4, CV_32FC1);
    fillRandom(Q, 0.1, 1.0);

    cv::gpu::GpuMat d_src(src);
    cv::gpu::GpuMat d_dst;

    cv::gpu::reprojectImageTo3D(d_src, d_dst, Q);

    TEST_CYCLE()
    {
        cv::gpu::reprojectImageTo3D(d_src, d_dst, Q);
    }
}

//////////////////////////////////////////////////////////////////////
// DrawColorDisp

PERF_TEST_P(Sz_Depth, Calib3D_DrawColorDisp, Combine(GPU_TYPICAL_MAT_SIZES, Values(CV_8U, CV_16S)))
{
    cv::Size size = GET_PARAM(0);
    int type = GET_PARAM(1);

    cv::Mat src(size, type);
    fillRandom(src, 0, 255);

    cv::gpu::GpuMat d_src(src);
    cv::gpu::GpuMat d_dst;

    cv::gpu::drawColorDisp(d_src, d_dst, 255);

    TEST_CYCLE()
    {
        cv::gpu::drawColorDisp(d_src, d_dst, 255);
    }
}

} // namespace
