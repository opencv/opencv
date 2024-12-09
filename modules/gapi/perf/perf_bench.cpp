#include "perf_precomp.hpp"
#include "../test/common/gapi_tests_common.hpp"

namespace opencv_test
{

struct SobelEdgeDetector:  public TestPerfParams<cv::Size> {};
PERF_TEST_P_(SobelEdgeDetector, Fluid)
{
    Size sz = GetParam();
    initMatsRandU(CV_8UC3, sz, CV_8UC3, false);

    GMat in;
    GMat gx  = gapi::Sobel(in, CV_32F, 1, 0);
    GMat gy  = gapi::Sobel(in, CV_32F, 0, 1);
    GMat mag = gapi::sqrt(gapi::mul(gx, gx) + gapi::mul(gy, gy));
    GMat out = gapi::convertTo(mag, CV_8U);
    GComputation sobel(in, out);
    auto pkg = gapi::combine(gapi::core::fluid::kernels(),
                             gapi::imgproc::fluid::kernels());
    auto cc = sobel.compile(cv::descr_of(in_mat1),
                            cv::compile_args(cv::gapi::use_only{pkg}));
    cc(in_mat1, out_mat_gapi);

    TEST_CYCLE()
    {
        cc(in_mat1, out_mat_gapi);
    }
    SANITY_CHECK_NOTHING();
}
PERF_TEST_P_(SobelEdgeDetector, OpenCV)
{
    Size sz = GetParam();
    initMatsRandU(CV_8UC3, sz, CV_8UC3, false);

    Mat gx, gy;
    Mat mag;
    auto cc = [&](const cv::Mat &in_mat, cv::Mat &out_mat) {
        using namespace cv;

        Sobel(in_mat, gx, CV_32F, 1, 0);
        Sobel(in_mat, gy, CV_32F, 0, 1);
        sqrt(gx.mul(gx) + gy.mul(gy), mag);
        mag.convertTo(out_mat, CV_8U);
    };
    cc(in_mat1, out_mat_gapi);

    TEST_CYCLE()
    {
        cc(in_mat1, out_mat_gapi);
    }
    SANITY_CHECK_NOTHING();
}
PERF_TEST_P_(SobelEdgeDetector, OpenCV_Smarter)
{
    Size sz = GetParam();
    initMatsRandU(CV_8UC3, sz, CV_8UC3, false);

    Mat gx, gy;
    Mat ggx, ggy;
    Mat sum;
    Mat mag;

    auto cc = [&](const cv::Mat &in_mat, cv::Mat &out_mat) {
        cv::Sobel(in_mat, gx, CV_32F, 1, 0);
        cv::Sobel(in_mat, gy, CV_32F, 0, 1);
        cv::multiply(gx, gx, ggx);
        cv::multiply(gy, gy, ggy);
        cv::add(ggx, ggy, sum);
        cv::sqrt(sum, mag);
        mag.convertTo(out_mat, CV_8U);
    };
    cc(in_mat1, out_mat_gapi);

    TEST_CYCLE()
    {
        cc(in_mat1, out_mat_gapi);
    }
    SANITY_CHECK_NOTHING();
}
INSTANTIATE_TEST_CASE_P(Benchmark, SobelEdgeDetector,
                        Values(cv::Size(320, 240),
                               cv::Size(640, 480),
                               cv::Size(1280, 720),
                               cv::Size(1920, 1080),
                               cv::Size(3840, 2170)));

} // opencv_test
