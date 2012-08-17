#include "perf_precomp.hpp"

using namespace std;
using namespace testing;

namespace {

DEF_PARAM_TEST_1(Image, string);

PERF_TEST_P(Image, Labeling_ConnectedComponents, Values<string>("gpu/labeling/aloe-disp.png"))
{
    cv::Mat image = readImage(GetParam(), cv::IMREAD_GRAYSCALE);

    // cv::threshold(image, image, 150, 255, CV_THRESH_BINARY);

    cv::gpu::GpuMat mask;
    mask.create(image.rows, image.cols, CV_8UC1);

    cv::gpu::GpuMat components;
    components.create(image.rows, image.cols, CV_32SC1);

    cv::gpu::connectivityMask(cv::gpu::GpuMat(image), mask, cv::Scalar::all(0), cv::Scalar::all(2));

    ASSERT_NO_THROW(cv::gpu::labelComponents(mask, components));

    declare.time(1.0);

    TEST_CYCLE()
    {
        cv::gpu::labelComponents(mask, components);
    }
}

} // namespace
