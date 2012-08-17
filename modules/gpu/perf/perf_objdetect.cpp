#include "perf_precomp.hpp"

using namespace std;
using namespace testing;

namespace {

///////////////////////////////////////////////////////////////
// HOG

DEF_PARAM_TEST_1(Image, string);

PERF_TEST_P(Image, ObjDetect_HOG, Values<string>("gpu/hog/road.png"))
{
    cv::Mat img = readImage(GetParam(), cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(img.empty());

    std::vector<cv::Rect> found_locations;

    cv::gpu::GpuMat d_img(img);

    cv::gpu::HOGDescriptor d_hog;
    d_hog.setSVMDetector(cv::gpu::HOGDescriptor::getDefaultPeopleDetector());

    d_hog.detectMultiScale(d_img, found_locations);

    TEST_CYCLE()
    {
        d_hog.detectMultiScale(d_img, found_locations);
    }
}

///////////////////////////////////////////////////////////////
// HaarClassifier

typedef pair<string, string> pair_string;
DEF_PARAM_TEST_1(ImageAndCascade, pair_string);

PERF_TEST_P(ImageAndCascade, ObjDetect_HaarClassifier,
    Values<pair_string>(make_pair("gpu/haarcascade/group_1_640x480_VGA.pgm", "gpu/perf/haarcascade_frontalface_alt.xml")))
{
    cv::Mat img = readImage(GetParam().first, cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(img.empty());

    cv::gpu::CascadeClassifier_GPU d_cascade;

    ASSERT_TRUE(d_cascade.load(perf::TestBase::getDataPath(GetParam().second)));

    cv::gpu::GpuMat d_img(img);
    cv::gpu::GpuMat d_objects_buffer;

    d_cascade.detectMultiScale(d_img, d_objects_buffer);

    TEST_CYCLE()
    {
        d_cascade.detectMultiScale(d_img, d_objects_buffer);
    }
}

///////////////////////////////////////////////////////////////
// LBP cascade

PERF_TEST_P(ImageAndCascade, ObjDetect_LBPClassifier,
    Values<pair_string>(make_pair("gpu/haarcascade/group_1_640x480_VGA.pgm", "gpu/lbpcascade/lbpcascade_frontalface.xml")))
{
    cv::Mat img = readImage(GetParam().first, cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(img.empty());

    cv::gpu::CascadeClassifier_GPU d_cascade;

    ASSERT_TRUE(d_cascade.load(perf::TestBase::getDataPath(GetParam().second)));

    cv::gpu::GpuMat d_img(img);
    cv::gpu::GpuMat d_gpu_rects;

    d_cascade.detectMultiScale(d_img, d_gpu_rects);

    TEST_CYCLE()
    {
        d_cascade.detectMultiScale(d_img, d_gpu_rects);
    }
}

} // namespace
