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

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_img(img);

        cv::gpu::HOGDescriptor d_hog;
        d_hog.setSVMDetector(cv::gpu::HOGDescriptor::getDefaultPeopleDetector());

        d_hog.detectMultiScale(d_img, found_locations);

        TEST_CYCLE()
        {
            d_hog.detectMultiScale(d_img, found_locations);
        }
    }
    else
    {
        cv::HOGDescriptor hog;
        hog.setSVMDetector(cv::gpu::HOGDescriptor::getDefaultPeopleDetector());

        hog.detectMultiScale(img, found_locations);

        TEST_CYCLE()
        {
            hog.detectMultiScale(img, found_locations);
        }
    }

    SANITY_CHECK(found_locations);
}

//===========test for CalTech data =============//
DEF_PARAM_TEST_1(HOG, string);

PERF_TEST_P(HOG, CalTech, Values<string>("gpu/caltech/image_00000009_0.png", "gpu/caltech/image_00000032_0.png",
    "gpu/caltech/image_00000165_0.png", "gpu/caltech/image_00000261_0.png", "gpu/caltech/image_00000469_0.png",
    "gpu/caltech/image_00000527_0.png", "gpu/caltech/image_00000574_0.png"))
{
    cv::Mat img = readImage(GetParam(), cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(img.empty());

    std::vector<cv::Rect> found_locations;

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_img(img);

        cv::gpu::HOGDescriptor d_hog;
        d_hog.setSVMDetector(cv::gpu::HOGDescriptor::getDefaultPeopleDetector());

        d_hog.detectMultiScale(d_img, found_locations);

        TEST_CYCLE()
        {
            d_hog.detectMultiScale(d_img, found_locations);
        }
    }
    else
    {
        cv::HOGDescriptor hog;
        hog.setSVMDetector(cv::gpu::HOGDescriptor::getDefaultPeopleDetector());

        hog.detectMultiScale(img, found_locations);

        TEST_CYCLE()
        {
            hog.detectMultiScale(img, found_locations);
        }
    }

    SANITY_CHECK(found_locations);
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

    if (PERF_RUN_GPU())
    {
        cv::gpu::CascadeClassifier_GPU d_cascade;
        ASSERT_TRUE(d_cascade.load(perf::TestBase::getDataPath(GetParam().second)));

        cv::gpu::GpuMat d_img(img);
        cv::gpu::GpuMat d_objects_buffer;

        d_cascade.detectMultiScale(d_img, d_objects_buffer);

        TEST_CYCLE()
        {
            d_cascade.detectMultiScale(d_img, d_objects_buffer);
        }

        GPU_SANITY_CHECK(d_objects_buffer);
    }
    else
    {
        cv::CascadeClassifier cascade;
        ASSERT_TRUE(cascade.load(perf::TestBase::getDataPath("gpu/perf/haarcascade_frontalface_alt.xml")));

        std::vector<cv::Rect> rects;

        cascade.detectMultiScale(img, rects);

        TEST_CYCLE()
        {
            cascade.detectMultiScale(img, rects);
        }

        CPU_SANITY_CHECK(rects);
    }
}

///////////////////////////////////////////////////////////////
// LBP cascade

PERF_TEST_P(ImageAndCascade, ObjDetect_LBPClassifier,
    Values<pair_string>(make_pair("gpu/haarcascade/group_1_640x480_VGA.pgm", "gpu/lbpcascade/lbpcascade_frontalface.xml")))
{
    cv::Mat img = readImage(GetParam().first, cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(img.empty());

    if (PERF_RUN_GPU())
    {
        cv::gpu::CascadeClassifier_GPU d_cascade;
        ASSERT_TRUE(d_cascade.load(perf::TestBase::getDataPath(GetParam().second)));

        cv::gpu::GpuMat d_img(img);
        cv::gpu::GpuMat d_gpu_rects;

        d_cascade.detectMultiScale(d_img, d_gpu_rects);

        TEST_CYCLE()
        {
            d_cascade.detectMultiScale(d_img, d_gpu_rects);
        }

        GPU_SANITY_CHECK(d_gpu_rects);
    }
    else
    {
        cv::CascadeClassifier cascade;
        ASSERT_TRUE(cascade.load(perf::TestBase::getDataPath("gpu/lbpcascade/lbpcascade_frontalface.xml")));

        std::vector<cv::Rect> rects;

        cascade.detectMultiScale(img, rects);

        TEST_CYCLE()
        {
            cascade.detectMultiScale(img, rects);
        }

        CPU_SANITY_CHECK(rects);
    }
}

} // namespace
