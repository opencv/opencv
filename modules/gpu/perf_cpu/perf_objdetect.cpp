#include "perf_cpu_precomp.hpp"

#ifdef HAVE_CUDA

///////////////////////////////////////////////////////////////
// HOG

GPU_PERF_TEST_1(HOG, cv::gpu::DeviceInfo)
{
    cv::Mat img = readImage("gpu/hog/road.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(img.empty());

    std::vector<cv::Rect> found_locations;

    cv::HOGDescriptor hog;
    hog.setSVMDetector(cv::gpu::HOGDescriptor::getDefaultPeopleDetector());

    hog.detectMultiScale(img, found_locations);

    TEST_CYCLE()
    {
        hog.detectMultiScale(img, found_locations);
    }
}

INSTANTIATE_TEST_CASE_P(ObjDetect, HOG, ALL_DEVICES);

///////////////////////////////////////////////////////////////
// HaarClassifier

GPU_PERF_TEST_1(HaarClassifier, cv::gpu::DeviceInfo)
{
    cv::Mat img = readImage("gpu/haarcascade/group_1_640x480_VGA.pgm", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(img.empty());

    cv::CascadeClassifier cascade;

    ASSERT_TRUE(cascade.load(perf::TestBase::getDataPath("gpu/perf/haarcascade_frontalface_alt.xml")));

    std::vector<cv::Rect> rects;

    cascade.detectMultiScale(img, rects);

    TEST_CYCLE()
    {
        cascade.detectMultiScale(img, rects);
    }
}

INSTANTIATE_TEST_CASE_P(ObjDetect, HaarClassifier, ALL_DEVICES);

#endif
