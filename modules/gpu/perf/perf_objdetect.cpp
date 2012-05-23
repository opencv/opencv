#include "perf_precomp.hpp"

#ifdef HAVE_CUDA

///////////////////////////////////////////////////////////////
// HOG

GPU_PERF_TEST_1(HOG, cv::gpu::DeviceInfo)
{
    cv::gpu::DeviceInfo devInfo = GetParam();
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat img_host = readImage("gpu/hog/road.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(img_host.empty());

    cv::gpu::GpuMat img(img_host);
    std::vector<cv::Rect> found_locations;

    cv::gpu::HOGDescriptor hog;
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
    cv::gpu::DeviceInfo devInfo = GetParam();
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat img_host = readImage("gpu/haarcascade/group_1_640x480_VGA.pgm", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(img_host.empty());

    cv::gpu::CascadeClassifier_GPU cascade;

    ASSERT_TRUE(cascade.load(perf::TestBase::getDataPath("gpu/perf/haarcascade_frontalface_alt.xml")));

    cv::gpu::GpuMat img(img_host);
    cv::gpu::GpuMat objects_buffer;

    cascade.detectMultiScale(img, objects_buffer);

    TEST_CYCLE()
    {
        cascade.detectMultiScale(img, objects_buffer);
    }
}

INSTANTIATE_TEST_CASE_P(ObjDetect, HaarClassifier, ALL_DEVICES);

#endif
