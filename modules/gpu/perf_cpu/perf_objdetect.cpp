#include "perf_cpu_precomp.hpp"

#ifdef HAVE_CUDA

GPU_PERF_TEST_1(HOG, cv::gpu::DeviceInfo)
{
    cv::Mat img = readImage("gpu/hog/road.png", cv::IMREAD_GRAYSCALE);

    std::vector<cv::Rect> found_locations;

    cv::HOGDescriptor hog;
    hog.setSVMDetector(cv::gpu::HOGDescriptor::getDefaultPeopleDetector());

    TEST_CYCLE()
    {
        hog.detectMultiScale(img, found_locations);
    }
}

INSTANTIATE_TEST_CASE_P(ObjDetect, HOG, ALL_DEVICES);

#endif
