#include "perf_precomp.hpp"

#ifdef HAVE_CUDA

GPU_PERF_TEST_1(HOG, cv::gpu::DeviceInfo)
{
    cv::gpu::DeviceInfo devInfo = GetParam();

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat img_host = readImage("gpu/hog/road.png", cv::IMREAD_GRAYSCALE);

    cv::gpu::GpuMat img(img_host);
    std::vector<cv::Rect> found_locations;

    cv::gpu::HOGDescriptor hog;
    hog.setSVMDetector(cv::gpu::HOGDescriptor::getDefaultPeopleDetector());

    TEST_CYCLE(100)
    {
        hog.detectMultiScale(img, found_locations);
    }
}

INSTANTIATE_TEST_CASE_P(ObjDetect, HOG, ALL_DEVICES);

#endif
