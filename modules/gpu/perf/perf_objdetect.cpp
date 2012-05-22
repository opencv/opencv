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

    TEST_CYCLE()
    {
        hog.detectMultiScale(img, found_locations);
    }
}

INSTANTIATE_TEST_CASE_P(ObjDetect, HOG, ALL_DEVICES);


GPU_PERF_TEST_1(HaarClassifier, cv::gpu::DeviceInfo)
{
    cv::gpu::DeviceInfo devInfo = GetParam();
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat img_host = readImage("gpu/haarcascade/group_1_640x480_VGA.pgm", cv::IMREAD_GRAYSCALE);

    cv::gpu::CascadeClassifier_GPU cascade;

    if (!cascade.load("haarcascade_frontalface_alt.xml"))
        CV_Error(0, "Can't load cascade");

    cv::gpu::GpuMat img(img_host);
    cv::gpu::GpuMat objects_buffer(1, 100, cv::DataType<cv::Rect>::type);

    TEST_CYCLE()
    {
        cascade.detectMultiScale(img, objects_buffer);
    }
}

INSTANTIATE_TEST_CASE_P(ObjDetect, HaarClassifier, ALL_DEVICES);




#endif
