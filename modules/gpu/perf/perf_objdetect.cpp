#include "perf_precomp.hpp"

PERF_TEST_P(DevInfo, HOGDescriptor, testing::ValuesIn(devices()))
{
    DeviceInfo devInfo = GetParam();

    setDevice(devInfo.deviceID());

    Mat img_host = readImage("gpu/hog/road.png", CV_LOAD_IMAGE_GRAYSCALE);

    GpuMat img(img_host);
    vector<Rect> found_locations;

    gpu::HOGDescriptor hog;
    hog.setSVMDetector(gpu::HOGDescriptor::getDefaultPeopleDetector());

    TEST_CYCLE(100)
    {
        hog.detectMultiScale(img, found_locations);
    }
}
