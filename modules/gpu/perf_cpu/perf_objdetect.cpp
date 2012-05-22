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

GPU_PERF_TEST_1(HaarClassifier, cv::gpu::DeviceInfo)
{    
    cv::Mat img = readImage("gpu/haarcascade/group_1_640x480_VGA.pgm", cv::IMREAD_GRAYSCALE);
        
    cv::CascadeClassifier cascade;

    if (!cascade.load("haarcascade_frontalface_alt.xml"))
        CV_Error(0, "Can't load cascade");
        
    
    std::vector<cv::Rect> rects;
    rects.reserve(1000);

    TEST_CYCLE()
    {
        cascade.detectMultiScale(img, rects);        
    }
}

INSTANTIATE_TEST_CASE_P(ObjDetect, HaarClassifier, ALL_DEVICES);



#endif
