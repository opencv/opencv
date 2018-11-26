#include <iostream>

#include <opencv2/core.hpp>

#include <opencv2/imgcodecs/legacy/constants_c.h>
#include <opencv2/videoio/legacy/constants_c.h>
#include <opencv2/photo/legacy/constants_c.h>
#include <opencv2/video/legacy/constants_c.h>

using namespace cv;

int main(int /*argc*/, const char** /*argv*/)
{
    std::cout
        << (int)CV_LOAD_IMAGE_GRAYSCALE
        << (int)CV_CAP_FFMPEG
        << std::endl;
    return 0;
}
