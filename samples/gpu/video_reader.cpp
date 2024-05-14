#include <iostream>

#include "cvconfig.h"
#include "opencv2/opencv_modules.hpp"

#if defined(HAVE_OPENCV_CUDACODEC)

#include <algorithm>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/opengl.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/highgui.hpp>

int main(int argc, const char* argv[])
{
    if (argc != 2)
        return -1;

    const std::string fname(argv[1]);
    cv::VideoCapture reader(fname);

    cv::namedWindow("CPU", cv::WINDOW_NORMAL);
#if defined(HAVE_QT_OPENGL)
    cv::namedWindow("GPU - QT with opengl", cv::WINDOW_OPENGL);
    cv::Size const imageSize{(int)reader.get(cv::CAP_PROP_FRAME_WIDTH),
        (int)reader.get(cv::CAP_PROP_FRAME_HEIGHT)};
    cv::resizeWindow("GPU - QT with opengl", imageSize);
#else
    cv::namedWindow("GPU", cv::WINDOW_NORMAL);
#endif

    cv::TickMeter tm;
    cv::Mat frame;
    for (;;)
    {
        if (!reader.read(frame))
            break;
        cv::imshow("CPU", frame);
        if (cv::waitKey(1) > 0)
            break;
    }

    cv::cuda::GpuMat d_frame;
    cv::Ptr<cv::cudacodec::VideoReader> d_reader = cv::cudacodec::createVideoReader(fname);
    for (;;)
    {
        if (!d_reader->nextFrame(d_frame))
            break;

#if defined(HAVE_QT_OPENGL)
        cv::imshow("GPU - QT with opengl", d_frame);
#else
        d_frame.download(frame);
        cv::imshow("GPU", frame);
#endif
        if (cv::waitKey(1) > 0)
            break;
    }

    return 0;
}

#else

int main()
{
    std::cout << "OpenCV was built without CUDA Video decoding support\n" << std::endl;
    return 0;
}

#endif
