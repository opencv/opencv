#include <iostream>

#include "opencv2/opencv_modules.hpp"

#if defined(HAVE_OPENCV_CUDACODEC)

#include <string>
#include <vector>
#include <algorithm>
#include <numeric>

#include <opencv2/core.hpp>
#include <opencv2/core/opengl.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/highgui.hpp>

int main(int argc, const char* argv[])
{
    if (argc != 2)
        return -1;

    const std::string fname(argv[1]);

    cv::namedWindow("CPU", cv::WINDOW_NORMAL);
#if defined(HAVE_OPENGL)
    cv::namedWindow("GPU", cv::WINDOW_OPENGL);
    cv::cuda::setGlDevice();
#else
    cv::namedWindow("GPU", cv::WINDOW_NORMAL);
#endif

    cv::TickMeter tm;
    cv::Mat frame;
    cv::VideoCapture reader(fname);
    for (;;)
    {
        if (!reader.read(frame))
            break;
        cv::imshow("CPU", frame);
        if (cv::waitKey(3) > 0)
            break;
    }

    cv::cuda::GpuMat d_frame;
    cv::Ptr<cv::cudacodec::VideoReader> d_reader = cv::cudacodec::createVideoReader(fname);
    for (;;)
    {
        if (!d_reader->nextFrame(d_frame))
            break;
#if defined(HAVE_OPENGL)
        cv::imshow("GPU", cv::ogl::Texture2D(d_frame));
#else
        d_frame.download(frame);
        cv::imshow("GPU", frame);
#endif
        if (cv::waitKey(3) > 0)
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
