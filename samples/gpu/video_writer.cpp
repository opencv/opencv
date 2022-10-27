#include <iostream>

#include "opencv2/opencv_modules.hpp"

#if defined(HAVE_OPENCV_CUDACODEC)

#include <vector>
#include <numeric>

#include "opencv2/core.hpp"
#include "opencv2/cudacodec.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
int main(int argc, const char* argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage : video_writer <input video file>" << std::endl;
        return -1;
    }

    constexpr double fps = 25.0;

    cv::VideoCapture reader(argv[1]);

    if (!reader.isOpened())
    {
        std::cerr << "Can't open input video file" << std::endl;
        return -1;
    }

    cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());

    cv::VideoWriter writer;
    cv::Ptr<cv::cudacodec::VideoWriter> d_writer;

    cv::Mat frame;
    cv::cuda::GpuMat d_frame;
    cv::cuda::Stream stream;

    for (int i = 1;; ++i)
    {
        std::cout << "Read " << i << " frame" << std::endl;
        reader >> frame;
        if (frame.empty())
        {
            std::cout << "Stop" << std::endl;
            break;
        }

        if (!writer.isOpened())
        {
            std::cout << "Frame Size : " << frame.cols << "x" << frame.rows << std::endl;
            std::cout << "Open CPU Writer" << std::endl;
            const String outputFilename = "output_cpu.avi";
            if (!writer.open(outputFilename, cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), fps, frame.size()))
                return -1;
            std::cout << "Writing to " << outputFilename << std::endl;
        }

        if (d_writer.empty())
        {
            std::cout << "Open CUDA Writer" << std::endl;
            const cv::String outputFilename = "output_gpu.h264";
            d_writer = cv::cudacodec::createVideoWriter(outputFilename, frame.size(), cv::cudacodec::Codec::H264, fps, cv::cudacodec::ColorFormat::BGR, 0, stream);
            std::cout << "Writing to " << outputFilename << std::endl;
        }

        d_frame.upload(frame, stream);
        std::cout << "Write " << i << " frame" << std::endl;
        writer.write(frame);
        d_writer->write(d_frame);
    }

    return 0;
}

#else

int main()
{
    std::cout << "OpenCV was built without CUDA Video encoding support\n" << std::endl;
    return 0;
}

#endif
