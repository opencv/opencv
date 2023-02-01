#include <iostream>

#include "opencv2/opencv_modules.hpp"

#if defined(HAVE_OPENCV_CUDACODEC) && defined(_WIN32)

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

    const double FPS = 25.0;

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

    std::vector<double> cpu_times;
    std::vector<double> gpu_times;
    TickMeter tm;

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

            if (!writer.open("output_cpu.avi", cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), FPS, frame.size()))
                return -1;
        }

        if (d_writer.empty())
        {
            std::cout << "Open CUDA Writer" << std::endl;

            const cv::String outputFilename = "output_gpu.avi";
            d_writer = cv::cudacodec::createVideoWriter(outputFilename, frame.size(), FPS);
        }

        d_frame.upload(frame);

        std::cout << "Write " << i << " frame" << std::endl;

        tm.reset(); tm.start();
        writer.write(frame);
        tm.stop();
        cpu_times.push_back(tm.getTimeMilli());

        tm.reset(); tm.start();
        d_writer->write(d_frame);
        tm.stop();
        gpu_times.push_back(tm.getTimeMilli());
    }

    std::cout << std::endl << "Results:" << std::endl;

    std::sort(cpu_times.begin(), cpu_times.end());
    std::sort(gpu_times.begin(), gpu_times.end());

    double cpu_avg = std::accumulate(cpu_times.begin(), cpu_times.end(), 0.0) / cpu_times.size();
    double gpu_avg = std::accumulate(gpu_times.begin(), gpu_times.end(), 0.0) / gpu_times.size();

    std::cout << "CPU [XVID] : Avg : " << cpu_avg << " ms FPS : " << 1000.0 / cpu_avg << std::endl;
    std::cout << "GPU [H264] : Avg : " << gpu_avg << " ms FPS : " << 1000.0 / gpu_avg << std::endl;

    return 0;
}

#else

int main()
{
    std::cout << "OpenCV was built without CUDA Video encoding support\n" << std::endl;
    return 0;
}

#endif
