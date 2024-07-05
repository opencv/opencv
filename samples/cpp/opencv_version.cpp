/**
@file opencv_version.cpp
@brief a program to print version, build configuration and environment details
@modified by Gursimar Singh
*/

#include <opencv2/core/utility.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#elif defined(__linux__) || defined(__unix__)
#include <unistd.h>
#endif

// Check if OpenCV is built with CUDA support
#ifdef HAVE_OPENCV_CUDAARITHM
#include <opencv2/core/cuda.hpp>
#endif

static const std::string keys =
    "{ b build   | | print complete build info              }"
    "{ e env     | | print information about the environment, CPU, GPU, etc. }"
    "{ h help    | | print this help                        }";

int main(int argc, const char* argv[])
{
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("This sample outputs OpenCV version, build configuration, and environment info.");

    if (parser.has("help"))
    {
        parser.printMessage();
    }
    else if (!parser.check())
    {
        parser.printErrors();
    }
    else if (parser.has("build"))
    {
        std::cout << cv::getBuildInformation() << std::endl;
    }
    else if (parser.has("env"))
    {
        unsigned int nThreads = std::thread::hardware_concurrency();
        std::cout << "Number of concurrent threads supported: " << nThreads << std::endl;

        int nProcessors = cv::getNumThreads();
        std::cout << "Number of logical processors: " << nProcessors << std::endl;

        #ifdef HAVE_OPENCV_CUDAARITHM
        std::cout << "Environment: GPU" << std::endl;
        std::cout << "Number of OpenCV-supported GPUs: " << cv::cuda::getCudaEnabledDeviceCount() << std::endl;
        if (cv::cuda::getCudaEnabledDeviceCount() > 0)
        {
            cv::cuda::DeviceInfo deviceInfo(0); // Querying the first GPU
            std::cout << "GPU Name: " << deviceInfo.name() << std::endl;
            std::cout << "Is integrated: " << (deviceInfo.integrated() ? "True" : "False") << std::endl;
            std::cout << "Supports OpenGL: " << (deviceInfo.canMapHostMemory() ? "True" : "False") << std::endl;
            std::cout << "Compute Compatibility: " << deviceInfo.majorVersion()<<"."<< deviceInfo.minorVersion() << std::endl;
            std::cout << "Total Memory: " << deviceInfo.totalMemory() / (1024.0 * 1024.0) << " MB" << std::endl;

        }
        #else
        std::cout << "Environment: CPU" << std::endl;
        #endif
    }
    else
    {
        std::cout << "Welcome to OpenCV " << CV_VERSION << std::endl;
    }
    return 0;
}
