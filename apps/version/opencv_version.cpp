// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/core/utils/trace.hpp>

#include <opencv2/core/opencl/opencl_info.hpp>

#ifdef HAVE_OPENCV_DNN
#include <opencv2/dnn.hpp>
#endif

#ifdef OPENCV_WIN32_API
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

// defined in core/private.hpp
namespace cv {
CV_EXPORTS const char* currentParallelFramework();
}

static void dumpHWFeatures(bool showAll = false)
{
    std::cout << "OpenCV's HW features list:" << std::endl;
    int count = 0;
    for (int i = 0; i < CV_HARDWARE_MAX_FEATURE; i++)
    {
        cv::String name = cv::getHardwareFeatureName(i);
        if (name.empty())
            continue;
        bool enabled = cv::checkHardwareSupport(i);
        if (enabled)
            count++;
        if (enabled || showAll)
        {
            printf("    ID=%3d (%s) -> %s\n", i, name.c_str(), enabled ? "ON" : "N/A");
        }
    }
    std::cout << "Total available: " << count << std::endl;
}

static void dumpParallelFramework()
{
    const char* parallelFramework = cv::currentParallelFramework();
    if (parallelFramework)
    {
        int threads = cv::getNumThreads();
        std::cout << "Parallel framework: " << parallelFramework << " (nthreads=" << threads << ")" << std::endl;
    }
}

#ifdef HAVE_OPENCV_DNN

static void PrintTo(const cv::dnn::Backend& v, std::ostream* os)
{
    using namespace cv::dnn;
    switch (v) {
        case DNN_BACKEND_DEFAULT: *os << "DEFAULT"; return;
        case DNN_BACKEND_HALIDE: *os << "HALIDE"; return;
        case DNN_BACKEND_INFERENCE_ENGINE: *os << "DLIE*"; return;
        case DNN_BACKEND_OPENCV: *os << "OCV"; return;
        case DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019: *os << "DLIE"; return;
        case DNN_BACKEND_INFERENCE_ENGINE_NGRAPH: *os << "NGRAPH"; return;
        default: /* do nothing */;
    } // don't use "default:" to emit compiler warnings
    *os << "DNN_BACKEND_UNKNOWN(" << (int)v << ")";
}

static void PrintTo(const cv::dnn::Target& v, std::ostream* os)
{
    using namespace cv::dnn;
    switch (v) {
    case DNN_TARGET_CPU: *os << "CPU"; return;
    case DNN_TARGET_OPENCL: *os << "OCL"; return;
    case DNN_TARGET_OPENCL_FP16: *os << "OCL_FP16"; return;
    case DNN_TARGET_MYRIAD: *os << "MYRIAD"; return;
    case DNN_TARGET_FPGA: *os << "FPGA"; return;
    } // don't use "default:" to emit compiler warnings
    *os << "DNN_TARGET_UNKNOWN(" << (int)v << ")";
}

static void dumpDNNBackendsAndTargets()
{
    using namespace cv::dnn;
    std::vector< std::pair<Backend, Target> > available = getAvailableBackends();
    std::cout << "OpenCV DNN: available " << available.size() << " targets:" << std::endl;
    for (size_t i = 0; i < available.size(); i++)
    {
        std::cout << "- ";
        PrintTo(available[i].first, &std::cout);
        std::cout << "/";
        PrintTo(available[i].second, &std::cout);
        std::cout << std::endl;
    }
}

#else

static void dumpDNNBackendsAndTargets()
{
    std::cout << "OpenCV is built without DNN module" << std::endl;
}

#endif  // HAVE_OPENCV_DNN

int main(int argc, const char** argv)
{
    CV_TRACE_FUNCTION();
    CV_TRACE_ARG(argc);
    CV_TRACE_ARG_VALUE(argv0, "argv0", argv[0]);
    CV_TRACE_ARG_VALUE(argv1, "argv1", argv[1]);

#ifndef OPENCV_WIN32_API
    cv::CommandLineParser parser(argc, argv,
        "{ help h usage ? |      | show this help message }"
        "{ verbose v      |      | show build configuration log }"
        "{ opencl         |      | show information about OpenCL (available platforms/devices, default selected device) }"
        "{ hw             |      | show detected HW features (see cv::checkHardwareSupport() function). Use --hw=0 to show available features only }"
        "{ threads        |      | show configured parallel framework and number of active threads }"
        "{ dnn            |      | show available DNN backends and targets }"
    );

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    if (parser.has("verbose"))
    {
        std::cout << cv::getBuildInformation().c_str() << std::endl;
    }
    else
    {
        std::cout << CV_VERSION << std::endl;
    }

    if (parser.has("opencl"))
    {
        cv::dumpOpenCLInformation();
    }

    if (parser.has("hw"))
    {
        dumpHWFeatures(parser.get<bool>("hw"));
    }

    if (parser.has("threads"))
    {
        dumpParallelFramework();
    }

    if (parser.has("dnn"))
    {
        dumpDNNBackendsAndTargets();
    }

#else
    std::cout << cv::getBuildInformation().c_str() << std::endl;
    cv::dumpOpenCLInformation();
    dumpHWFeatures();
    dumpParallelFramework();
    dumpDNNBackendsAndTargets();
    MessageBoxA(NULL, "Check console window output", "OpenCV(" CV_VERSION ")", MB_ICONINFORMATION | MB_OK);
#endif

    return 0;
}
