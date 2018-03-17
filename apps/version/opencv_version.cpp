// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/core/utils/trace.hpp>

#include <opencv2/core/opencl/opencl_info.hpp>

int main(int argc, const char** argv)
{
    CV_TRACE_FUNCTION();
    CV_TRACE_ARG(argc);
    CV_TRACE_ARG_VALUE(argv0, "argv0", argv[0]);
    CV_TRACE_ARG_VALUE(argv1, "argv1", argv[1]);

    cv::CommandLineParser parser(argc, argv,
        "{ help h usage ? |      | show this help message }"
        "{ verbose v      |      | show build configuration log }"
        "{ opencl         |      | show information about OpenCL (available platforms/devices, default selected device) }"
        "{ hw             |      | show detected HW features (see cv::checkHardwareSupport() function). Use --hw=0 to show available features only }"
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
        bool showAll = parser.get<bool>("hw");
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

    return 0;
}
