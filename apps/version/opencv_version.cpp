// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/core/utils/trace.hpp>

int main(int argc, const char** argv)
{
    CV_TRACE_FUNCTION();
    CV_TRACE_ARG(argc);
    CV_TRACE_ARG_VALUE(argv0, "argv0", argv[0]);
    CV_TRACE_ARG_VALUE(argv1, "argv1", argv[1]);

    cv::CommandLineParser parser(argc, argv,
        "{ help h usage ? |      | show this help message }"
        "{ verbose v      |      | show build configuration log }"
    );
    if (parser.has("help"))
    {
        parser.printMessage();
    }
    else if (parser.has("verbose"))
    {
        std::cout << cv::getBuildInformation().c_str() << std::endl;
    }
    else
    {
        std::cout << CV_VERSION << std::endl;
    }
    return 0;
}
