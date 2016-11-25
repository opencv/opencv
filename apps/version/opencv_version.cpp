// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <iostream>

#include <opencv2/core.hpp>

int main(int argc, const char** argv)
{
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
