// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <iostream>

#include <opencv2/core.hpp>

int main(int argc, const char** argv)
{
    cv::CommandLineParser parser(argc, argv,
        "{ h|help    | false | show this help message }"
        "{ v|verbose | false | show build configuration log }"
    );
    if (parser.get<bool>("help"))
    {
        parser.printParams();
    }
    else if (parser.get<bool>("verbose"))
    {
        std::cout << cv::getBuildInformation().c_str() << std::endl;
    }
    else
    {
        std::cout << CV_VERSION << std::endl;
    }
    return 0;
}
