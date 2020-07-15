// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include <opencv2/core/utility.hpp>
#include <iostream>

static const std::string keys = "{ b build | | print complete build info }"
                                "{ h help  | | print this help           }";

int main(int argc, const char* argv[])
{
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("This sample outputs OpenCV version and build configuration.");
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
    else
    {
        std::cout << "Welcome to OpenCV " << CV_VERSION << std::endl;
    }
    return 0;
}
