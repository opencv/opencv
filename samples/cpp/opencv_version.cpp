#include "opencv2/core/core.hpp"
#include <iostream>

const char* keys = 
{
    "{ b |build |false | print complete build info }"
    "{ h |help  |false | print this help           }"
};

int main(int argc, const char* argv[])
{
    cv::CommandLineParser parser(argc, argv, keys);

    if (parser.get<bool>("help"))
    {
        parser.printParams();
    }
    else if (parser.get<bool>("build"))
    {
        std::cout << cv::getBuildInformation() << std::endl;
    }
    else
    {
        std::cout << "OpenCV " << CV_VERSION << std::endl;	
    }

    return 0;
}