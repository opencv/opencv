#include <opencv2/core/utility.hpp>
#include <iostream>

const char* keys =
{
    "{ b build | | print complete build info }"
    "{ h help  | | print this help           }"
};

int main(int argc, const char* argv[])
{
    cv::CommandLineParser parser(argc, argv, keys);

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
        std::cout << "OpenCV " << CV_VERSION << std::endl;
    }

    return 0;
}
