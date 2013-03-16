#include "test_precomp.hpp"

#if defined(HAVE_OPENCV_GPU) && defined(HAVE_CUDA)

using namespace cv;
using namespace cv::gpu;
using namespace cvtest;
using namespace testing;

int main(int argc, char** argv)
{
    try
    {
         const char*  keys =
                "{ h | help ?            | false | Print help}"
                "{ i | info              | false | Print information about system and exit }"
                "{ d | device            | -1   | Device on which tests will be executed (-1 means all devices) }"
                ;

        CommandLineParser cmd(argc, (const char**)argv, keys);

        if (cmd.get<bool>("help"))
        {
            cmd.printParams();
            return 0;
        }

        printCudaInfo();

        if (cmd.get<bool>("info"))
        {
            return 0;
        }

        int device = cmd.get<int>("device");
        if (device < 0)
        {
            DeviceManager::instance().loadAll();

            std::cout << "Run tests on all supported devices \n" << std::endl;
        }
        else
        {
            DeviceManager::instance().load(device);

            DeviceInfo info(device);
            std::cout << "Run tests on device " << device << " [" << info.name() << "] \n" << std::endl;
        }

        TS::ptr()->init("cv");
        InitGoogleTest(&argc, argv);

        return RUN_ALL_TESTS();
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return -1;
    }
    catch (...)
    {
        std::cerr << "Unknown error" << std::endl;
        return -1;
    }

    return 0;
}

#else // HAVE_CUDA

CV_TEST_MAIN("cv")
