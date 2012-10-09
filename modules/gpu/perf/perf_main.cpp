#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace cv::gpu;
using namespace cvtest;
using namespace testing;

int main(int argc, char** argv)
{
    const std::string keys =
            "{ h help ? |   | Print help}"
            "{ i info   |   | Print information about system and exit }"
            "{ device   | 0 | Device on which tests will be executed }"
            "{ cpu      |   | Run tests on cpu }"
            ;

    CommandLineParser cmd(argc, (const char**) argv, keys);

    if (cmd.has("help"))
    {
        cmd.printMessage();
        return 0;
    }

    ts::printOsInfo();
    ts::printCudaInfo();


    if (cmd.has("info"))
    {
        return 0;
    }

    int device = cmd.get<int>("device");
    bool cpu   = cmd.has("cpu");
#if !defined HAVE_CUDA || defined(CUDA_DISABLER)
    cpu = true;
#endif

    if (cpu)
    {
        runOnGpu = false;

        cout << "Run tests on CPU \n" << endl;
    }
    else
    {
        runOnGpu = true;

        if (device < 0 || device >= getCudaEnabledDeviceCount())
        {
            cerr << "Incorrect device index - " << device << endl;
            return -1;
        }

        DeviceInfo info(device);
        if (!info.isCompatible())
        {
            cerr << "Device " << device << " [" << info.name() << "] is NOT compatible with current GPU module build" << endl;
            return -1;
        }

        setDevice(device);

        cout << "Run tests on device " << device << " [" << info.name() << "] \n" << endl;
    }

    InitGoogleTest(&argc, argv);
    perf::TestBase::Init(argc, argv);
    return RUN_ALL_TESTS();
}
