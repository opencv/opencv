#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace cv::gpu;
using namespace cvtest;
using namespace testing;

void printOsInfo()
{
#if defined _WIN32
#   if defined _WIN64
        cout << "OS: Windows x64 \n" << endl;
#   else
        cout << "OS: Windows x32 \n" << endl;
#   endif
#elif defined linux
#   if defined _LP64
        cout << "OS: Linux x64 \n" << endl;
#   else
        cout << "OS: Linux x32 \n" << endl;
#   endif
#elif defined __APPLE__
#   if defined _LP64
        cout << "OS: Apple x64 \n" << endl;
#   else
        cout << "OS: Apple x32 \n" << endl;
#   endif
#endif
}

void printCudaInfo()
{
#ifndef HAVE_CUDA
    cout << "OpenCV was built without CUDA support \n" << endl;
#else
    int driver;
    cudaDriverGetVersion(&driver);

    cout << "CUDA Driver  version: " << driver << '\n';
    cout << "CUDA Runtime version: " << CUDART_VERSION << '\n';

    cout << endl;

    cout << "GPU module was compiled for the following GPU archs:" << endl;
    cout << "    BIN: " << CUDA_ARCH_BIN << '\n';
    cout << "    PTX: " << CUDA_ARCH_PTX << '\n';

    cout << endl;

    int deviceCount = getCudaEnabledDeviceCount();
    cout << "CUDA device count: " << deviceCount << '\n';

    cout << endl;

    for (int i = 0; i < deviceCount; ++i)
    {
        DeviceInfo info(i);

        cout << "Device [" << i << "] \n";
        cout << "\t Name: " << info.name() << '\n';
        cout << "\t Compute capability: " << info.majorVersion() << '.' << info.minorVersion()<< '\n';
        cout << "\t Multi Processor Count: " << info.multiProcessorCount() << '\n';
        cout << "\t Total memory: " << static_cast<int>(static_cast<int>(info.totalMemory() / 1024.0) / 1024.0) << " Mb \n";
        cout << "\t Free  memory: " << static_cast<int>(static_cast<int>(info.freeMemory() / 1024.0) / 1024.0) << " Mb \n";
        if (!info.isCompatible())
            cout << "\t !!! This device is NOT compatible with current GPU module build \n";

        cout << endl;
    }
#endif
}

int main(int argc, char **argv)
{
    CommandLineParser cmd(argc, argv,
        "{ print_info_only | print_info_only | false | Print information about system and exit }"
        "{ device | device | 0 | Device on which tests will be executed }"
        "{ cpu | cpu | false | Run tests on cpu }"
    );

    printOsInfo();
    printCudaInfo();

    if (cmd.get<bool>("print_info_only"))
        return 0;

    int device = cmd.get<int>("device");
    bool cpu = cmd.get<bool>("cpu");
#ifndef HAVE_CUDA
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

    testing::InitGoogleTest(&argc, argv);
    perf::TestBase::Init(argc, argv);
    return RUN_ALL_TESTS();
}
