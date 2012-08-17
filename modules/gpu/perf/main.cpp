#include "perf_precomp.hpp"

#ifdef HAVE_CUDA

using namespace std;
using namespace cv;
using namespace cv::gpu;
using namespace cvtest;
using namespace testing;

void printInfo()
{
#if defined _WIN32
#   if defined _WIN64
        puts("OS: Windows x64");
#   else
        puts("OS: Windows x32");
#   endif
#elif defined linux
#   if defined _LP64
        puts("OS: Linux x64");
#   else
        puts("OS: Linux x32");
#   endif
#elif defined __APPLE__
#   if defined _LP64
        puts("OS: Apple x64");
#   else
        puts("OS: Apple x32");
#   endif
#endif

    int driver;
    cudaDriverGetVersion(&driver);

    printf("CUDA Driver  version: %d\n", driver);
    printf("CUDA Runtime version: %d\n", CUDART_VERSION);

    puts("GPU module was compiled for the following GPU archs:");
    printf("    BIN: %s\n", CUDA_ARCH_BIN);
    printf("    PTX: %s\n\n", CUDA_ARCH_PTX);

    int deviceCount = getCudaEnabledDeviceCount();
    printf("CUDA device count: %d\n\n", deviceCount);

    for (int i = 0; i < deviceCount; ++i)
    {
        DeviceInfo info(i);

        printf("Device %d:\n", i);
        printf("    Name: %s\n", info.name().c_str());
        printf("    Compute capability version: %d.%d\n", info.majorVersion(), info.minorVersion());
        printf("    Multi Processor Count: %d\n", info.multiProcessorCount());
        printf("    Total memory: %d Mb\n", static_cast<int>(static_cast<int>(info.totalMemory() / 1024.0) / 1024.0));
        printf("    Free  memory: %d Mb\n", static_cast<int>(static_cast<int>(info.freeMemory() / 1024.0) / 1024.0));
        if (!info.isCompatible())
            puts("    !!! This device is NOT compatible with current GPU module build\n");
        printf("\n");
    }
}

int main(int argc, char **argv)
{
    CommandLineParser parser(argc, (const char**)argv,
                             "{ print_info_only | print_info_only | false | Print information about system and exit }"
                             "{ device | device | 0 | Device on which tests will be executed }");

    printInfo();

    if (parser.get<bool>("print_info_only"))
        return 0;

    int device = parser.get<int>("device");

    if (device < 0 || device >= getCudaEnabledDeviceCount())
    {
        cerr << "Incorrect device number - " << device << endl;
        return -1;
    }

    DeviceInfo info(device);
    if (!info.isCompatible())
    {
        cerr << "Device " << device << " [" << info.name() << "] is NOT compatible with current GPU module build" << endl;
        return -1;
    }

    std::cout << "Run tests on device " << device << '\n' << std::endl;

    setDevice(device);

    testing::InitGoogleTest(&argc, argv);
    perf::TestBase::Init(argc, argv);
    return RUN_ALL_TESTS();

    return 0;
}

#else

int main()
{
    printf("OpenCV was built without CUDA support\n");
    return 0;
}

#endif
