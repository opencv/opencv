/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"

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

enum OutputLevel
{
    OutputLevelNone,
    OutputLevelCompact,
    OutputLevelFull
};

extern OutputLevel nvidiaTestOutputLevel;

int main(int argc, char** argv)
{
    try
    {
        CommandLineParser parser(argc, (const char**)argv,
                                 "{ print_info_only | print_info_only | false | Print information about system and exit }"
                                 "{ device | device | -1 | Device on which tests will be executed (-1 means all devices) }"
                                 "{ nvtest_output_level | nvtest_output_level | compact | NVidia test verbosity level }");

        printInfo();

        if (parser.get<bool>("print_info_only"))
            return 0;

        int device = parser.get<int>("device");
        if (device < 0)
        {
            DeviceManager::instance().loadAll();
            std::cout << "Run tests on all supported devices\n" << std::endl;
        }
        else
        {
            DeviceManager::instance().load(device);
            std::cout << "Run tests on device " << device << '\n' << std::endl;
        }

        string outputLevel = parser.get<string>("nvtest_output_level");

        if (outputLevel == "none")
            nvidiaTestOutputLevel = OutputLevelNone;
        else if (outputLevel == "compact")
            nvidiaTestOutputLevel = OutputLevelCompact;
        else if (outputLevel == "full")
            nvidiaTestOutputLevel = OutputLevelFull;

        TS::ptr()->init("gpu");
        InitGoogleTest(&argc, argv);

        return RUN_ALL_TESTS();
    }
    catch (const exception& e)
    {
        cerr << e.what() << endl;
        return -1;
    }
    catch (...)
    {
        cerr << "Unknown error" << endl;
        return -1;
    }

    return 0;
}

#else // HAVE_CUDA

int main()
{
    printf("OpenCV was built without CUDA support\n");
    return 0;
}

#endif // HAVE_CUDA
