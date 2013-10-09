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

#include "test_precomp.hpp"

#ifdef HAVE_OPENCL

using namespace cv;
using namespace cv::ocl;
using namespace cvtest;
using namespace testing;
using std::cout;
using std::endl;

void print_info()
{
    printf("\n");
#if defined _WIN32
#   if defined _WIN64
    puts("OS: Windows 64");
#   else
    puts("OS: Windows 32");
#   endif
#elif defined linux
#   if defined _LP64
    puts("OS: Linux 64");
#   else
    puts("OS: Linux 32");
#   endif
#elif defined __APPLE__
#   if defined _LP64
    puts("OS: Apple 64");
#   else
    puts("OS: Apple 32");
#   endif
#endif

}
int main(int argc, char **argv)
{
    TS::ptr()->init(".");
    InitGoogleTest(&argc, argv);
    const char *keys =
        "{ h | false              | print help message }"
        "{ t | gpu                | set device type:i.e. -t=cpu or gpu}"
        "{ p | -1                 | set platform id i.e. -p=0}"
        "{ d | 0                  | set device id i.e. -d=0}";

    if (getenv("OPENCV_OPENCL_DEVICE") == NULL) // TODO Remove this after buildbot updates
    {
        CommandLineParser cmd(argc, argv, keys);
        if (cmd.has("help"))
        {
            cout << "Available options besides google test option:" << endl;
            cmd.printMessage();
            return 0;
        }
        string type = cmd.get<string>("type");
        int pid = cmd.get<int>("platform");
        int device = cmd.get<int>("device");

        print_info();
        int flag = CVCL_DEVICE_TYPE_GPU;
        if(type == "cpu")
        {
            flag = CVCL_DEVICE_TYPE_CPU;
        }

        cv::ocl::PlatformsInfo platformsInfo;
        cv::ocl::getOpenCLPlatforms(platformsInfo);
        if (pid >= (int)platformsInfo.size())
        {
            std::cout << "platform is invalid\n";
            return 1;
        }

        cv::ocl::DevicesInfo devicesInfo;
        int devnums = cv::ocl::getOpenCLDevices(devicesInfo, flag, (pid < 0) ? NULL : platformsInfo[pid]);
        if (device < 0 || device >= devnums)
        {
            std::cout << "device/platform invalid\n";
            return 1;
        }

        cv::ocl::setDevice(devicesInfo[device]);
    }

    const DeviceInfo& deviceInfo = cv::ocl::Context::getContext()->getDeviceInfo();

    cout << "Device type: " << (deviceInfo.deviceType == CVCL_DEVICE_TYPE_CPU ?
                "CPU" :
                (deviceInfo.deviceType == CVCL_DEVICE_TYPE_GPU ? "GPU" : "unknown")) << endl
         << "Platform name: " << deviceInfo.platform->platformName << endl
         << "Device name: " << deviceInfo.deviceName << endl;
    return RUN_ALL_TESTS();
}

#else // DON'T HAVE_OPENCL

int main()
{
    printf("OpenCV was built without OpenCL support\n");
    return 0;
}


#endif // HAVE_OPENCL
