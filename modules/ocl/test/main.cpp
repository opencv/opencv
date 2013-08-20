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

using namespace std;
using namespace cv;
using namespace cv::ocl;
using namespace cvtest;
using namespace testing;

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
        "{ p | 0                  | set platform id i.e. -p=0}"
        "{ d | 0                  | set device id i.e. -d=0}";

    CommandLineParser cmd(argc, argv, keys);
    if (cmd.get<string>("h")=="true")
    {
        cout << "Avaible options besides goole test option:" << endl;
        cmd.printMessage();
        return 0;
    }
    string type = cmd.get<string>("t");
    unsigned int pid = cmd.get<unsigned int>("p");
    int device = cmd.get<int>("d");

    print_info();
    int flag = CVCL_DEVICE_TYPE_GPU;
    if(type == "cpu")
    {
        flag = CVCL_DEVICE_TYPE_CPU;
    }
    std::vector<cv::ocl::Info> oclinfo;
    int devnums = getDevice(oclinfo, flag);
    if(devnums <= device || device < 0)
    {
        std::cout << "device invalid\n";
        return -1;
    }
    if(pid >= oclinfo.size())
    {
        std::cout << "platform invalid\n";
        return -1;
    }

    setDevice(oclinfo[pid], device);
    setBinaryDiskCache(CACHE_UPDATE);

    cout << "Platform name:" << oclinfo[pid].PlatformName << endl;
    cout << "Device type:" << type << endl << "Device name:" << oclinfo[pid].DeviceName[device] << endl;
    return RUN_ALL_TESTS();
}

#else // DON'T HAVE_OPENCL

int main()
{
    printf("OpenCV was built without OpenCL support\n");
    return 0;
}


#endif // HAVE_OPENCL
