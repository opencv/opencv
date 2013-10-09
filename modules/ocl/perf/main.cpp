/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other oclMaterials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors as is and
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

#include "perf_precomp.hpp"

const char * impls[] =
{
    IMPL_OCL,
    IMPL_PLAIN,
#ifdef HAVE_OPENCV_GPU
    IMPL_GPU
#endif
};

using namespace cv::ocl;

int main(int argc, char ** argv)
{
    const char * keys =
        "{ h help     | false              | print help message }"
        "{ t type     | gpu                | set device type:cpu or gpu}"
        "{ p platform | -1                  | set platform id }"
        "{ d device   | 0                  | set device id }";

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

        int flag = type == "cpu" ? cv::ocl::CVCL_DEVICE_TYPE_CPU :
                                   cv::ocl::CVCL_DEVICE_TYPE_GPU;

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

    CV_PERF_TEST_MAIN_INTERNALS(ocl, impls)
}
