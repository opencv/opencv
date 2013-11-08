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
// Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
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
//     and/or other oclMaterials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#if !defined(DUMP_MESSAGE_STDOUT) && !defined(DUMP_PROPERTY_XML)
#error Invalid usage
#endif

#if !defined(DUMP_PROPERTY_XML)
#define DUMP_PROPERTY_XML(...)
#endif

#if !defined(DUMP_MESSAGE_STDOUT)
#define DUMP_MESSAGE_STDOUT(...)
#endif

#include <sstream>

static std::string bytesToStringRepr(size_t value)
{
    size_t b = value % 1024;
    value /= 1024;

    size_t kb = value % 1024;
    value /= 1024;

    size_t mb = value % 1024;
    value /= 1024;

    size_t gb = value;

    std::ostringstream stream;

    if (gb > 0)
        stream << gb << " GB ";
    if (mb > 0)
        stream << mb << " MB ";
    if (kb > 0)
        stream << kb << " kB ";
    if (b > 0)
        stream << b << " B";

    return stream.str();
}

static void dumpOpenCLDevice()
{
    using namespace cv::ocl;
    try
    {
        cv::ocl::PlatformsInfo platforms;
        cv::ocl::getOpenCLPlatforms(platforms);
        DUMP_MESSAGE_STDOUT("OpenCL Platforms: ");
        const char* deviceTypeStr;
        for(unsigned int i=0; i < platforms.size(); i++)
        {
            DUMP_MESSAGE_STDOUT("    " << platforms.at(i)->platformName);
            const cv::ocl::DevicesInfo& devices = platforms.at(i)->devices;
            for(unsigned int j=0; j < devices.size(); j++)
            {
                const cv::ocl::DeviceInfo& current_device = *devices.at(j);
                deviceTypeStr = current_device.deviceType == CVCL_DEVICE_TYPE_CPU
                            ? ("CPU") : (current_device.deviceType == CVCL_DEVICE_TYPE_GPU ? "GPU" : "unknown");
                DUMP_MESSAGE_STDOUT( "        " << deviceTypeStr << " : " << current_device.deviceName << " : " << current_device.deviceVersion );
                DUMP_PROPERTY_XML("cv_ocl_platform_"<< i<<"_device_"<<j, "(Platform=" << current_device.platform->platformName << ")(Type="
                    << deviceTypeStr <<")(Name="<< current_device.deviceName << ")(Version="<< current_device.deviceVersion<<")");
            }
        }
        DUMP_MESSAGE_STDOUT("Current OpenCL device: ");

        const cv::ocl::DeviceInfo& deviceInfo = cv::ocl::Context::getContext()->getDeviceInfo();

        DUMP_MESSAGE_STDOUT("    Platform = "<< deviceInfo.platform->platformName);
        DUMP_PROPERTY_XML("cv_ocl_current_platformName", deviceInfo.platform->platformName);

        deviceTypeStr = deviceInfo.deviceType == CVCL_DEVICE_TYPE_CPU
                        ? "CPU" : (deviceInfo.deviceType == CVCL_DEVICE_TYPE_GPU ? "GPU" : "unknown");
        DUMP_MESSAGE_STDOUT("    Type = "<< deviceTypeStr);
        DUMP_PROPERTY_XML("cv_ocl_current_deviceType", deviceTypeStr);

        DUMP_MESSAGE_STDOUT("    Name = "<< deviceInfo.deviceName);
        DUMP_PROPERTY_XML("cv_ocl_current_deviceName", deviceInfo.deviceName);

        DUMP_MESSAGE_STDOUT("    Version = " << deviceInfo.deviceVersion);
        DUMP_PROPERTY_XML("cv_ocl_current_deviceVersion", deviceInfo.deviceVersion);

        DUMP_MESSAGE_STDOUT("    Compute units = "<< deviceInfo.maxComputeUnits);
        DUMP_PROPERTY_XML("cv_ocl_current_maxComputeUnits", deviceInfo.maxComputeUnits);

        DUMP_MESSAGE_STDOUT("    Max work group size = "<< deviceInfo.maxWorkGroupSize);
        DUMP_PROPERTY_XML("cv_ocl_current_maxWorkGroupSize", deviceInfo.maxWorkGroupSize);

        std::string localMemorySizeStr = bytesToStringRepr(deviceInfo.localMemorySize);
        DUMP_MESSAGE_STDOUT("    Local memory size = "<< localMemorySizeStr.c_str());
        DUMP_PROPERTY_XML("cv_ocl_current_localMemorySize", deviceInfo.localMemorySize);

        std::string maxMemAllocSizeStr = bytesToStringRepr(deviceInfo.maxMemAllocSize);
        DUMP_MESSAGE_STDOUT("    Max memory allocation size = "<< maxMemAllocSizeStr.c_str());
        DUMP_PROPERTY_XML("cv_ocl_current_maxMemAllocSize", deviceInfo.maxMemAllocSize);

        const char* doubleSupportStr = deviceInfo.haveDoubleSupport ? "Yes" : "No";
        DUMP_MESSAGE_STDOUT("    Double support = "<< doubleSupportStr);
        DUMP_PROPERTY_XML("cv_ocl_current_haveDoubleSupport", deviceInfo.haveDoubleSupport);

        const char* isUnifiedMemoryStr = deviceInfo.isUnifiedMemory ? "Yes" : "No";
        DUMP_MESSAGE_STDOUT("    Unified memory = "<< isUnifiedMemoryStr);
        DUMP_PROPERTY_XML("cv_ocl_current_isUnifiedMemory", deviceInfo.isUnifiedMemory);
    }
    catch (...)
    {
        DUMP_MESSAGE_STDOUT("OpenCL device not available");
        DUMP_PROPERTY_XML("cv_ocl", "not available");
    }
}

#undef DUMP_MESSAGE_STDOUT
#undef DUMP_PROPERTY_XML
