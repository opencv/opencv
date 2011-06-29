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

#ifdef HAVE_CUDA

void print_info()
{
    int deviceCount = cv::gpu::getCudaEnabledDeviceCount();

    printf("Found %d CUDA devices\n\n", deviceCount);

    for (int i = 0; i < deviceCount; ++i)
    {
        cv::gpu::DeviceInfo info(i);
        printf("Device %d:\n", i);
        printf("\tName: %s\n", info.name().c_str());
        printf("\tCompute capability version: %d.%d\n", info.majorVersion(), info.minorVersion());
        printf("\tTotal memory: %d Mb\n", static_cast<int>(static_cast<int>(info.totalMemory() / 1024.0) / 1024.0));
        printf("\tFree memory: %d Mb\n", static_cast<int>(static_cast<int>(info.freeMemory() / 1024.0) / 1024.0));
        if (!info.isCompatible())
            printf("\tThis device is not compatible with current GPU module build\n");
        printf("\n");
    }
    
    printf("GPU module was compiled for next GPU archs:\n");
    printf("\tBIN:%s\n", CUDA_ARCH_BIN);
    printf("\tPTX:%s\n\n", CUDA_ARCH_PTX);
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
    cvtest::TS::ptr()->init("gpu");
    testing::InitGoogleTest(&argc, argv);

    cv::CommandLineParser parser(argc, (const char**)argv);

    std::string outputLevel = parser.get<std::string>("nvtest_output_level", "none");

    if (outputLevel == "none")
        nvidiaTestOutputLevel = OutputLevelNone;
    else if (outputLevel == "compact")
        nvidiaTestOutputLevel = OutputLevelCompact;
    else if (outputLevel == "full")
        nvidiaTestOutputLevel = OutputLevelFull;

    print_info();
    return RUN_ALL_TESTS();
}

#else // HAVE_CUDA

int main(int argc, char** argv)
{
    printf("OpenCV was built without CUDA support\n");
    return 0;
}

#endif // HAVE_CUDA