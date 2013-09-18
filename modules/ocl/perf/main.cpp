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

int main(int argc, char ** argv)
{
    const char * keys =
        "{ h help     | false              | print help message }"
        "{ t type     | gpu                | set device type:cpu or gpu}"
        "{ p platform | 0                  | set platform id }"
        "{ d device   | 0                  | set device id }";

    CommandLineParser cmd(argc, argv, keys);
    if (cmd.has("help"))
    {
        cout << "Available options besides google test option:" << endl;
        cmd.printMessage();
        return 0;
    }

    string type = cmd.get<string>("type");
    unsigned int pid = cmd.get<unsigned int>("platform");
    int device = cmd.get<int>("device");

    int flag = type == "cpu" ? cv::ocl::CVCL_DEVICE_TYPE_CPU :
                               cv::ocl::CVCL_DEVICE_TYPE_GPU;

    std::vector<cv::ocl::Info> oclinfo;
    int devnums = cv::ocl::getDevice(oclinfo, flag);
    if (devnums <= device || device < 0)
    {
        std::cout << "device invalid\n";
        return -1;
    }

    if (pid >= oclinfo.size())
    {
        std::cout << "platform invalid\n";
        return -1;
    }

    cv::ocl::setDevice(oclinfo[pid], device);
    cv::ocl::setBinaryDiskCache(cv::ocl::CACHE_UPDATE);

    CV_PERF_TEST_MAIN_INTERNALS(ocl, impls)
}
