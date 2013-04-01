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

#include "precomp.hpp"

int main(int argc, const char *argv[])
{
    vector<ocl::Info> oclinfo;
    int num_devices = getDevice(oclinfo);

    if (num_devices < 1)
    {
        cerr << "no device found\n";
        return -1;
    }

    int devidx = 0;

    for (size_t i = 0; i < oclinfo.size(); i++)
    {
        for (size_t j = 0; j < oclinfo[i].DeviceName.size(); j++)
        {
            printf("device %d: %s\n", devidx++, oclinfo[i].DeviceName[j].c_str());
        }
    }

    redirectError(cvErrorCallback);

    const char *keys =
        "{ h | help    | false | print help message }"
        "{ f | filter  |       | filter for test }"
        "{ w | workdir |       | set working directory }"
        "{ l | list    | false | show all tests }"
        "{ d | device  | 0     | device id }"
        "{ i | iters   | 10    | iteration count }"
        "{ m | warmup  | 1     | gpu warm up iteration count}"
        "{ t | xtop    | 1.1	  | xfactor top boundary}"
        "{ b | xbottom | 0.9	  | xfactor bottom boundary}"
        "{ v | verify  | false | only run gpu once to verify if problems occur}";

    CommandLineParser cmd(argc, argv, keys);

    if (cmd.get<bool>("help"))
    {
        cout << "Avaible options:" << endl;
        cmd.printParams();
        return 0;
    }

    int device = cmd.get<int>("device");

    if (device < 0 || device >= num_devices)
    {
        cerr << "Invalid device ID" << endl;
        return -1;
    }

    if (cmd.get<bool>("verify"))
    {
        TestSystem::instance().setNumIters(1);
        TestSystem::instance().setGPUWarmupIters(0);
        TestSystem::instance().setCPUIters(0);
    }

    devidx = 0;

    for (size_t i = 0; i < oclinfo.size(); i++)
    {
        for (size_t j = 0; j < oclinfo[i].DeviceName.size(); j++, devidx++)
        {
            if (device == devidx)
            {
                ocl::setDevice(oclinfo[i], (int)j);
                TestSystem::instance().setRecordName(oclinfo[i].DeviceName[j]);
                printf("\nuse %d: %s\n", devidx, oclinfo[i].DeviceName[j].c_str());
                goto END_DEV;
            }
        }
    }

END_DEV:

    string filter = cmd.get<string>("filter");
    string workdir = cmd.get<string>("workdir");
    bool list = cmd.get<bool>("list");
    int iters = cmd.get<int>("iters");
    int wu_iters = cmd.get<int>("warmup");
    double x_top = cmd.get<double>("xtop");
    double x_bottom = cmd.get<double>("xbottom");

    TestSystem::instance().setTopThreshold(x_top);
    TestSystem::instance().setBottomThreshold(x_bottom);

    if (!filter.empty())
    {
        TestSystem::instance().setTestFilter(filter);
    }

    if (!workdir.empty())
    {
        if (workdir[workdir.size() - 1] != '/' && workdir[workdir.size() - 1] != '\\')
        {
            workdir += '/';
        }

        TestSystem::instance().setWorkingDir(workdir);
    }

    if (list)
    {
        TestSystem::instance().setListMode(true);
    }

    TestSystem::instance().setNumIters(iters);
    TestSystem::instance().setGPUWarmupIters(wu_iters);

    TestSystem::instance().run();

    return 0;
}