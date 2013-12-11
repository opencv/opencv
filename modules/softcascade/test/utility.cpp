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


using namespace std;
using namespace cv;
using namespace cv::cuda;
using namespace cvtest;
using namespace testing;
using namespace testing::internal;

//////////////////////////////////////////////////////////////////////
// Gpu devices

bool supportFeature(const DeviceInfo& info, FeatureSet feature)
{
    return TargetArchs::builtWith(feature) && info.supports(feature);
}

DeviceManager& DeviceManager::instance()
{
    static DeviceManager obj;
    return obj;
}

void DeviceManager::load(int i)
{
    devices_.clear();
    devices_.reserve(1);

    std::ostringstream msg;

    if (i < 0 || i >= getCudaEnabledDeviceCount())
    {
        msg << "Incorrect device number - " << i;
        CV_Error(cv::Error::StsBadArg, msg.str());
    }

    DeviceInfo info(i);

    if (!info.isCompatible())
    {
        msg << "Device " << i << " [" << info.name() << "] is NOT compatible with current CUDA module build";
        CV_Error(cv::Error::StsBadArg, msg.str());
    }

    devices_.push_back(info);
}

void DeviceManager::loadAll()
{
    int deviceCount = getCudaEnabledDeviceCount();

    devices_.clear();
    devices_.reserve(deviceCount);

    for (int i = 0; i < deviceCount; ++i)
    {
        DeviceInfo info(i);
        if (info.isCompatible())
        {
            devices_.push_back(info);
        }
    }
}

#endif // HAVE_CUDA
