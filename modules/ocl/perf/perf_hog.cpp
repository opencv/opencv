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
//
// @Authors
//    Fangfang Bai, fangfang@multicorewareinc.com
//    Jin Ma,       jin@multicorewareinc.com
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

///////////// HOG////////////////////////

PERFTEST(HOG)
{
    Mat src = imread(abspath("road.png"), cv::IMREAD_GRAYSCALE);

    if (src.empty())
    {
        throw runtime_error("can't open road.png");
    }

    cv::HOGDescriptor hog;
    hog.setSVMDetector(hog.getDefaultPeopleDetector());
    std::vector<cv::Rect> found_locations;
    std::vector<cv::Rect> d_found_locations;

    SUBTEST << src.cols << 'x' << src.rows << "; road.png";

    hog.detectMultiScale(src, found_locations);

    CPU_ON;
    hog.detectMultiScale(src, found_locations);
    CPU_OFF;

    cv::ocl::HOGDescriptor ocl_hog;
    ocl_hog.setSVMDetector(ocl_hog.getDefaultPeopleDetector());
    ocl::oclMat d_src;
    d_src.upload(src);

    WARMUP_ON;
    ocl_hog.detectMultiScale(d_src, d_found_locations);
    WARMUP_OFF;
    
    if(d_found_locations.size() == found_locations.size())
        TestSystem::instance().setAccurate(1, 0);
    else
        TestSystem::instance().setAccurate(0, abs((int)found_locations.size() - (int)d_found_locations.size()));

    GPU_ON;
    ocl_hog.detectMultiScale(d_src, found_locations);
    GPU_OFF;

    GPU_FULL_ON;
    d_src.upload(src);
    ocl_hog.detectMultiScale(d_src, found_locations);
    GPU_FULL_OFF;
}