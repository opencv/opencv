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
#include "precomp.hpp"

///////////// HOG////////////////////////
bool match_rect(cv::Rect r1, cv::Rect r2, int threshold)
{
    return ((abs(r1.x - r2.x) < threshold) && (abs(r1.y - r2.y) < threshold) &&
        (abs(r1.width - r2.width) < threshold) && (abs(r1.height - r2.height) < threshold));
}

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

    SUBTEST << 768 << 'x' << 576 << "; road.png";

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
    
    // Ground-truth rectangular people window
    cv::Rect win1_64x128(231, 190, 72, 144);
    cv::Rect win2_64x128(621, 156, 97, 194);
    cv::Rect win1_48x96(238, 198, 63, 126);
    cv::Rect win2_48x96(619, 161, 92, 185);
    cv::Rect win3_48x96(488, 136, 56, 112);

    // Compare whether ground-truth windows are detected and compare the number of windows detected.
    std::vector<int> d_comp(4);
    std::vector<int> comp(4);
    for(int i = 0; i < (int)d_comp.size(); i++)
    {
        d_comp[i] = 0;
        comp[i] = 0;
    }

    int threshold = 10;
    int val = 32;
    d_comp[0] = (int)d_found_locations.size();
    comp[0] = (int)found_locations.size();

    cv::Size winSize = hog.winSize;

    if (winSize == cv::Size(48, 96))
    {
        for(int i = 0; i < (int)d_found_locations.size(); i++)
        {
            if (match_rect(d_found_locations[i], win1_48x96, threshold))
                d_comp[1] = val;
            if (match_rect(d_found_locations[i], win2_48x96, threshold))
                d_comp[2] = val;
            if (match_rect(d_found_locations[i], win3_48x96, threshold))
                d_comp[3] = val;
        }
        for(int i = 0; i < (int)found_locations.size(); i++)
        {
            if (match_rect(found_locations[i], win1_48x96, threshold))
                comp[1] = val;
            if (match_rect(found_locations[i], win2_48x96, threshold))
                comp[2] = val;
            if (match_rect(found_locations[i], win3_48x96, threshold))
                comp[3] = val;
        }
    }
    else if (winSize == cv::Size(64, 128))
    {
        for(int i = 0; i < (int)d_found_locations.size(); i++)
        {
            if (match_rect(d_found_locations[i], win1_64x128, threshold))
                d_comp[1] = val;
            if (match_rect(d_found_locations[i], win2_64x128, threshold))
                d_comp[2] = val;
        }
        for(int i = 0; i < (int)found_locations.size(); i++)
        {
            if (match_rect(found_locations[i], win1_64x128, threshold))
                comp[1] = val;
            if (match_rect(found_locations[i], win2_64x128, threshold))
                comp[2] = val;
        }
    }

    cv::Mat gpu_rst(d_comp), cpu_rst(comp);
    TestSystem::instance().ExpectedMatNear(gpu_rst, cpu_rst, 3);

    GPU_ON;
    ocl_hog.detectMultiScale(d_src, found_locations);
    GPU_OFF;

    GPU_FULL_ON;
    d_src.upload(src);
    ocl_hog.detectMultiScale(d_src, found_locations);
    GPU_FULL_OFF;
}