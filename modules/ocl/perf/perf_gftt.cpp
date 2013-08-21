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
//    Peng Xiao, pengxiao@outlook.com
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

///////////// GoodFeaturesToTrack ////////////////////////
PERFTEST(GoodFeaturesToTrack)
{
    using namespace cv;

    int maxCorners = 2000;
    double qualityLevel = 0.01;

    std::string images[] = { "rubberwhale1.png", "aloeL.jpg" };

    std::vector<cv::Point2f> pts_gold, pts_ocl;

    for(size_t imgIdx = 0; imgIdx < (sizeof(images)/sizeof(std::string)); ++imgIdx)
    {
        Mat frame = imread(abspath(images[imgIdx]), IMREAD_GRAYSCALE);
        CV_Assert(!frame.empty());

        for(float minDistance = 0; minDistance < 4; minDistance += 3.0)
        {
            SUBTEST << "image = " << images[imgIdx] << "; ";
            SUBTEST << "minDistance = " << minDistance << "; ";

            cv::goodFeaturesToTrack(frame, pts_gold, maxCorners, qualityLevel, minDistance);

            CPU_ON;
            cv::goodFeaturesToTrack(frame, pts_gold, maxCorners, qualityLevel, minDistance);
            CPU_OFF;

            cv::ocl::GoodFeaturesToTrackDetector_OCL detector(maxCorners, qualityLevel, minDistance);

            ocl::oclMat frame_ocl(frame), pts_oclmat;

            WARMUP_ON;
            detector(frame_ocl, pts_oclmat);
            WARMUP_OFF;

            detector.downloadPoints(pts_oclmat, pts_ocl);

            double diff = abs(static_cast<float>(pts_gold.size() - pts_ocl.size()));
            TestSystem::instance().setAccurate(diff == 0.0, diff);

            GPU_ON;
            detector(frame_ocl, pts_oclmat);
            GPU_OFF;

            GPU_FULL_ON;
            frame_ocl.upload(frame);
            detector(frame_ocl, pts_oclmat);
            detector.downloadPoints(pts_oclmat, pts_ocl);
            GPU_FULL_OFF;
        }
    }
}
