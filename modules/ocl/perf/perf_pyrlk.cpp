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

///////////// PyrLKOpticalFlow ////////////////////////
TEST(PyrLKOpticalFlow)
{
    std::string images1[] = {"rubberwhale1.png", "aloeL.jpg"};
    std::string images2[] = {"rubberwhale2.png", "aloeR.jpg"};

    for (size_t i = 0; i < sizeof(images1) / sizeof(std::string); i++)
    {
        Mat frame0 = imread(abspath(images1[i]), i == 0 ? IMREAD_COLOR : IMREAD_GRAYSCALE);

        if (frame0.empty())
        {
            std::string errstr = "can't open " + images1[i];
            throw runtime_error(errstr);
        }

        Mat frame1 = imread(abspath(images2[i]), i == 0 ? IMREAD_COLOR : IMREAD_GRAYSCALE);

        if (frame1.empty())
        {
            std::string errstr = "can't open " + images2[i];
            throw runtime_error(errstr);
        }

        Mat gray_frame;

        if (i == 0)
        {
            cvtColor(frame0, gray_frame, COLOR_BGR2GRAY);
        }

        for (int points = Min_Size; points <= Max_Size; points *= Multiple)
        {
            if (i == 0)
                SUBTEST << frame0.cols << "x" << frame0.rows << "; color; " << points << " points";
            else
                SUBTEST << frame0.cols << "x" << frame0.rows << "; gray; " << points << " points";
            Mat nextPts_cpu;
            Mat status_cpu;

            vector<Point2f> pts;
            goodFeaturesToTrack(i == 0 ? gray_frame : frame0, pts, points, 0.01, 0.0);

            vector<Point2f> nextPts;
            vector<unsigned char> status;

            vector<float> err;

            calcOpticalFlowPyrLK(frame0, frame1, pts, nextPts, status, err);

            CPU_ON;
            calcOpticalFlowPyrLK(frame0, frame1, pts, nextPts, status, err);
            CPU_OFF;

            ocl::PyrLKOpticalFlow d_pyrLK;

            ocl::oclMat d_frame0(frame0);
            ocl::oclMat d_frame1(frame1);

            ocl::oclMat d_pts;
            Mat pts_mat(1, (int)pts.size(), CV_32FC2, (void *)&pts[0]);
            d_pts.upload(pts_mat);

            ocl::oclMat d_nextPts;
            ocl::oclMat d_status;
            ocl::oclMat d_err;

            WARMUP_ON;
            d_pyrLK.sparse(d_frame0, d_frame1, d_pts, d_nextPts, d_status, &d_err);
            WARMUP_OFF;

            GPU_ON;
            d_pyrLK.sparse(d_frame0, d_frame1, d_pts, d_nextPts, d_status, &d_err);
             ;
            GPU_OFF;

            GPU_FULL_ON;
            d_frame0.upload(frame0);
            d_frame1.upload(frame1);
            d_pts.upload(pts_mat);
            d_pyrLK.sparse(d_frame0, d_frame1, d_pts, d_nextPts, d_status, &d_err);

            if (!d_nextPts.empty())
            {
                d_nextPts.download(nextPts_cpu);
            }

            if (!d_status.empty())
            {
                d_status.download(status_cpu);
            }

            GPU_FULL_OFF;
        }

    }
}
