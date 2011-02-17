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

#include <iostream>
#include <string>
#include <iosfwd>
#include "test_precomp.hpp"
using namespace cv;
using namespace cv::gpu;
using namespace std;

struct CV_GpuMeanShiftSegmentationTest : public cvtest::BaseTest {
    CV_GpuMeanShiftSegmentationTest() {}

    void run(int) 
    {
        bool cc12_ok = TargetArchs::builtWith(FEATURE_SET_COMPUTE_12) && DeviceInfo().supports(FEATURE_SET_COMPUTE_12);
        if (!cc12_ok)
        {
            ts->printf(cvtest::TS::CONSOLE, "\nCompute capability 1.2 is required");
            ts->set_failed_test_info(cvtest::TS::FAIL_GENERIC);
            return;
        }

        Mat img_rgb = imread(string(ts->get_data_path()) + "meanshift/cones.png");
        if (img_rgb.empty())
        {
            ts->set_failed_test_info(cvtest::TS::FAIL_MISSING_TEST_DATA);
            return;
        }

        Mat img;
        cvtColor(img_rgb, img, CV_BGR2BGRA);


        for (int minsize = 0; minsize < 2000; minsize = (minsize + 1) * 4)
        {
            stringstream path;
            path << ts->get_data_path() << "meanshift/cones_segmented_sp10_sr10_minsize" << minsize;
            if (TargetArchs::builtWith(FEATURE_SET_COMPUTE_20) && DeviceInfo().supports(FEATURE_SET_COMPUTE_20))
                path << ".png";
            else
                path << "_CC1X.png";

            Mat dst;
            meanShiftSegmentation((GpuMat)img, dst, 10, 10, minsize);
            Mat dst_rgb;
            cvtColor(dst, dst_rgb, CV_BGRA2BGR);

            //imwrite(path.str(), dst_rgb);
            Mat dst_ref = imread(path.str());
            if (dst_ref.empty())
            {
                ts->set_failed_test_info(cvtest::TS::FAIL_MISSING_TEST_DATA);
                return;
            }
            if (CheckSimilarity(dst_rgb, dst_ref, 1e-3f) != cvtest::TS::OK)
            {
                ts->printf(cvtest::TS::LOG, "\ndiffers from image *minsize%d.png\n", minsize);
                ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
            }
        }

        ts->set_failed_test_info(cvtest::TS::OK);
    }    

    int CheckSimilarity(const Mat& m1, const Mat& m2, float max_err)
    {
        Mat diff;
        cv::matchTemplate(m1, m2, diff, CV_TM_CCORR_NORMED);

        float err = abs(diff.at<float>(0, 0) - 1.f);

        if (err > max_err)
            return cvtest::TS::FAIL_INVALID_OUTPUT;

        return cvtest::TS::OK;
    }


};


TEST(meanShiftSegmentation, regression) { CV_GpuMeanShiftSegmentationTest test; test.safe_run(); }
