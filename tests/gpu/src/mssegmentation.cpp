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
#include "gputest.hpp"
using namespace cv;
using namespace cv::gpu;
using namespace std;

struct CV_GpuMeanShiftSegmentationTest : public CvTest {
    CV_GpuMeanShiftSegmentationTest() : CvTest( "GPU-MeanShiftSegmentation", "MeanShiftSegmentation" ) {}

    void run(int) 
    {
        try 
        {
            bool cc12_ok = TargetArchs::builtWith(COMPUTE_12) && DeviceInfo().supports(COMPUTE_12);
            if (!cc12_ok)
            {
                ts->printf(CvTS::CONSOLE, "\nCompute capability 1.2 is required");
                ts->set_failed_test_info(CvTS::FAIL_GENERIC);
                return;
            }

            Mat img_rgb = imread(string(ts->get_data_path()) + "meanshift/cones.png");
            if (img_rgb.empty())
            {
                ts->set_failed_test_info(CvTS::FAIL_MISSING_TEST_DATA);
                return;
            }

            Mat img;
            cvtColor(img_rgb, img, CV_BGR2BGRA);
            

            for (int minsize = 0; minsize < 2000; minsize = (minsize + 1) * 4) 
            {
                stringstream path;
                path << ts->get_data_path() << "meanshift/cones_segmented_sp10_sr10_minsize" << minsize;
                if (TargetArchs::builtWith(COMPUTE_20) && DeviceInfo().supports(COMPUTE_20))
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
                    ts->set_failed_test_info(CvTS::FAIL_MISSING_TEST_DATA);
                    return;
                }
                if (CheckSimilarity(dst_rgb, dst_ref, 1e-3f) != CvTS::OK)
                {
                    ts->printf(CvTS::LOG, "\ndiffers from image *minsize%d.png\n", minsize);
                    ts->set_failed_test_info(CvTS::FAIL_BAD_ACCURACY);
                }
            }
        }
        catch (const cv::Exception& e) 
        {
            if (!check_and_treat_gpu_exception(e, ts))
                throw;
            return;
        }

        ts->set_failed_test_info(CvTS::OK);
    }    

    int CheckSimilarity(const Mat& m1, const Mat& m2, float max_err)
    {
        Mat diff;
        cv::matchTemplate(m1, m2, diff, CV_TM_CCORR_NORMED);

        float err = abs(diff.at<float>(0, 0) - 1.f);

        if (err > max_err)
            return CvTS::FAIL_INVALID_OUTPUT;

        return CvTS::OK;
    }


} ms_segm_test;
