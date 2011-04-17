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

using namespace cv;
using namespace cv::gpu;

struct CV_GpuStereoBMTest : public cvtest::BaseTest
{
    void run_stress()
    {                
        RNG rng;

        for(int i = 0; i < 10; ++i)
        {
            int winSize = cvRound(rng.uniform(2, 11)) * 2 + 1;

            for(int j = 0; j < 10; ++j)
            {
                int ndisp = cvRound(rng.uniform(5, 32)) * 8;

                for(int s = 0; s < 10; ++s)
                {
                    int w =  cvRound(rng.uniform(1024, 2048));
                    int h =  cvRound(rng.uniform(768, 1152));

                    for(int p = 0; p < 2; ++p)
                    {
                        //int winSize = winsz[i];
                        //int disp = disps[j];
                        Size imgSize(w, h);//res[s];
                        int preset = p;

                        printf("Preset = %d, nidsp = %d, winsz = %d, width = %d, height = %d\n", p, ndisp, winSize, imgSize.width, imgSize.height);

                        GpuMat l(imgSize, CV_8U);
                        GpuMat r(imgSize, CV_8U);

                        GpuMat disparity;
                        StereoBM_GPU bm(preset, ndisp, winSize);
                        bm(l, r, disparity);

            
                    }
                }
            }
        }
    }

    void run(int )
    {
        /*run_stress();
        return;*/

	    cv::Mat img_l = cv::imread(std::string(ts->get_data_path()) + "stereobm/aloe-L.png", 0);
	    cv::Mat img_r = cv::imread(std::string(ts->get_data_path()) + "stereobm/aloe-R.png", 0);
	    cv::Mat img_reference = cv::imread(std::string(ts->get_data_path()) + "stereobm/aloe-disp.png", 0);

        if (img_l.empty() || img_r.empty() || img_reference.empty())
        {
            ts->set_failed_test_info(cvtest::TS::FAIL_MISSING_TEST_DATA);
            return;
        }

        cv::gpu::GpuMat disp;
        cv::gpu::StereoBM_GPU bm(0, 128, 19);
        bm(cv::gpu::GpuMat(img_l), cv::gpu::GpuMat(img_r), disp);

        disp.convertTo(disp, img_reference.type());
        double norm = cv::norm((Mat)disp, img_reference, cv::NORM_INF);

        //cv::imwrite(std::string(ts->get_data_path()) + "stereobm/aloe-disp.png", disp);

        /*cv::imshow("disp", disp);
        cv::imshow("img_reference", img_reference);

        cv::Mat diff = (cv::Mat)disp - (cv::Mat)img_reference;
        cv::imshow("diff", diff);
        cv::waitKey();*/

        if (norm >= 100)
        {
            ts->printf(cvtest::TS::LOG, "\nStereoBM norm = %f\n", norm);
            ts->set_failed_test_info(cvtest::TS::FAIL_GENERIC);
            return;
        }

        ts->set_failed_test_info(cvtest::TS::OK);
    }
};

TEST(StereoBM, regression) { CV_GpuStereoBMTest test; test.safe_run(); }
