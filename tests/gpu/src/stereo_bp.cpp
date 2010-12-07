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

#include "gputest.hpp"
#include <iostream>
#include <string>

struct CV_GpuStereoBPTest : public CvTest
{
    CV_GpuStereoBPTest() : CvTest( "GPU-StereoBP", "StereoBP" ){}
    ~CV_GpuStereoBPTest() {}
  
    void run(int )
    {
        cv::Mat img_l = cv::imread(std::string(ts->get_data_path()) + "stereobp/aloe-L.png");
        cv::Mat img_r = cv::imread(std::string(ts->get_data_path()) + "stereobp/aloe-R.png");
        cv::Mat img_template = cv::imread(std::string(ts->get_data_path()) + "stereobp/aloe-disp.png", 0);

        if (img_l.empty() || img_r.empty() || img_template.empty())
        {
            ts->set_failed_test_info(CvTS::FAIL_MISSING_TEST_DATA);
            return;
        }

        try
        {
            {cv::Mat temp; cv::cvtColor(img_l, temp, CV_BGR2BGRA); cv::swap(temp, img_l);}
            {cv::Mat temp; cv::cvtColor(img_r, temp, CV_BGR2BGRA); cv::swap(temp, img_r);}

            cv::gpu::GpuMat disp;
            cv::gpu::StereoBeliefPropagation bpm(64, 8, 2, 25, 0.1f, 15, 1, CV_16S);

            bpm(cv::gpu::GpuMat(img_l), cv::gpu::GpuMat(img_r), disp);

            //cv::imwrite(std::string(ts->get_data_path()) + "stereobp/aloe-disp.png", disp);

            disp.convertTo(disp, img_template.type());

            double norm = cv::norm(disp, img_template, cv::NORM_INF);
	        if (norm >= 0.5) 
            {
                ts->printf(CvTS::LOG, "\nStereoBP norm = %f\n", norm);
                ts->set_failed_test_info(CvTS::FAIL_GENERIC);
                return;
            }
        }
        catch(const cv::Exception& e)
        {
            if (!check_and_treat_gpu_exception(e, ts))
                throw;
            return;
        }

        ts->set_failed_test_info(CvTS::OK);
    }
};

/////////////////////////////////////////////////////////////////////////////
/////////////////// tests registration  /////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

CV_GpuStereoBPTest CV_GpuStereoBP_test;
