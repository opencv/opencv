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
#include "highgui.h"
#include "cv.h"


using namespace cv;
using namespace std;

struct CV_GpuMatAsyncCallStereoBMTest : public CvTest
{
    public:
        CV_GpuMatAsyncCallStereoBMTest() : CvTest( "GPU-MatAsyncCallStereoBM", "asyncStereoBM" ) {}
        ~CV_GpuMatAsyncCallStereoBMTest() {}

    void run( int /* start_from */)
    {
	    cv::Mat img_l = cv::imread(std::string(ts->get_data_path()) + "stereobm/aloe-L.png", 0);
	    cv::Mat img_r = cv::imread(std::string(ts->get_data_path()) + "stereobm/aloe-R.png", 0);
	    cv::Mat img_reference = cv::imread(std::string(ts->get_data_path()) + "stereobm/aloe-disp.png", 0);

        if (img_l.empty() || img_r.empty() || img_reference.empty())
        {
            ts->set_failed_test_info(CvTS::FAIL_MISSING_TEST_DATA);
            return;
        }

        try
        {
	        cv::gpu::GpuMat disp;
	        cv::gpu::StereoBM_GPU bm(0, 128, 19);

	        cv::gpu::Stream stream;

	        for (size_t i = 0; i < 50; i++)
	        {
		        bm(cv::gpu::GpuMat(img_l), cv::gpu::GpuMat(img_r), disp, stream);
	        }

	        stream.waitForCompletion();
	        disp.convertTo(disp, img_reference.type());
	        double norm = cv::norm(disp, img_reference, cv::NORM_INF);

	        if (norm >= 100) 
            {
                ts->printf(CvTS::LOG, "\nStereoBM norm = %f\n", norm);
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

CV_GpuMatAsyncCallStereoBMTest CV_GpuMatAsyncCallStereoBMTest_test;
