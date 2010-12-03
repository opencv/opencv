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


struct CV_GpuMeanShiftTest : public CvTest
{
    CV_GpuMeanShiftTest(): CvTest( "GPU-MeanShift", "MeanShift" ){}

    void run(int)
    {
        int spatialRad = 30;
        int colorRad = 30;

        cv::Mat img = cv::imread(std::string(ts->get_data_path()) + "meanshift/cones.png");
        cv::Mat img_template = cv::imread(std::string(ts->get_data_path()) + "meanshift/con_result.png");

        if (img.empty() || img_template.empty())
        {
            ts->set_failed_test_info(CvTS::FAIL_MISSING_TEST_DATA);
            return;
        }

        cv::Mat rgba;
        cvtColor(img, rgba, CV_BGR2BGRA);

        try
        {
            cv::gpu::GpuMat res;
            cv::gpu::meanShiftFiltering( cv::gpu::GpuMat(rgba), res, spatialRad, colorRad );
            if (res.type() != CV_8UC4)
            {
                ts->set_failed_test_info(CvTS::FAIL_INVALID_OUTPUT);
                return;
            }

            cv::Mat result;
            res.download(result);

            uchar maxDiff = 0;
            for (int j = 0; j < result.rows; ++j)
            {
                const uchar* res_line = result.ptr<uchar>(j);
                const uchar* ref_line = img_template.ptr<uchar>(j);

                for (int i = 0; i < result.cols; ++i)
                {
                    for (int k = 0; k < 3; ++k)
                    {
                        const uchar& ch1 = res_line[result.channels()*i + k];
                        const uchar& ch2 = ref_line[img_template.channels()*i + k];
                        uchar diff = static_cast<uchar>(abs(ch1 - ch2));
                        if (maxDiff < diff)
                            maxDiff = diff;
                    }
                }
            }
            if (maxDiff > 0) 
            {
                ts->printf(CvTS::LOG, "\nMeanShift maxDiff = %d\n", maxDiff);
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

CV_GpuMeanShiftTest CV_GpuMeanShift_test;

struct CV_GpuMeanShiftProcTest : public CvTest
{
    CV_GpuMeanShiftProcTest(): CvTest( "GPU-MeanShiftProc", "MeanShiftProc" ){}

    void run(int)
    {
        int spatialRad = 30;
        int colorRad = 30;

        cv::Mat img = cv::imread(std::string(ts->get_data_path()) + "meanshift/cones.png");

        if (img.empty())
        {
            ts->set_failed_test_info(CvTS::FAIL_MISSING_TEST_DATA);
            return;
        }

        cv::Mat rgba;
        cvtColor(img, rgba, CV_BGR2BGRA);

        try
        {
            cv::gpu::GpuMat h_rmap_filtered;
            cv::gpu::meanShiftFiltering( cv::gpu::GpuMat(rgba), h_rmap_filtered, spatialRad, colorRad );

            cv::gpu::GpuMat d_rmap;
            cv::gpu::GpuMat d_spmap;
            cv::gpu::meanShiftProc( cv::gpu::GpuMat(rgba), d_rmap, d_spmap, spatialRad, colorRad );

            if (d_rmap.type() != CV_8UC4)
            {
                ts->set_failed_test_info(CvTS::FAIL_INVALID_OUTPUT);
                return;
            }

            cv::Mat rmap_filtered;
            h_rmap_filtered.download(rmap_filtered);

            cv::Mat rmap;
            d_rmap.download(rmap);

            uchar maxDiff = 0;
            for (int j = 0; j < rmap_filtered.rows; ++j)
            {
                const uchar* res_line = rmap_filtered.ptr<uchar>(j);
                const uchar* ref_line = rmap.ptr<uchar>(j);

                for (int i = 0; i < rmap_filtered.cols; ++i)
                {
                    for (int k = 0; k < 3; ++k)
                    {
                        const uchar& ch1 = res_line[rmap_filtered.channels()*i + k];
                        const uchar& ch2 = ref_line[rmap.channels()*i + k];
                        uchar diff = static_cast<uchar>(abs(ch1 - ch2));
                        if (maxDiff < diff)
                            maxDiff = diff;
                    }
                }
            }
            if (maxDiff > 0) 
            {
                ts->printf(CvTS::LOG, "\nMeanShiftProc maxDiff = %d\n", maxDiff);
                ts->set_failed_test_info(CvTS::FAIL_GENERIC);
                return;
            }

            cv::Mat spmap;
            d_spmap.download(spmap);

            cv::Mat spmap_template;
            cv::FileStorage fs;

            int major, minor;
            cv::gpu::getComputeCapability(cv::gpu::getDevice(), major, minor);

            if (major == 1)
                fs.open(std::string(ts->get_data_path()) + "meanshift/spmap_1x.yaml", cv::FileStorage::READ);
            else
                fs.open(std::string(ts->get_data_path()) + "meanshift/spmap.yaml", cv::FileStorage::READ);
            fs["spmap"] >> spmap_template;

            for (int y = 0; y < spmap.rows; ++y) {
                for (int x = 0; x < spmap.cols; ++x) {
                    cv::Point_<short> expected = spmap_template.at<cv::Point_<short> >(y, x);
                    cv::Point_<short> actual = spmap.at<cv::Point_<short> >(y, x);
                    int diff = (expected - actual).dot(expected - actual);
                    if (actual != expected) {
                        ts->printf(CvTS::LOG, "\nMeanShiftProc SpMap is bad, diff=%d\n", diff);
                        ts->set_failed_test_info(CvTS::FAIL_GENERIC);
                        return;
                    }
                }
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

CV_GpuMeanShiftProcTest CV_GpuMeanShiftProc_test;
