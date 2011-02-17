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
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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
//     and/or other GpuMaterials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or bpied warranties, including, but not limited to, the bpied
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

using namespace std;
using namespace cv;

////////////////////////////////////////////////////////////////////////////////
// Merge

struct CV_MergeTest : public cvtest::BaseTest
{
    void can_merge(size_t rows, size_t cols);
    void can_merge_submatrixes(size_t rows, size_t cols);
    void run(int);
};


void CV_MergeTest::can_merge(size_t rows, size_t cols)
{
    bool double_ok = gpu::TargetArchs::builtWith(gpu::NATIVE_DOUBLE) && 
                     gpu::DeviceInfo().supports(gpu::NATIVE_DOUBLE);
    size_t depth_end = double_ok ? CV_64F : CV_32F;

    for (size_t num_channels = 1; num_channels <= 4; ++num_channels)
        for (size_t depth = CV_8U; depth <= depth_end; ++depth)
        {
            vector<Mat> src;
            for (size_t i = 0; i < num_channels; ++i)
                src.push_back(Mat(rows, cols, depth, Scalar::all(static_cast<double>(i))));
            
            Mat dst(rows, cols, CV_MAKETYPE(depth, num_channels));   

            cv::merge(src, dst);   

            vector<gpu::GpuMat> dev_src;
            for (size_t i = 0; i < num_channels; ++i)
                dev_src.push_back(gpu::GpuMat(src[i]));

            gpu::GpuMat dev_dst(rows, cols, CV_MAKETYPE(depth, num_channels));
            cv::gpu::merge(dev_src, dev_dst); 

            Mat host_dst = dev_dst;

            double err = norm(dst, host_dst, NORM_INF);

            if (err > 1e-3)
            {
                //ts->printf(cvtest::TS::CONSOLE, "\nNorm: %f\n", err);
                //ts->printf(cvtest::TS::CONSOLE, "Depth: %d\n", depth);
                //ts->printf(cvtest::TS::CONSOLE, "Rows: %d\n", rows);
                //ts->printf(cvtest::TS::CONSOLE, "Cols: %d\n", cols);
                //ts->printf(cvtest::TS::CONSOLE, "NumChannels: %d\n", num_channels);
                ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
                return;
            }
        }
}


void CV_MergeTest::can_merge_submatrixes(size_t rows, size_t cols)
{
    bool double_ok = gpu::TargetArchs::builtWith(gpu::NATIVE_DOUBLE) && 
                     gpu::DeviceInfo().supports(gpu::NATIVE_DOUBLE);
    size_t depth_end = double_ok ? CV_64F : CV_32F;

    for (size_t num_channels = 1; num_channels <= 4; ++num_channels)
        for (size_t depth = CV_8U; depth <= depth_end; ++depth)
        {
            vector<Mat> src;
            for (size_t i = 0; i < num_channels; ++i) 
            {
                Mat m(rows * 2, cols * 2, depth, Scalar::all(static_cast<double>(i)));
                src.push_back(m(Range(rows / 2, rows / 2 + rows), Range(cols / 2, cols / 2 + cols)));
            }

            Mat dst(rows, cols, CV_MAKETYPE(depth, num_channels));   

            cv::merge(src, dst);   

            vector<gpu::GpuMat> dev_src;
            for (size_t i = 0; i < num_channels; ++i)
                dev_src.push_back(gpu::GpuMat(src[i]));

            gpu::GpuMat dev_dst(rows, cols, CV_MAKETYPE(depth, num_channels));
            cv::gpu::merge(dev_src, dev_dst);

            Mat host_dst = dev_dst;

            double err = norm(dst, host_dst, NORM_INF);

            if (err > 1e-3)
            {
                //ts->printf(cvtest::TS::CONSOLE, "\nNorm: %f\n", err);
                //ts->printf(cvtest::TS::CONSOLE, "Depth: %d\n", depth);
                //ts->printf(cvtest::TS::CONSOLE, "Rows: %d\n", rows);
                //ts->printf(cvtest::TS::CONSOLE, "Cols: %d\n", cols);
                //ts->printf(cvtest::TS::CONSOLE, "NumChannels: %d\n", num_channels);
                ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
                return;
            }
        }
}

void CV_MergeTest::run(int) 
{
    can_merge(1, 1);
    can_merge(1, 7);
    can_merge(53, 7);
    can_merge_submatrixes(1, 1);
    can_merge_submatrixes(1, 7);
    can_merge_submatrixes(53, 7);
}


////////////////////////////////////////////////////////////////////////////////
// Split

struct CV_SplitTest : public cvtest::BaseTest
{
    void can_split(size_t rows, size_t cols);    
    void can_split_submatrix(size_t rows, size_t cols);
    void run(int);
};

void CV_SplitTest::can_split(size_t rows, size_t cols)
{
    bool double_ok = gpu::TargetArchs::builtWith(gpu::NATIVE_DOUBLE) && 
                     gpu::DeviceInfo().supports(gpu::NATIVE_DOUBLE);
    size_t depth_end = double_ok ? CV_64F : CV_32F;

    for (size_t num_channels = 1; num_channels <= 4; ++num_channels)
        for (size_t depth = CV_8U; depth <= depth_end; ++depth)
        {
            Mat src(rows, cols, CV_MAKETYPE(depth, num_channels), Scalar(1.0, 2.0, 3.0, 4.0));   
            vector<Mat> dst;
            cv::split(src, dst);   

            gpu::GpuMat dev_src(src);
            vector<gpu::GpuMat> dev_dst;
            cv::gpu::split(dev_src, dev_dst);

            if (dev_dst.size() != dst.size())
            {
                ts->printf(cvtest::TS::CONSOLE, "Bad output sizes");
                ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
            }

            for (size_t i = 0; i < num_channels; ++i)
            {
                Mat host_dst = dev_dst[i];
                double err = norm(dst[i], host_dst, NORM_INF);

                if (err > 1e-3)
                {
                    //ts->printf(cvtest::TS::CONSOLE, "\nNorm: %f\n", err);
                    //ts->printf(cvtest::TS::CONSOLE, "Depth: %d\n", depth);
                    //ts->printf(cvtest::TS::CONSOLE, "Rows: %d\n", rows);
                    //ts->printf(cvtest::TS::CONSOLE, "Cols: %d\n", cols);
                    //ts->printf(cvtest::TS::CONSOLE, "NumChannels: %d\n", num_channels);
                    ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
                    return;
                }
            }
        }
}

void CV_SplitTest::can_split_submatrix(size_t rows, size_t cols)
{
    bool double_ok = gpu::TargetArchs::builtWith(gpu::NATIVE_DOUBLE) && 
                     gpu::DeviceInfo().supports(gpu::NATIVE_DOUBLE);
    size_t depth_end = double_ok ? CV_64F : CV_32F;

    for (size_t num_channels = 1; num_channels <= 4; ++num_channels)
        for (size_t depth = CV_8U; depth <= depth_end; ++depth)
        {
            Mat src_data(rows * 2, cols * 2, CV_MAKETYPE(depth, num_channels), Scalar(1.0, 2.0, 3.0, 4.0));   
            Mat src(src_data(Range(rows / 2, rows / 2 + rows), Range(cols / 2, cols / 2 + cols)));
            vector<Mat> dst;
            cv::split(src, dst);   

            gpu::GpuMat dev_src(src);
            vector<gpu::GpuMat> dev_dst;
            cv::gpu::split(dev_src, dev_dst);

            if (dev_dst.size() != dst.size())
            {
                ts->printf(cvtest::TS::CONSOLE, "Bad output sizes");
                ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
            }

            for (size_t i = 0; i < num_channels; ++i)
            {
                Mat host_dst = dev_dst[i];
                double err = norm(dst[i], host_dst, NORM_INF);

                if (err > 1e-3)
                {
                    //ts->printf(cvtest::TS::CONSOLE, "\nNorm: %f\n", err);
                    //ts->printf(cvtest::TS::CONSOLE, "Depth: %d\n", depth);
                    //ts->printf(cvtest::TS::CONSOLE, "Rows: %d\n", rows);
                    //ts->printf(cvtest::TS::CONSOLE, "Cols: %d\n", cols);
                    //ts->printf(cvtest::TS::CONSOLE, "NumChannels: %d\n", num_channels);
                    ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
                    return;
                }
            }
        }
}

void CV_SplitTest::run(int)
{
    can_split(1, 1);
    can_split(1, 7);
    can_split(7, 53);
    can_split_submatrix(1, 1);
    can_split_submatrix(1, 7);
    can_split_submatrix(7, 53);
}


////////////////////////////////////////////////////////////////////////////////
// Split and merge

struct CV_SplitMergeTest : public cvtest::BaseTest
{
    void can_split_merge(size_t rows, size_t cols);    
    void run(int);
};

void CV_SplitMergeTest::can_split_merge(size_t rows, size_t cols) {
    bool double_ok = gpu::TargetArchs::builtWith(gpu::NATIVE_DOUBLE) && 
                     gpu::DeviceInfo().supports(gpu::NATIVE_DOUBLE);
    size_t depth_end = double_ok ? CV_64F : CV_32F;

    for (size_t num_channels = 1; num_channels <= 4; ++num_channels)
        for (size_t depth = CV_8U; depth <= depth_end; ++depth)
        {
            Mat orig(rows, cols, CV_MAKETYPE(depth, num_channels), Scalar(1.0, 2.0, 3.0, 4.0));   
            gpu::GpuMat dev_orig(orig);
            vector<gpu::GpuMat> dev_vec;
            cv::gpu::split(dev_orig, dev_vec);

            gpu::GpuMat dev_final(rows, cols, CV_MAKETYPE(depth, num_channels));
            cv::gpu::merge(dev_vec, dev_final);

            double err = cv::norm((Mat)dev_orig, (Mat)dev_final, NORM_INF);
            if (err > 1e-3)
            {
                //ts->printf(cvtest::TS::CONSOLE, "\nNorm: %f\n", err);
                //ts->printf(cvtest::TS::CONSOLE, "Depth: %d\n", depth);
                //ts->printf(cvtest::TS::CONSOLE, "Rows: %d\n", rows);
                //ts->printf(cvtest::TS::CONSOLE, "Cols: %d\n", cols);
                //ts->printf(cvtest::TS::CONSOLE, "NumChannels: %d\n", num_channels);
                ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
                return;
            }
        }
}


void CV_SplitMergeTest::run(int) 
{
    can_split_merge(1, 1);
    can_split_merge(1, 7);
    can_split_merge(7, 53);
}


TEST(merge, accuracy) { CV_MergeTest test; test.safe_run(); }
TEST(split, accuracy) { CV_SplitTest test; test.safe_run(); }
TEST(split, merge_consistency) { CV_SplitMergeTest test; test.safe_run(); }
