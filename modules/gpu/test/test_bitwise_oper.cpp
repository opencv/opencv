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
#include <limits>
#include "test_precomp.hpp"

#define CHECK(pred, err) if (!(pred)) { \
    ts->printf(cvtest::TS::CONSOLE, "Fail: \"%s\" at line: %d\n", #pred, __LINE__); \
    ts->set_failed_test_info(err); \
    return; }

using namespace cv;
using namespace std;

struct CV_GpuBitwiseTest: public cvtest::BaseTest
{
    CV_GpuBitwiseTest() {}

    void run(int)
    {
        int rows, cols;

        bool double_ok = gpu::TargetArchs::builtWith(gpu::NATIVE_DOUBLE) && 
                         gpu::DeviceInfo().supports(gpu::NATIVE_DOUBLE);
        int depth_end = double_ok ? CV_64F : CV_32F;

        for (int depth = CV_8U; depth <= depth_end; ++depth)
            for (int cn = 1; cn <= 4; ++cn)
                for (int attempt = 0; attempt < 3; ++attempt)
                {
                    rows = 1 + rand() % 100;
                    cols = 1 + rand() % 100;
                    test_bitwise_not(rows, cols, CV_MAKETYPE(depth, cn));
                    test_bitwise_or(rows, cols, CV_MAKETYPE(depth, cn));
                    test_bitwise_and(rows, cols, CV_MAKETYPE(depth, cn));
                    test_bitwise_xor(rows, cols, CV_MAKETYPE(depth, cn));
                }
    }

    void test_bitwise_not(int rows, int cols, int type)
    {
        Mat src(rows, cols, type);

        RNG rng;
        for (int i = 0; i < src.rows; ++i)
        {
            Mat row(1, src.cols * src.elemSize(), CV_8U, src.ptr(i));
            rng.fill(row, RNG::UNIFORM, Scalar(0), Scalar(255));
        }

        Mat dst_gold = ~src;

        gpu::GpuMat mask(src.size(), CV_8U);
        mask.setTo(Scalar(1));

        gpu::GpuMat dst;
        gpu::bitwise_not(gpu::GpuMat(src), dst);

        CHECK(dst_gold.size() == dst.size(), cvtest::TS::FAIL_INVALID_OUTPUT);
        CHECK(dst_gold.type() == dst.type(), cvtest::TS::FAIL_INVALID_OUTPUT);        

        Mat dsth(dst);
        for (int i = 0; i < dst_gold.rows; ++i)       
            CHECK(memcmp(dst_gold.ptr(i), dsth.ptr(i), dst_gold.cols * dst_gold.elemSize()) == 0, cvtest::TS::FAIL_INVALID_OUTPUT);

        dst.setTo(Scalar::all(0));
        gpu::bitwise_not(gpu::GpuMat(src), dst, mask);

        CHECK(dst_gold.size() == dst.size(), cvtest::TS::FAIL_INVALID_OUTPUT);
        CHECK(dst_gold.type() == dst.type(), cvtest::TS::FAIL_INVALID_OUTPUT);        

        dsth = dst;
        for (int i = 0; i < dst_gold.rows; ++i)       
            CHECK(memcmp(dst_gold.ptr(i), dsth.ptr(i), dst_gold.cols * dst_gold.elemSize()) == 0, cvtest::TS::FAIL_INVALID_OUTPUT)
    }

    void test_bitwise_or(int rows, int cols, int type)
    {
        Mat src1(rows, cols, type);
        Mat src2(rows, cols, type);

        RNG rng;
        for (int i = 0; i < src1.rows; ++i)
        {
            Mat row1(1, src1.cols * src1.elemSize(), CV_8U, src1.ptr(i));
            rng.fill(row1, RNG::UNIFORM, Scalar(0), Scalar(255));
            Mat row2(1, src2.cols * src2.elemSize(), CV_8U, src2.ptr(i));
            rng.fill(row2, RNG::UNIFORM, Scalar(0), Scalar(255));
        }

        Mat dst_gold = src1 | src2;
        gpu::GpuMat dst = gpu::GpuMat(src1) | gpu::GpuMat(src2);

        CHECK(dst_gold.size() == dst.size(), cvtest::TS::FAIL_INVALID_OUTPUT);
        CHECK(dst_gold.type() == dst.type(), cvtest::TS::FAIL_INVALID_OUTPUT);        
        Mat dsth(dst);
        for (int i = 0; i < dst_gold.rows; ++i)       
            CHECK(memcmp(dst_gold.ptr(i), dsth.ptr(i), dst_gold.cols * dst_gold.elemSize()) == 0, cvtest::TS::FAIL_INVALID_OUTPUT)

        Mat mask(src1.size(), CV_8U);
        randu(mask, Scalar(0), Scalar(255));

        Mat dst_gold2(dst_gold.size(), dst_gold.type()); dst_gold2.setTo(Scalar::all(0));
        gpu::GpuMat dst2(dst.size(), dst.type()); dst2.setTo(Scalar::all(0));
        bitwise_or(src1, src2, dst_gold2, mask);
        gpu::bitwise_or(gpu::GpuMat(src1), gpu::GpuMat(src2), dst2, gpu::GpuMat(mask));

        CHECK(dst_gold2.size() == dst2.size(), cvtest::TS::FAIL_INVALID_OUTPUT);
        CHECK(dst_gold2.type() == dst2.type(), cvtest::TS::FAIL_INVALID_OUTPUT);        
        dsth = dst2;
        for (int i = 0; i < dst_gold.rows; ++i)       
            CHECK(memcmp(dst_gold2.ptr(i), dsth.ptr(i), dst_gold2.cols * dst_gold2.elemSize()) == 0, cvtest::TS::FAIL_INVALID_OUTPUT)
    }

    void test_bitwise_and(int rows, int cols, int type)
    {
        Mat src1(rows, cols, type);
        Mat src2(rows, cols, type);

        RNG rng;
        for (int i = 0; i < src1.rows; ++i)
        {
            Mat row1(1, src1.cols * src1.elemSize(), CV_8U, src1.ptr(i));
            rng.fill(row1, RNG::UNIFORM, Scalar(0), Scalar(255));
            Mat row2(1, src2.cols * src2.elemSize(), CV_8U, src2.ptr(i));
            rng.fill(row2, RNG::UNIFORM, Scalar(0), Scalar(255));
        }

        Mat dst_gold = src1 & src2;

        gpu::GpuMat dst = gpu::GpuMat(src1) & gpu::GpuMat(src2);

        CHECK(dst_gold.size() == dst.size(), cvtest::TS::FAIL_INVALID_OUTPUT);
        CHECK(dst_gold.type() == dst.type(), cvtest::TS::FAIL_INVALID_OUTPUT);        
        Mat dsth(dst);
        for (int i = 0; i < dst_gold.rows; ++i)       
            CHECK(memcmp(dst_gold.ptr(i), dsth.ptr(i), dst_gold.cols * dst_gold.elemSize()) == 0, cvtest::TS::FAIL_INVALID_OUTPUT)


        Mat mask(src1.size(), CV_8U);
        randu(mask, Scalar(0), Scalar(255));

        Mat dst_gold2(dst_gold.size(), dst_gold.type()); dst_gold2.setTo(Scalar::all(0));
        gpu::GpuMat dst2(dst.size(), dst.type()); dst2.setTo(Scalar::all(0));
        bitwise_and(src1, src2, dst_gold2, mask);
        gpu::bitwise_and(gpu::GpuMat(src1), gpu::GpuMat(src2), dst2, gpu::GpuMat(mask));

        CHECK(dst_gold2.size() == dst2.size(), cvtest::TS::FAIL_INVALID_OUTPUT);
        CHECK(dst_gold2.type() == dst2.type(), cvtest::TS::FAIL_INVALID_OUTPUT);        
        dsth = dst2;
        for (int i = 0; i < dst_gold.rows; ++i)       
            CHECK(memcmp(dst_gold2.ptr(i), dsth.ptr(i), dst_gold2.cols * dst_gold2.elemSize()) == 0, cvtest::TS::FAIL_INVALID_OUTPUT)
    }

    void test_bitwise_xor(int rows, int cols, int type)
    {
        Mat src1(rows, cols, type);
        Mat src2(rows, cols, type);

        RNG rng;
        for (int i = 0; i < src1.rows; ++i)
        {
            Mat row1(1, src1.cols * src1.elemSize(), CV_8U, src1.ptr(i));
            rng.fill(row1, RNG::UNIFORM, Scalar(0), Scalar(255));
            Mat row2(1, src2.cols * src2.elemSize(), CV_8U, src2.ptr(i));
            rng.fill(row2, RNG::UNIFORM, Scalar(0), Scalar(255));
        }

        Mat dst_gold = src1 ^ src2;

        gpu::GpuMat dst = gpu::GpuMat(src1) ^ gpu::GpuMat(src2);

        CHECK(dst_gold.size() == dst.size(), cvtest::TS::FAIL_INVALID_OUTPUT);
        CHECK(dst_gold.type() == dst.type(), cvtest::TS::FAIL_INVALID_OUTPUT);        
        Mat dsth(dst);
        for (int i = 0; i < dst_gold.rows; ++i)       
            CHECK(memcmp(dst_gold.ptr(i), dsth.ptr(i), dst_gold.cols * dst_gold.elemSize()) == 0, cvtest::TS::FAIL_INVALID_OUTPUT)


        Mat mask(src1.size(), CV_8U);
        randu(mask, Scalar(0), Scalar(255));

        Mat dst_gold2(dst_gold.size(), dst_gold.type()); dst_gold2.setTo(Scalar::all(0));
        gpu::GpuMat dst2(dst.size(), dst.type()); dst2.setTo(Scalar::all(0));
        bitwise_xor(src1, src2, dst_gold2, mask);
        gpu::bitwise_xor(gpu::GpuMat(src1), gpu::GpuMat(src2), dst2, gpu::GpuMat(mask));

        CHECK(dst_gold2.size() == dst2.size(), cvtest::TS::FAIL_INVALID_OUTPUT);
        CHECK(dst_gold2.type() == dst2.type(), cvtest::TS::FAIL_INVALID_OUTPUT);        
        dsth = dst2;
        for (int i = 0; i < dst_gold.rows; ++i)       
            CHECK(memcmp(dst_gold2.ptr(i), dsth.ptr(i), dst_gold2.cols * dst_gold2.elemSize()) == 0, cvtest::TS::FAIL_INVALID_OUTPUT)
    }
};

TEST(BitwiseOperations, accuracy) { CV_GpuBitwiseTest test; test; }
