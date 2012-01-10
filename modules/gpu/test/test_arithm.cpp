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

#ifdef HAVE_CUDA

using namespace cvtest;
using namespace testing;

PARAM_TEST_CASE(ArithmTestBase, cv::gpu::DeviceInfo, MatType, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    int type;
    bool useRoi;

    cv::Size size;
    cv::Mat mat1; 
    cv::Mat mat2;
    cv::Scalar val;
        
    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        type = GET_PARAM(1);
        useRoi = GET_PARAM(2);

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));
        
        mat1 = randomMat(rng, size, type, 1, 16, false);
        mat2 = randomMat(rng, size, type, 1, 16, false);

        val = cv::Scalar(rng.uniform(0.1, 3.0), rng.uniform(0.1, 3.0), rng.uniform(0.1, 3.0), rng.uniform(0.1, 3.0));
    }
};

////////////////////////////////////////////////////////////////////////////////
// add

struct Add : ArithmTestBase {};

TEST_P(Add, Array) 
{    
    cv::Mat dst_gold;
    cv::add(mat1, mat2, dst_gold);

    cv::Mat dst;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpuRes;

        cv::gpu::add(loadMat(mat1, useRoi), loadMat(mat2, useRoi), gpuRes);

        gpuRes.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(Add, Scalar) 
{    
    cv::Mat dst_gold;
    cv::add(mat1, val, dst_gold);

    cv::Mat dst;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpuRes;

        cv::gpu::add(loadMat(mat1, useRoi), val, gpuRes);

        gpuRes.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

INSTANTIATE_TEST_CASE_P(Arithm, Add, Combine(
                        ALL_DEVICES,
                        Values(CV_8UC1, CV_16UC1, CV_32SC1, CV_32FC1),
                        USE_ROI));

////////////////////////////////////////////////////////////////////////////////
// subtract

struct Subtract : ArithmTestBase {};

TEST_P(Subtract, Array) 
{    
    cv::Mat dst_gold;
    cv::subtract(mat1, mat2, dst_gold);

    cv::Mat dst;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpuRes;

        cv::gpu::subtract(loadMat(mat1, useRoi), loadMat(mat2, useRoi), gpuRes);

        gpuRes.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(Subtract, Scalar) 
{    
    cv::Mat dst_gold;
    cv::subtract(mat1, val, dst_gold);

    cv::Mat dst;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpuRes;

        cv::gpu::subtract(loadMat(mat1, useRoi), val, gpuRes);

        gpuRes.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

INSTANTIATE_TEST_CASE_P(Arithm, Subtract, Combine(
                        ALL_DEVICES,
                        Values(CV_8UC1, CV_16UC1, CV_32SC1, CV_32FC1),
                        USE_ROI));

////////////////////////////////////////////////////////////////////////////////
// multiply

struct Multiply : ArithmTestBase {};

TEST_P(Multiply, Array) 
{    
    cv::Mat dst_gold;
    cv::multiply(mat1, mat2, dst_gold);

    cv::Mat dst;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpuRes;

        cv::gpu::multiply(loadMat(mat1, useRoi), loadMat(mat2, useRoi), gpuRes);

        gpuRes.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(Multiply, Scalar) 
{    
    cv::Mat dst_gold;
    cv::multiply(mat1, val, dst_gold);

    cv::Mat dst;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpuRes;

        cv::gpu::multiply(loadMat(mat1, useRoi), val, gpuRes);

        gpuRes.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

INSTANTIATE_TEST_CASE_P(Arithm, Multiply, Combine(
                        ALL_DEVICES,
                        Values(CV_8UC1, CV_16UC1, CV_32SC1, CV_32FC1),
                        USE_ROI));

////////////////////////////////////////////////////////////////////////////////
// divide

struct Divide : ArithmTestBase {};

TEST_P(Divide, Array) 
{    
    cv::Mat dst_gold;
    cv::divide(mat1, mat2, dst_gold);

    cv::Mat dst;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpuRes;

        cv::gpu::divide(loadMat(mat1, useRoi), loadMat(mat2, useRoi), gpuRes);

        gpuRes.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 1.0);
}

TEST_P(Divide, Scalar) 
{    
    cv::Mat dst_gold;
    cv::divide(mat1, val, dst_gold);

    cv::Mat dst;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpuRes;

        cv::gpu::divide(loadMat(mat1, useRoi), val, gpuRes);

        gpuRes.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

INSTANTIATE_TEST_CASE_P(Arithm, Divide, Combine(
                        ALL_DEVICES,
                        Values(CV_8UC1, CV_16UC1, CV_32SC1, CV_32FC1),
                        USE_ROI));

////////////////////////////////////////////////////////////////////////////////
// transpose

struct Transpose : ArithmTestBase {};

TEST_P(Transpose, Accuracy) 
{
    cv::Mat dst_gold;
    cv::transpose(mat1, dst_gold);

    cv::Mat dst;
    
    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpuRes;

        cv::gpu::transpose(loadMat(mat1, useRoi), gpuRes);

        gpuRes.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(Arithm, Transpose, Combine(
                        ALL_DEVICES,
                        Values(CV_8UC1, CV_8UC4, CV_8SC1, CV_8SC4, CV_16UC2, CV_16SC2, CV_32SC1, CV_32SC2, CV_32FC1, CV_32FC2, CV_64FC1),
                        USE_ROI));

////////////////////////////////////////////////////////////////////////////////
// absdiff

struct Absdiff : ArithmTestBase {};

TEST_P(Absdiff, Array) 
{    
    cv::Mat dst_gold;
    cv::absdiff(mat1, mat2, dst_gold);

    cv::Mat dst;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpuRes;

        cv::gpu::absdiff(loadMat(mat1, useRoi), loadMat(mat2, useRoi), gpuRes);

        gpuRes.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(Absdiff, Scalar) 
{    
    cv::Mat dst_gold;
    cv::absdiff(mat1, val, dst_gold);

    cv::Mat dst;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpuRes;

        cv::gpu::absdiff(loadMat(mat1, useRoi), val, gpuRes);

        gpuRes.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

INSTANTIATE_TEST_CASE_P(Arithm, Absdiff, Combine(
                        ALL_DEVICES,
                        Values(CV_8UC1, CV_16UC1, CV_32SC1, CV_32FC1),
                        USE_ROI));

////////////////////////////////////////////////////////////////////////////////
// compare

PARAM_TEST_CASE(Compare, cv::gpu::DeviceInfo, MatType, CmpCode, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    int type;
    int cmp_code;
    bool useRoi;

    cv::Size size;
    cv::Mat mat1, mat2;

    cv::Mat dst_gold;
        
    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        type = GET_PARAM(1);
        cmp_code = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));
        
        mat1 = randomMat(rng, size, type, 1, 16, false);
        mat2 = randomMat(rng, size, type, 1, 16, false);

        cv::compare(mat1, mat2, dst_gold, cmp_code);
    }
};

TEST_P(Compare, Accuracy) 
{
    cv::Mat dst;
    
    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpuRes;

        cv::gpu::compare(loadMat(mat1, useRoi), loadMat(mat2, useRoi), gpuRes, cmp_code);

        gpuRes.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(Arithm, Compare, Combine(
                        ALL_DEVICES,
                        Values(CV_8UC1, CV_16UC1, CV_32SC1),
                        Values((int) cv::CMP_EQ, (int) cv::CMP_GT, (int) cv::CMP_GE, (int) cv::CMP_LT, (int) cv::CMP_LE, (int) cv::CMP_NE),
                        USE_ROI));

////////////////////////////////////////////////////////////////////////////////
// meanStdDev

PARAM_TEST_CASE(MeanStdDev, cv::gpu::DeviceInfo, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    bool useRoi;

    cv::Size size;
    cv::Mat mat;

    cv::Scalar mean_gold;
    cv::Scalar stddev_gold;

    virtual void SetUp() 
    {
        devInfo = GET_PARAM(0);
        useRoi = GET_PARAM(1);

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));
        
        mat = randomMat(rng, size, CV_8UC1, 1, 255, false);

        cv::meanStdDev(mat, mean_gold, stddev_gold);
    }
};

TEST_P(MeanStdDev, Accuracy) 
{
    cv::Scalar mean;
    cv::Scalar stddev;
    
    ASSERT_NO_THROW(
        cv::gpu::meanStdDev(loadMat(mat, useRoi), mean, stddev);
    );

    EXPECT_NEAR(mean_gold[0], mean[0], 1e-5);
    EXPECT_NEAR(mean_gold[1], mean[1], 1e-5);
    EXPECT_NEAR(mean_gold[2], mean[2], 1e-5);
    EXPECT_NEAR(mean_gold[3], mean[3], 1e-5);

    EXPECT_NEAR(stddev_gold[0], stddev[0], 1e-5);
    EXPECT_NEAR(stddev_gold[1], stddev[1], 1e-5);
    EXPECT_NEAR(stddev_gold[2], stddev[2], 1e-5);
    EXPECT_NEAR(stddev_gold[3], stddev[3], 1e-5);
}

INSTANTIATE_TEST_CASE_P(Arithm, MeanStdDev, Combine(
                        ALL_DEVICES,
                        USE_ROI));

////////////////////////////////////////////////////////////////////////////////
// normDiff

PARAM_TEST_CASE(NormDiff, cv::gpu::DeviceInfo, NormCode, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    int normCode;
    bool useRoi;

    cv::Size size;
    cv::Mat mat1, mat2;

    double norm_gold;

    virtual void SetUp() 
    {
        devInfo = GET_PARAM(0);
        normCode = GET_PARAM(1);
        useRoi = GET_PARAM(2);

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));
        
        mat1 = randomMat(rng, size, CV_8UC1, 1, 255, false);
        mat2 = randomMat(rng, size, CV_8UC1, 1, 255, false);

        norm_gold = cv::norm(mat1, mat2, normCode);
    }
};

TEST_P(NormDiff, Accuracy) 
{    
    double norm;
    
    ASSERT_NO_THROW(
        norm = cv::gpu::norm(loadMat(mat1, useRoi), loadMat(mat2, useRoi), normCode);
    );

    EXPECT_NEAR(norm_gold, norm, 1e-6);
}

INSTANTIATE_TEST_CASE_P(Arithm, NormDiff, Combine(
                        ALL_DEVICES,
                        Values((int) cv::NORM_INF, (int) cv::NORM_L1, (int) cv::NORM_L2),
                        USE_ROI));

////////////////////////////////////////////////////////////////////////////////
// flip

PARAM_TEST_CASE(Flip, cv::gpu::DeviceInfo, MatType, FlipCode, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    int type;
    int flip_code;
    bool useRoi;

    cv::Size size;
    cv::Mat mat;

    cv::Mat dst_gold;

    virtual void SetUp() 
    {
        devInfo = GET_PARAM(0);
        type = GET_PARAM(1);
        flip_code = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));
        
        mat = randomMat(rng, size, type, 1, 255, false);

        cv::flip(mat, dst_gold, flip_code);
    }
};

TEST_P(Flip, Accuracy) 
{    
    cv::Mat dst;
    
    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpu_res;

        cv::gpu::flip(loadMat(mat, useRoi), gpu_res, flip_code);

        gpu_res.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(Arithm, Flip, Combine(
                        ALL_DEVICES,
                        Values(CV_8UC1, CV_8UC4),
                        Values((int)FLIP_BOTH, (int)FLIP_X, (int)FLIP_Y),
                        USE_ROI));

////////////////////////////////////////////////////////////////////////////////
// LUT

PARAM_TEST_CASE(LUT, cv::gpu::DeviceInfo, MatType, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    int type;
    bool useRoi;

    cv::Size size;
    cv::Mat mat;
    cv::Mat lut;

    cv::Mat dst_gold;

    virtual void SetUp() 
    {
        devInfo = GET_PARAM(0);
        type = GET_PARAM(1);
        useRoi = GET_PARAM(2);

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));
        
        mat = randomMat(rng, size, type, 1, 255, false);
        lut = randomMat(rng, cv::Size(256, 1), CV_8UC1, 100, 200, false);

        cv::LUT(mat, lut, dst_gold);
    }
};

TEST_P(LUT, Accuracy) 
{
    cv::Mat dst;
    
    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpu_res;

        cv::gpu::LUT(loadMat(mat, useRoi), lut, gpu_res);

        gpu_res.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(Arithm, LUT, Combine(
                        ALL_DEVICES,
                        Values(CV_8UC1, CV_8UC3),
                        USE_ROI));

////////////////////////////////////////////////////////////////////////////////
// exp

PARAM_TEST_CASE(Exp, cv::gpu::DeviceInfo, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    bool useRoi;

    cv::Size size;
    cv::Mat mat;

    cv::Mat dst_gold;

    virtual void SetUp() 
    {
        devInfo = GET_PARAM(0);
        useRoi = GET_PARAM(1);

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));

        mat = randomMat(rng, size, CV_32FC1, -10.0, 2.0, false);        

        cv::exp(mat, dst_gold);
    }
};

TEST_P(Exp, Accuracy) 
{
    cv::Mat dst;
    
    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpu_res;

        cv::gpu::exp(loadMat(mat, useRoi), gpu_res);

        gpu_res.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

INSTANTIATE_TEST_CASE_P(Arithm, Exp, Combine(
                        ALL_DEVICES,
                        USE_ROI));

////////////////////////////////////////////////////////////////////////////////
// pow

PARAM_TEST_CASE(Pow, cv::gpu::DeviceInfo, MatType, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    int type;
    bool useRoi;

    double power;
    cv::Size size;
    cv::Mat mat;

    cv::Mat dst_gold;

    virtual void SetUp() 
    {
        devInfo = GET_PARAM(0);
        type = GET_PARAM(1);
        useRoi = GET_PARAM(2);

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));

        mat = randomMat(rng, size, type, 0.0, 100.0, false);        

        if (mat.depth() == CV_32F)
            power = rng.uniform(1.2f, 3.f);
        else
        {
            int ipower = rng.uniform(2, 8);
            power = (float)ipower;
        }

        cv::pow(mat, power, dst_gold);
    }
};

TEST_P(Pow, Accuracy) 
{
    cv::Mat dst;
    
    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpu_res;

        cv::gpu::pow(loadMat(mat, useRoi), power, gpu_res);

        gpu_res.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 2);
}

INSTANTIATE_TEST_CASE_P(Arithm, Pow, Combine(
                        ALL_DEVICES,
                        Values(CV_32F, CV_32FC3),
                        USE_ROI));

////////////////////////////////////////////////////////////////////////////////
// log

PARAM_TEST_CASE(Log, cv::gpu::DeviceInfo, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    bool useRoi;

    cv::Size size;
    cv::Mat mat;

    cv::Mat dst_gold;

    virtual void SetUp() 
    {
        devInfo = GET_PARAM(0);
        useRoi = GET_PARAM(1);

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));

        mat = randomMat(rng, size, CV_32FC1, 0.0, 100.0, false);        

        cv::log(mat, dst_gold);
    }
};

TEST_P(Log, Accuracy) 
{
    cv::Mat dst;
    
    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpu_res;

        cv::gpu::log(loadMat(mat, useRoi), gpu_res);

        gpu_res.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

INSTANTIATE_TEST_CASE_P(Arithm, Log, Combine(
                        ALL_DEVICES,
                        USE_ROI));

////////////////////////////////////////////////////////////////////////////////
// magnitude

PARAM_TEST_CASE(Magnitude, cv::gpu::DeviceInfo, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    bool useRoi;

    cv::Size size;
    cv::Mat mat1, mat2;

    cv::Mat dst_gold;

    virtual void SetUp() 
    {
        devInfo = GET_PARAM(0);
        useRoi = GET_PARAM(1);

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));

        mat1 = randomMat(rng, size, CV_32FC1, 0.0, 100.0, false);
        mat2 = randomMat(rng, size, CV_32FC1, 0.0, 100.0, false);       

        cv::magnitude(mat1, mat2, dst_gold);
    }
};

TEST_P(Magnitude, Accuracy) 
{
    cv::Mat dst;
    
    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpu_res;

        cv::gpu::magnitude(loadMat(mat1, useRoi), loadMat(mat2, useRoi), gpu_res);

        gpu_res.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-4);
}

INSTANTIATE_TEST_CASE_P(Arithm, Magnitude, Combine(
                        ALL_DEVICES,
                        USE_ROI));

////////////////////////////////////////////////////////////////////////////////
// phase

PARAM_TEST_CASE(Phase, cv::gpu::DeviceInfo, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    bool useRoi;

    cv::Size size;
    cv::Mat mat1, mat2;

    cv::Mat dst_gold;

    virtual void SetUp() 
    {
        devInfo = GET_PARAM(0);
        useRoi = GET_PARAM(1);

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));

        mat1 = randomMat(rng, size, CV_32FC1, 0.0, 100.0, false);
        mat2 = randomMat(rng, size, CV_32FC1, 0.0, 100.0, false);       

        cv::phase(mat1, mat2, dst_gold);
    }
};

TEST_P(Phase, Accuracy) 
{
    cv::Mat dst;
    
    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpu_res;

        cv::gpu::phase(loadMat(mat1, useRoi), loadMat(mat2, useRoi), gpu_res);

        gpu_res.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-3);
}

INSTANTIATE_TEST_CASE_P(Arithm, Phase, Combine(
                        ALL_DEVICES,
                        USE_ROI));

////////////////////////////////////////////////////////////////////////////////
// cartToPolar

PARAM_TEST_CASE(CartToPolar, cv::gpu::DeviceInfo, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    bool useRoi;

    cv::Size size;
    cv::Mat mat1, mat2;

    cv::Mat mag_gold;
    cv::Mat angle_gold;

    virtual void SetUp() 
    {
        devInfo = GET_PARAM(0);
        useRoi = GET_PARAM(1);

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));

        mat1 = randomMat(rng, size, CV_32FC1, -100.0, 100.0, false);
        mat2 = randomMat(rng, size, CV_32FC1, -100.0, 100.0, false);       

        cv::cartToPolar(mat1, mat2, mag_gold, angle_gold);
    }
};

TEST_P(CartToPolar, Accuracy) 
{
    cv::Mat mag, angle;
    
    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpuMag;
        cv::gpu::GpuMat gpuAngle;

        cv::gpu::cartToPolar(loadMat(mat1, useRoi), loadMat(mat2, useRoi), gpuMag, gpuAngle);

        gpuMag.download(mag);
        gpuAngle.download(angle);
    );

    EXPECT_MAT_NEAR(mag_gold, mag, 1e-4);
    EXPECT_MAT_NEAR(angle_gold, angle, 1e-3);
}

INSTANTIATE_TEST_CASE_P(Arithm, CartToPolar, Combine(
                        ALL_DEVICES,
                        USE_ROI));

////////////////////////////////////////////////////////////////////////////////
// polarToCart

PARAM_TEST_CASE(PolarToCart, cv::gpu::DeviceInfo, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    bool useRoi;

    cv::Size size;
    cv::Mat mag;
    cv::Mat angle;

    cv::Mat x_gold;
    cv::Mat y_gold;

    virtual void SetUp() 
    {
        devInfo = GET_PARAM(0);
        useRoi = GET_PARAM(1);

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));

        mag = randomMat(rng, size, CV_32FC1, -100.0, 100.0, false);
        angle = randomMat(rng, size, CV_32FC1, 0.0, 2.0 * CV_PI, false);       

        cv::polarToCart(mag, angle, x_gold, y_gold);
    }
};

TEST_P(PolarToCart, Accuracy) 
{
    cv::Mat x, y;
    
    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpuX;
        cv::gpu::GpuMat gpuY;

        cv::gpu::polarToCart(loadMat(mag, useRoi), loadMat(angle, useRoi), gpuX, gpuY);

        gpuX.download(x);
        gpuY.download(y);
    );

    EXPECT_MAT_NEAR(x_gold, x, 1e-4);
    EXPECT_MAT_NEAR(y_gold, y, 1e-4);
}

INSTANTIATE_TEST_CASE_P(Arithm, PolarToCart, Combine(
                        ALL_DEVICES,
                        USE_ROI));

////////////////////////////////////////////////////////////////////////////////
// minMax

PARAM_TEST_CASE(MinMax, cv::gpu::DeviceInfo, MatType, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    int type;
    bool useRoi;

    cv::Size size;
    cv::Mat mat;
    cv::Mat mask;

    double minVal_gold;
    double maxVal_gold;

    virtual void SetUp() 
    {
        devInfo = GET_PARAM(0);
        type = GET_PARAM(1);
        useRoi = GET_PARAM(2);

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));

        mat = randomMat(rng, size, type, 0.0, 127.0, false);
        mask = randomMat(rng, size, CV_8UC1, 0, 2, false);

        if (type != CV_8S)
        {
            cv::minMaxLoc(mat, &minVal_gold, &maxVal_gold, 0, 0, mask);
        }
        else 
        {
            // OpenCV's minMaxLoc doesn't support CV_8S type 
            minVal_gold = std::numeric_limits<double>::max();
            maxVal_gold = -std::numeric_limits<double>::max();
            for (int i = 0; i < mat.rows; ++i)
            {
                const signed char* mat_row = mat.ptr<signed char>(i);
                const unsigned char* mask_row = mask.ptr<unsigned char>(i);
                for (int j = 0; j < mat.cols; ++j)
                {
                    if (mask_row[j]) 
                    { 
                        signed char val = mat_row[j];
                        if (val < minVal_gold) minVal_gold = val;
                        if (val > maxVal_gold) maxVal_gold = val; 
                    }
                }
            }
        }
    }
};

TEST_P(MinMax, Accuracy) 
{
    if (type == CV_64F && !supportFeature(devInfo,  cv::gpu::NATIVE_DOUBLE))
        return;

    double minVal, maxVal;
    
    ASSERT_NO_THROW(
        cv::gpu::minMax(loadMat(mat, useRoi), &minVal, &maxVal, loadMat(mask, useRoi));
    );

    EXPECT_DOUBLE_EQ(minVal_gold, minVal);
    EXPECT_DOUBLE_EQ(maxVal_gold, maxVal);
}

INSTANTIATE_TEST_CASE_P(Arithm, MinMax, Combine(
                        ALL_DEVICES,
                        Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F),
                        USE_ROI));

////////////////////////////////////////////////////////////////////////////////
// minMaxLoc

PARAM_TEST_CASE(MinMaxLoc, cv::gpu::DeviceInfo, MatType, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    int type;
    bool useRoi;

    cv::Size size;
    cv::Mat mat;
    cv::Mat mask;

    double minVal_gold;
    double maxVal_gold;
    cv::Point minLoc_gold;
    cv::Point maxLoc_gold;

    virtual void SetUp() 
    {
        devInfo = GET_PARAM(0);
        type = GET_PARAM(1);
        useRoi = GET_PARAM(2);

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));

        mat = randomMat(rng, size, type, 0.0, 127.0, false);
        mask = randomMat(rng, size, CV_8UC1, 0, 2, false);

        if (type != CV_8S)
        {
            cv::minMaxLoc(mat, &minVal_gold, &maxVal_gold, &minLoc_gold, &maxLoc_gold, mask);
        }
        else 
        {
            // OpenCV's minMaxLoc doesn't support CV_8S type 
            minVal_gold = std::numeric_limits<double>::max();
            maxVal_gold = -std::numeric_limits<double>::max();
            for (int i = 0; i < mat.rows; ++i)
            {
                const signed char* mat_row = mat.ptr<signed char>(i);
                const unsigned char* mask_row = mask.ptr<unsigned char>(i);
                for (int j = 0; j < mat.cols; ++j)
                {
                    if (mask_row[j]) 
                    { 
                        signed char val = mat_row[j];
                        if (val < minVal_gold) { minVal_gold = val; minLoc_gold = cv::Point(j, i); }
                        if (val > maxVal_gold) { maxVal_gold = val; maxLoc_gold = cv::Point(j, i); }
                    }
                }
            }
        }
    }
};

TEST_P(MinMaxLoc, Accuracy) 
{
    if (type == CV_64F && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
        return;

    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    
    ASSERT_NO_THROW(
        cv::gpu::minMaxLoc(loadMat(mat, useRoi), &minVal, &maxVal, &minLoc, &maxLoc, loadMat(mask, useRoi));
    );

    EXPECT_DOUBLE_EQ(minVal_gold, minVal);
    EXPECT_DOUBLE_EQ(maxVal_gold, maxVal);

    int cmpMinVals = memcmp(mat.data + minLoc_gold.y * mat.step + minLoc_gold.x * mat.elemSize(), 
                            mat.data + minLoc.y * mat.step + minLoc.x * mat.elemSize(), 
                            mat.elemSize());
    int cmpMaxVals = memcmp(mat.data + maxLoc_gold.y * mat.step + maxLoc_gold.x * mat.elemSize(), 
                            mat.data + maxLoc.y * mat.step + maxLoc.x * mat.elemSize(), 
                            mat.elemSize());

    EXPECT_EQ(0, cmpMinVals);
    EXPECT_EQ(0, cmpMaxVals);
}

INSTANTIATE_TEST_CASE_P(Arithm, MinMaxLoc, Combine(
                        ALL_DEVICES,
                        Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F),
                        USE_ROI));

////////////////////////////////////////////////////////////////////////////
// countNonZero

PARAM_TEST_CASE(CountNonZero, cv::gpu::DeviceInfo, MatType, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    int type;
    bool useRoi;

    cv::Size size;
    cv::Mat mat;

    int n_gold;

    virtual void SetUp() 
    {
        devInfo = GET_PARAM(0);
        type = GET_PARAM(1);
        useRoi = GET_PARAM(2);

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));

        cv::Mat matBase = randomMat(rng, size, CV_8U, 0.0, 1.0, false);
        matBase.convertTo(mat, type);

        n_gold = cv::countNonZero(mat);
    }
};

TEST_P(CountNonZero, Accuracy) 
{
    if (type == CV_64F && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
        return;

    int n;
    
    ASSERT_NO_THROW(
        n = cv::gpu::countNonZero(loadMat(mat, useRoi));
    );

    ASSERT_EQ(n_gold, n);
}

INSTANTIATE_TEST_CASE_P(Arithm, CountNonZero, Combine(
                        ALL_DEVICES,
                        Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F),
                        USE_ROI));

//////////////////////////////////////////////////////////////////////////////
// sum

PARAM_TEST_CASE(Sum, cv::gpu::DeviceInfo, MatType, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    int type;
    bool useRoi;

    cv::Size size;
    cv::Mat mat;

    virtual void SetUp() 
    {
        devInfo = GET_PARAM(0);
        type = GET_PARAM(1);
        useRoi = GET_PARAM(2);

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));

        mat = randomMat(rng, size, CV_8U, 0.0, 10.0, false);
    }
};

TEST_P(Sum, Simple) 
{
    if (type == CV_64F && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
        return;

    cv::Scalar sum_gold = cv::sum(mat);

    cv::Scalar sum;
    
    ASSERT_NO_THROW(
        sum = cv::gpu::sum(loadMat(mat, useRoi));
    );

    EXPECT_NEAR(sum[0], sum_gold[0], mat.size().area() * 1e-5);
    EXPECT_NEAR(sum[1], sum_gold[1], mat.size().area() * 1e-5);
    EXPECT_NEAR(sum[2], sum_gold[2], mat.size().area() * 1e-5);
    EXPECT_NEAR(sum[3], sum_gold[3], mat.size().area() * 1e-5);
}

TEST_P(Sum, Abs) 
{
    if (type == CV_64F && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
        return;

    cv::Scalar sum_gold = cv::norm(mat, cv::NORM_L1);

    cv::Scalar sum;
    
    ASSERT_NO_THROW(
        sum = cv::gpu::absSum(loadMat(mat, useRoi));
    );

    EXPECT_NEAR(sum[0], sum_gold[0], mat.size().area() * 1e-5);
    EXPECT_NEAR(sum[1], sum_gold[1], mat.size().area() * 1e-5);
    EXPECT_NEAR(sum[2], sum_gold[2], mat.size().area() * 1e-5);
    EXPECT_NEAR(sum[3], sum_gold[3], mat.size().area() * 1e-5);
}

TEST_P(Sum, Sqr) 
{
    if (type == CV_64F && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
        return;

    cv::Mat sqrmat;
    multiply(mat, mat, sqrmat);
    cv::Scalar sum_gold = sum(sqrmat);

    cv::Scalar sum;
    
    ASSERT_NO_THROW(
        sum = cv::gpu::sqrSum(loadMat(mat, useRoi));
    );

    EXPECT_NEAR(sum[0], sum_gold[0], mat.size().area() * 1e-5);
    EXPECT_NEAR(sum[1], sum_gold[1], mat.size().area() * 1e-5);
    EXPECT_NEAR(sum[2], sum_gold[2], mat.size().area() * 1e-5);
    EXPECT_NEAR(sum[3], sum_gold[3], mat.size().area() * 1e-5);
}

INSTANTIATE_TEST_CASE_P(Arithm, Sum, Combine(
                        ALL_DEVICES,
                        Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F),
                        USE_ROI));

//////////////////////////////////////////////////////////////////////////////
// bitwise

PARAM_TEST_CASE(Bitwise, cv::gpu::DeviceInfo, MatType)
{
    cv::gpu::DeviceInfo devInfo;
    int type;

    cv::Size size;
    cv::Mat mat1;
    cv::Mat mat2;

    virtual void SetUp() 
    {
        devInfo = GET_PARAM(0);
        type = GET_PARAM(1);

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = cvtest::TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));

        mat1.create(size, type);
        mat2.create(size, type);
        
        for (int i = 0; i < mat1.rows; ++i)
        {
            cv::Mat row1(1, static_cast<int>(mat1.cols * mat1.elemSize()), CV_8U, (void*)mat1.ptr(i));
            rng.fill(row1, cv::RNG::UNIFORM, cv::Scalar(0), cv::Scalar(255));

            cv::Mat row2(1, static_cast<int>(mat2.cols * mat2.elemSize()), CV_8U, (void*)mat2.ptr(i));
            rng.fill(row2, cv::RNG::UNIFORM, cv::Scalar(0), cv::Scalar(255));
        }
    }
};

TEST_P(Bitwise, Not) 
{
    if (mat1.depth() == CV_64F && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
        return;

    cv::Mat dst_gold = ~mat1;

    cv::Mat dst;
    
    ASSERT_NO_THROW(
        cv::gpu::GpuMat dev_dst;

        cv::gpu::bitwise_not(loadMat(mat1), dev_dst);

        dev_dst.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(Bitwise, Or) 
{
    if (mat1.depth() == CV_64F && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
        return;

    cv::Mat dst_gold = mat1 | mat2;

    cv::Mat dst;
    
    ASSERT_NO_THROW(
        cv::gpu::GpuMat dev_dst;

        cv::gpu::bitwise_or(loadMat(mat1), loadMat(mat2), dev_dst);

        dev_dst.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(Bitwise, And) 
{
    if (mat1.depth() == CV_64F && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
        return;

    cv::Mat dst_gold = mat1 & mat2;

    cv::Mat dst;
    
    ASSERT_NO_THROW(
        cv::gpu::GpuMat dev_dst;

        cv::gpu::bitwise_and(loadMat(mat1), loadMat(mat2), dev_dst);

        dev_dst.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(Bitwise, Xor) 
{
    if (mat1.depth() == CV_64F && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
        return;

    cv::Mat dst_gold = mat1 ^ mat2;

    cv::Mat dst;
    
    ASSERT_NO_THROW(
        cv::gpu::GpuMat dev_dst;

        cv::gpu::bitwise_xor(loadMat(mat1), loadMat(mat2), dev_dst);

        dev_dst.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(Arithm, Bitwise, Combine(
                        ALL_DEVICES,
                        ALL_TYPES));

//////////////////////////////////////////////////////////////////////////////
// addWeighted

PARAM_TEST_CASE(AddWeighted, cv::gpu::DeviceInfo, MatType, MatType, MatType, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    int type1;
    int type2;
    int dtype;
    bool useRoi;

    cv::Size size;
    cv::Mat src1;
    cv::Mat src2;
    double alpha;
    double beta;
    double gamma;

    cv::Mat dst_gold;

    virtual void SetUp() 
    {
        devInfo = GET_PARAM(0);
        type1 = GET_PARAM(1);
        type2 = GET_PARAM(2);
        dtype = GET_PARAM(3);
        useRoi = GET_PARAM(4);

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));

        src1 = randomMat(rng, size, type1, 0.0, 255.0, false);
        src2 = randomMat(rng, size, type2, 0.0, 255.0, false);

        alpha = rng.uniform(-10.0, 10.0);
        beta = rng.uniform(-10.0, 10.0);
        gamma = rng.uniform(-10.0, 10.0);

        cv::addWeighted(src1, alpha, src2, beta, gamma, dst_gold, dtype);
    }
};

TEST_P(AddWeighted, Accuracy) 
{
    if ((src1.depth() == CV_64F || src2.depth() == CV_64F || dst_gold.depth() == CV_64F) && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
        return;

    cv::Mat dst;
    
    ASSERT_NO_THROW(
        cv::gpu::GpuMat dev_dst;

        cv::gpu::addWeighted(loadMat(src1, useRoi), alpha, loadMat(src2, useRoi), beta, gamma, dev_dst, dtype);

        dev_dst.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, dtype < CV_32F ? 1.0 : 1e-12);
}

INSTANTIATE_TEST_CASE_P(Arithm, AddWeighted, Combine(
                        ALL_DEVICES,
                        TYPES(CV_8U, CV_64F, 1, 1),
                        TYPES(CV_8U, CV_64F, 1, 1),
                        TYPES(CV_8U, CV_64F, 1, 1),
                        USE_ROI));

//////////////////////////////////////////////////////////////////////////////
// reduce

PARAM_TEST_CASE(Reduce, cv::gpu::DeviceInfo, MatType, int, ReduceOp, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    int type;
    int dim;
    int reduceOp;    
    bool useRoi;

    cv::Size size;
    cv::Mat src;

    cv::Mat dst_gold;

    virtual void SetUp() 
    {
        devInfo = GET_PARAM(0);
        type = GET_PARAM(1);
        dim = GET_PARAM(2);
        reduceOp = GET_PARAM(3);
        useRoi = GET_PARAM(4);

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 400), rng.uniform(100, 400));

        src = randomMat(rng, size, type, 0.0, 255.0, false);

        cv::reduce(src, dst_gold, dim, reduceOp, reduceOp == CV_REDUCE_SUM || reduceOp == CV_REDUCE_AVG ? CV_32F : CV_MAT_DEPTH(type));

        if (dim == 1)
        {
            dst_gold.cols = dst_gold.rows;
            dst_gold.rows = 1;
            dst_gold.step = dst_gold.cols * dst_gold.elemSize();
        }
    }
};

TEST_P(Reduce, Accuracy) 
{
    cv::Mat dst;
    
    ASSERT_NO_THROW(
        cv::gpu::GpuMat dev_dst;

        cv::gpu::reduce(loadMat(src, useRoi), dev_dst, dim, reduceOp, reduceOp == CV_REDUCE_SUM || reduceOp == CV_REDUCE_AVG ? CV_32F : CV_MAT_DEPTH(type));

        dev_dst.download(dst);
    );

    double norm = reduceOp == CV_REDUCE_SUM || reduceOp == CV_REDUCE_AVG ? 1e-1 : 0.0;
    EXPECT_MAT_NEAR(dst_gold, dst, norm);
}

INSTANTIATE_TEST_CASE_P(Arithm, Reduce, Combine(
                        ALL_DEVICES,
                        Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_16UC3, CV_16UC4, CV_32FC1, CV_32FC3, CV_32FC4),
                        Values(0, 1),
                        Values((int)CV_REDUCE_SUM, (int)CV_REDUCE_AVG, (int)CV_REDUCE_MAX, (int)CV_REDUCE_MIN),
                        USE_ROI));

//////////////////////////////////////////////////////////////////////////////
// gemm

PARAM_TEST_CASE(GEMM, cv::gpu::DeviceInfo, MatType, GemmFlags, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    int type;
    int flags;
    bool useRoi;

    int size;
    cv::Mat src1;
    cv::Mat src2;
    cv::Mat src3;
    double alpha;
    double beta;

    cv::Mat dst_gold;

    virtual void SetUp() 
    {
        devInfo = GET_PARAM(0);
        type = GET_PARAM(1);
        flags = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = TS::ptr()->get_rng();

        size = rng.uniform(100, 200);

        src1 = randomMat(rng, cv::Size(size, size), type, -10.0, 10.0, false);
        src2 = randomMat(rng, cv::Size(size, size), type, -10.0, 10.0, false);
        src3 = randomMat(rng, cv::Size(size, size), type, -10.0, 10.0, false);
        alpha = rng.uniform(-10.0, 10.0);
        beta = rng.uniform(-10.0, 10.0);

        cv::gemm(src1, src2, alpha, src3, beta, dst_gold, flags);
    }
};

TEST_P(GEMM, Accuracy) 
{
    cv::Mat dst;
    
    ASSERT_NO_THROW(
        cv::gpu::GpuMat dev_dst;

        cv::gpu::gemm(loadMat(src1, useRoi), loadMat(src2, useRoi), alpha, loadMat(src3, useRoi), beta, dev_dst, flags);

        dev_dst.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-1);
}

INSTANTIATE_TEST_CASE_P(Arithm, GEMM, Combine(
                        ALL_DEVICES,
                        Values(CV_32FC1, CV_32FC2),
                        Values(0, (int) cv::GEMM_1_T, (int) cv::GEMM_2_T, (int) cv::GEMM_3_T),
                        USE_ROI));

#endif // HAVE_CUDA
