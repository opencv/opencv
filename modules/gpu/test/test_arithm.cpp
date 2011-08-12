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

struct ArithmTest : testing::TestWithParam< std::tr1::tuple<cv::gpu::DeviceInfo, int> >
{
    cv::gpu::DeviceInfo devInfo;
    int type;

    cv::Size size;
    cv::Mat mat1, mat2;
        
    virtual void SetUp()
    {
        devInfo = std::tr1::get<0>(GetParam());
        type = std::tr1::get<1>(GetParam());

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = cvtest::TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));
        
        mat1 = cvtest::randomMat(rng, size, type, 1, 16, false);
        mat2 = cvtest::randomMat(rng, size, type, 1, 16, false);
    }
};

////////////////////////////////////////////////////////////////////////////////
// add

struct AddArray : ArithmTest {};

TEST_P(AddArray, Accuracy) 
{
    PRINT_PARAM(devInfo);
    PRINT_TYPE(type);
    PRINT_PARAM(size);
    
    cv::Mat dst_gold;
    cv::add(mat1, mat2, dst_gold);

    cv::Mat dst;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpuRes;

        cv::gpu::add(cv::gpu::GpuMat(mat1), cv::gpu::GpuMat(mat2), gpuRes);

        gpuRes.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(Arithm, AddArray, testing::Combine(
                        testing::ValuesIn(devices()),
                        testing::Values(CV_8UC1, CV_8UC4, CV_32SC1, CV_32FC1)));

struct AddScalar : ArithmTest {};

TEST_P(AddScalar, Accuracy) 
{
    PRINT_PARAM(devInfo);
    PRINT_TYPE(type);
    PRINT_PARAM(size);

    cv::RNG& rng = cvtest::TS::ptr()->get_rng();

    cv::Scalar val(rng.uniform(0.1, 3.0), rng.uniform(0.1, 3.0));

    PRINT_PARAM(val);
    
    cv::Mat dst_gold;
    cv::add(mat1, val, dst_gold);

    cv::Mat dst;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpuRes;

        cv::gpu::add(cv::gpu::GpuMat(mat1), val, gpuRes);

        gpuRes.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

INSTANTIATE_TEST_CASE_P(Arithm, AddScalar, testing::Combine(
                        testing::ValuesIn(devices()),
                        testing::Values(CV_32FC1, CV_32FC2)));

////////////////////////////////////////////////////////////////////////////////
// subtract

struct SubtractArray : ArithmTest {};

TEST_P(SubtractArray, Accuracy) 
{
    PRINT_PARAM(devInfo);
    PRINT_TYPE(type);
    PRINT_PARAM(size);
    
    cv::Mat dst_gold;
    cv::subtract(mat1, mat2, dst_gold);

    cv::Mat dst;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpuRes;

        cv::gpu::subtract(cv::gpu::GpuMat(mat1), cv::gpu::GpuMat(mat2), gpuRes);

        gpuRes.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(Arithm, SubtractArray, testing::Combine(
                        testing::ValuesIn(devices()),
                        testing::Values(CV_8UC1, CV_8UC4, CV_32SC1, CV_32FC1)));

struct SubtractScalar : ArithmTest {};

TEST_P(SubtractScalar, Accuracy) 
{
    PRINT_PARAM(devInfo);
    PRINT_TYPE(type);
    PRINT_PARAM(size);

    cv::RNG& rng = cvtest::TS::ptr()->get_rng();

    cv::Scalar val(rng.uniform(0.1, 3.0), rng.uniform(0.1, 3.0));

    PRINT_PARAM(val);
    
    cv::Mat dst_gold;
    cv::subtract(mat1, val, dst_gold);

    cv::Mat dst;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpuRes;

        cv::gpu::subtract(cv::gpu::GpuMat(mat1), val, gpuRes);

        gpuRes.download(dst);
    );

    ASSERT_LE(checkNorm(dst_gold, dst), 1e-5);
}

INSTANTIATE_TEST_CASE_P(Arithm, SubtractScalar, testing::Combine(
                        testing::ValuesIn(devices()),
                        testing::Values(CV_32FC1, CV_32FC2)));

////////////////////////////////////////////////////////////////////////////////
// multiply

struct MultiplyArray : ArithmTest {};

TEST_P(MultiplyArray, Accuracy) 
{
    PRINT_PARAM(devInfo);
    PRINT_TYPE(type);
    PRINT_PARAM(size);
    
    cv::Mat dst_gold;
    cv::multiply(mat1, mat2, dst_gold);

    cv::Mat dst;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpuRes;

        cv::gpu::multiply(cv::gpu::GpuMat(mat1), cv::gpu::GpuMat(mat2), gpuRes);

        gpuRes.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(Arithm, MultiplyArray, testing::Combine(
                        testing::ValuesIn(devices()),
                        testing::Values(CV_8UC1, CV_8UC4, CV_32SC1, CV_32FC1)));

struct MultiplyScalar : ArithmTest {};

TEST_P(MultiplyScalar, Accuracy) 
{
    PRINT_PARAM(devInfo);
    PRINT_TYPE(type);
    PRINT_PARAM(size);

    cv::RNG& rng = cvtest::TS::ptr()->get_rng();

    cv::Scalar val(rng.uniform(0.1, 3.0), rng.uniform(0.1, 3.0));

    PRINT_PARAM(val);
    
    cv::Mat dst_gold;
    cv::multiply(mat1, val, dst_gold);

    cv::Mat dst;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpuRes;

        cv::gpu::multiply(cv::gpu::GpuMat(mat1), val, gpuRes);

        gpuRes.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

INSTANTIATE_TEST_CASE_P(Arithm, MultiplyScalar, testing::Combine(
                        testing::ValuesIn(devices()),
                        testing::Values(CV_32FC1)));

////////////////////////////////////////////////////////////////////////////////
// divide

struct DivideArray : ArithmTest {};

TEST_P(DivideArray, Accuracy) 
{
    PRINT_PARAM(devInfo);
    PRINT_TYPE(type);
    PRINT_PARAM(size);
    
    cv::Mat dst_gold;
    cv::divide(mat1, mat2, dst_gold);

    cv::Mat dst;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpuRes;

        cv::gpu::divide(cv::gpu::GpuMat(mat1), cv::gpu::GpuMat(mat2), gpuRes);

        gpuRes.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 1.0);
}

INSTANTIATE_TEST_CASE_P(Arithm, DivideArray, testing::Combine(
                        testing::ValuesIn(devices()),
                        testing::Values(CV_8UC1, CV_8UC4, CV_32SC1, CV_32FC1)));

struct DivideScalar : ArithmTest {};

TEST_P(DivideScalar, Accuracy) 
{
    PRINT_PARAM(devInfo);
    PRINT_TYPE(type);
    PRINT_PARAM(size);

    cv::RNG& rng = cvtest::TS::ptr()->get_rng();

    cv::Scalar val(rng.uniform(0.1, 3.0), rng.uniform(0.1, 3.0));

    PRINT_PARAM(val);
    
    cv::Mat dst_gold;
    cv::divide(mat1, val, dst_gold);

    cv::Mat dst;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpuRes;

        cv::gpu::divide(cv::gpu::GpuMat(mat1), val, gpuRes);

        gpuRes.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

INSTANTIATE_TEST_CASE_P(Arithm, DivideScalar, testing::Combine(
                        testing::ValuesIn(devices()),
                        testing::Values(CV_32FC1)));

////////////////////////////////////////////////////////////////////////////////
// transpose

struct Transpose : ArithmTest {};

TEST_P(Transpose, Accuracy) 
{
    PRINT_PARAM(devInfo);
    PRINT_TYPE(type);
    PRINT_PARAM(size);

    cv::Mat dst_gold;
    cv::transpose(mat1, dst_gold);

    cv::Mat dst;
    
    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpuRes;

        cv::gpu::transpose(cv::gpu::GpuMat(mat1), gpuRes);

        gpuRes.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(Arithm, Transpose, testing::Combine(
                        testing::ValuesIn(devices()),
                        testing::Values(CV_8UC1, CV_8UC4, CV_8SC1, CV_8SC4, CV_16UC2, CV_16SC2, CV_32SC1, CV_32SC2, CV_32FC1, CV_32FC2, CV_64FC1)));

////////////////////////////////////////////////////////////////////////////////
// absdiff

struct AbsdiffArray : ArithmTest {};

TEST_P(AbsdiffArray, Accuracy) 
{
    PRINT_PARAM(devInfo);
    PRINT_TYPE(type);
    PRINT_PARAM(size);
    
    cv::Mat dst_gold;
    cv::absdiff(mat1, mat2, dst_gold);

    cv::Mat dst;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpuRes;

        cv::gpu::absdiff(cv::gpu::GpuMat(mat1), cv::gpu::GpuMat(mat2), gpuRes);

        gpuRes.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(Arithm, AbsdiffArray, testing::Combine(
                        testing::ValuesIn(devices()),
                        testing::Values(CV_8UC1, CV_8UC4, CV_32SC1, CV_32FC1)));

struct AbsdiffScalar : ArithmTest {};

TEST_P(AbsdiffScalar, Accuracy) 
{
    PRINT_PARAM(devInfo);
    PRINT_TYPE(type);
    PRINT_PARAM(size);

    cv::RNG& rng = cvtest::TS::ptr()->get_rng();

    cv::Scalar val(rng.uniform(0.1, 3.0), rng.uniform(0.1, 3.0));

    PRINT_PARAM(val);
    
    cv::Mat dst_gold;
    cv::absdiff(mat1, val, dst_gold);

    cv::Mat dst;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpuRes;

        cv::gpu::absdiff(cv::gpu::GpuMat(mat1), val, gpuRes);

        gpuRes.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

INSTANTIATE_TEST_CASE_P(Arithm, AbsdiffScalar, testing::Combine(
                        testing::ValuesIn(devices()),
                        testing::Values(CV_32FC1)));

////////////////////////////////////////////////////////////////////////////////
// compare

struct Compare : testing::TestWithParam< std::tr1::tuple<cv::gpu::DeviceInfo, int> > 
{
    cv::gpu::DeviceInfo devInfo;
    int cmp_code;

    cv::Size size;
    cv::Mat mat1, mat2;

    cv::Mat dst_gold;
        
    virtual void SetUp()
    {
        devInfo = std::tr1::get<0>(GetParam());
        cmp_code = std::tr1::get<1>(GetParam());

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = cvtest::TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));
        
        mat1 = cvtest::randomMat(rng, size, CV_32FC1, 1, 16, false);
        mat2 = cvtest::randomMat(rng, size, CV_32FC1, 1, 16, false);

        cv::compare(mat1, mat2, dst_gold, cmp_code);
    }
};

TEST_P(Compare, Accuracy) 
{
    static const char* cmp_codes[] = {"CMP_EQ", "CMP_GT", "CMP_GE", "CMP_LT", "CMP_LE", "CMP_NE"};
    const char* cmpCodeStr = cmp_codes[cmp_code];

    PRINT_PARAM(devInfo);
    PRINT_PARAM(size);
    PRINT_PARAM(cmpCodeStr);

    cv::Mat dst;
    
    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpuRes;

        cv::gpu::compare(cv::gpu::GpuMat(mat1), cv::gpu::GpuMat(mat2), gpuRes, cmp_code);

        gpuRes.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(Arithm, Compare, testing::Combine(
                        testing::ValuesIn(devices()),
                        testing::Values((int)cv::CMP_EQ, (int)cv::CMP_GT, (int)cv::CMP_GE, (int)cv::CMP_LT, (int)cv::CMP_LE, (int)cv::CMP_NE)));

////////////////////////////////////////////////////////////////////////////////
// meanStdDev

struct MeanStdDev : testing::TestWithParam<cv::gpu::DeviceInfo>
{
    cv::gpu::DeviceInfo devInfo;

    cv::Size size;
    cv::Mat mat;

    cv::Scalar mean_gold;
    cv::Scalar stddev_gold;

    virtual void SetUp() 
    {
        devInfo = GetParam();

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = cvtest::TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));
        
        mat = cvtest::randomMat(rng, size, CV_8UC1, 1, 255, false);

        cv::meanStdDev(mat, mean_gold, stddev_gold);
    }
};

TEST_P(MeanStdDev, Accuracy) 
{
    PRINT_PARAM(devInfo);
    PRINT_PARAM(size);

    cv::Scalar mean;
    cv::Scalar stddev;
    
    ASSERT_NO_THROW(
        cv::gpu::meanStdDev(cv::gpu::GpuMat(mat), mean, stddev);
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

INSTANTIATE_TEST_CASE_P(Arithm, MeanStdDev, testing::ValuesIn(devices()));

////////////////////////////////////////////////////////////////////////////////
// normDiff

static const int norms[] = {cv::NORM_INF, cv::NORM_L1, cv::NORM_L2};
static const char* norms_str[] = {"NORM_INF", "NORM_L1", "NORM_L2"};

struct NormDiff : testing::TestWithParam< std::tr1::tuple<cv::gpu::DeviceInfo, int> >
{
    cv::gpu::DeviceInfo devInfo;
    int normIdx;

    cv::Size size;
    cv::Mat mat1, mat2;

    double norm_gold;

    virtual void SetUp() 
    {
        devInfo = std::tr1::get<0>(GetParam());
        normIdx = std::tr1::get<1>(GetParam());

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = cvtest::TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));
        
        mat1 = cvtest::randomMat(rng, size, CV_8UC1, 1, 255, false);
        mat2 = cvtest::randomMat(rng, size, CV_8UC1, 1, 255, false);

        norm_gold = cv::norm(mat1, mat2, norms[normIdx]);
    }
};

TEST_P(NormDiff, Accuracy) 
{
    const char* normStr = norms_str[normIdx];

    PRINT_PARAM(devInfo);
    PRINT_PARAM(size);
    PRINT_PARAM(normStr);
    
    double norm;
    
    ASSERT_NO_THROW(
        norm = cv::gpu::norm(cv::gpu::GpuMat(mat1), cv::gpu::GpuMat(mat2), norms[normIdx]);
    );

    EXPECT_NEAR(norm_gold, norm, 1e-6);
}

INSTANTIATE_TEST_CASE_P(Arithm, NormDiff, testing::Combine(
                        testing::ValuesIn(devices()),
                        testing::Range(0, 3)));

////////////////////////////////////////////////////////////////////////////////
// flip

struct Flip : testing::TestWithParam< std::tr1::tuple<cv::gpu::DeviceInfo, int, int> >
{
    cv::gpu::DeviceInfo devInfo;
    int type;
    int flip_code;

    cv::Size size;
    cv::Mat mat;

    cv::Mat dst_gold;

    virtual void SetUp() 
    {
        devInfo = std::tr1::get<0>(GetParam());
        type = std::tr1::get<1>(GetParam());
        flip_code = std::tr1::get<2>(GetParam());

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = cvtest::TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));
        
        mat = cvtest::randomMat(rng, size, type, 1, 255, false);

        cv::flip(mat, dst_gold, flip_code);
    }
};

TEST_P(Flip, Accuracy) 
{
    static const char* flip_axis[] = {"Both", "X", "Y"};
    const char* flipAxisStr = flip_axis[flip_code + 1];

    PRINT_PARAM(devInfo);
    PRINT_TYPE(type);
    PRINT_PARAM(size);
    PRINT_PARAM(flipAxisStr);
    
    cv::Mat dst;
    
    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpu_res;

        cv::gpu::flip(cv::gpu::GpuMat(mat), gpu_res, flip_code);

        gpu_res.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(Arithm, Flip, testing::Combine(
                        testing::ValuesIn(devices()),
                        testing::Values(CV_8UC1, CV_8UC4),
                        testing::Values(0, 1, -1)));

////////////////////////////////////////////////////////////////////////////////
// LUT

struct LUT : testing::TestWithParam< std::tr1::tuple<cv::gpu::DeviceInfo, int> >
{
    cv::gpu::DeviceInfo devInfo;
    int type;

    cv::Size size;
    cv::Mat mat;
    cv::Mat lut;

    cv::Mat dst_gold;

    virtual void SetUp() 
    {
        devInfo = std::tr1::get<0>(GetParam());
        type = std::tr1::get<1>(GetParam());

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = cvtest::TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));
        
        mat = cvtest::randomMat(rng, size, type, 1, 255, false);
        lut = cvtest::randomMat(rng, cv::Size(256, 1), CV_8UC1, 100, 200, false);

        cv::LUT(mat, lut, dst_gold);
    }
};

TEST_P(LUT, Accuracy) 
{
    PRINT_PARAM(devInfo);
    PRINT_TYPE(type);
    PRINT_PARAM(size);

    cv::Mat dst;
    
    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpu_res;

        cv::gpu::LUT(cv::gpu::GpuMat(mat), lut, gpu_res);

        gpu_res.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(Arithm, LUT, testing::Combine(
                        testing::ValuesIn(devices()),
                        testing::Values(CV_8UC1, CV_8UC3)));

////////////////////////////////////////////////////////////////////////////////
// exp

struct Exp : testing::TestWithParam<cv::gpu::DeviceInfo>
{
    cv::gpu::DeviceInfo devInfo;

    cv::Size size;
    cv::Mat mat;

    cv::Mat dst_gold;

    virtual void SetUp() 
    {
        devInfo = GetParam();

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = cvtest::TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));

        mat = cvtest::randomMat(rng, size, CV_32FC1, -10.0, 2.0, false);        

        cv::exp(mat, dst_gold);
    }
};

TEST_P(Exp, Accuracy) 
{
    PRINT_PARAM(devInfo);
    PRINT_PARAM(size);

    cv::Mat dst;
    
    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpu_res;

        cv::gpu::exp(cv::gpu::GpuMat(mat), gpu_res);

        gpu_res.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

INSTANTIATE_TEST_CASE_P(Arithm, Exp, testing::ValuesIn(devices()));



////////////////////////////////////////////////////////////////////////////////
// pow

struct Pow : testing::TestWithParam< std::tr1::tuple<cv::gpu::DeviceInfo, int> >
{
    cv::gpu::DeviceInfo devInfo;
    int type;

    double power;
    cv::Size size;
    cv::Mat mat;

    cv::Mat dst_gold;

    virtual void SetUp() 
    {        
        devInfo = std::tr1::get<0>(GetParam());
        type = std::tr1::get<1>(GetParam());        

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = cvtest::TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));
        //size = cv::Size(2, 2);

        mat = cvtest::randomMat(rng, size, type, 0.0, 100.0, false);        

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
    PRINT_PARAM(devInfo);
    PRINT_TYPE(type);
    PRINT_PARAM(size);
    PRINT_PARAM(power);

    cv::Mat dst;
    
    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpu_res;

        cv::gpu::pow(cv::gpu::GpuMat(mat), power, gpu_res);

        gpu_res.download(dst);
    );

    /*std::cout  << mat << std::endl << std::endl;
    std::cout  << dst << std::endl << std::endl;
    std::cout  << dst_gold << std::endl;*/
    EXPECT_MAT_NEAR(dst_gold, dst, 1);
}

INSTANTIATE_TEST_CASE_P(Arithm, Pow, testing::Combine(
                        testing::ValuesIn(devices()),
                        testing::Values(CV_32F, CV_32FC3)));

////////////////////////////////////////////////////////////////////////////////
// log

struct Log : testing::TestWithParam<cv::gpu::DeviceInfo>
{
    cv::gpu::DeviceInfo devInfo;

    cv::Size size;
    cv::Mat mat;

    cv::Mat dst_gold;

    virtual void SetUp() 
    {
        devInfo = GetParam();

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = cvtest::TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));

        mat = cvtest::randomMat(rng, size, CV_32FC1, 0.0, 100.0, false);        

        cv::log(mat, dst_gold);
    }
};

TEST_P(Log, Accuracy) 
{
    PRINT_PARAM(devInfo);
    PRINT_PARAM(size);

    cv::Mat dst;
    
    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpu_res;

        cv::gpu::log(cv::gpu::GpuMat(mat), gpu_res);

        gpu_res.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

INSTANTIATE_TEST_CASE_P(Arithm, Log, testing::ValuesIn(devices()));

////////////////////////////////////////////////////////////////////////////////
// magnitude

struct Magnitude : testing::TestWithParam<cv::gpu::DeviceInfo>
{
    cv::gpu::DeviceInfo devInfo;

    cv::Size size;
    cv::Mat mat1, mat2;

    cv::Mat dst_gold;

    virtual void SetUp() 
    {
        devInfo = GetParam();

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = cvtest::TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));

        mat1 = cvtest::randomMat(rng, size, CV_32FC1, 0.0, 100.0, false);
        mat2 = cvtest::randomMat(rng, size, CV_32FC1, 0.0, 100.0, false);       

        cv::magnitude(mat1, mat2, dst_gold);
    }
};

TEST_P(Magnitude, Accuracy) 
{
    PRINT_PARAM(devInfo);
    PRINT_PARAM(size);

    cv::Mat dst;
    
    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpu_res;

        cv::gpu::magnitude(cv::gpu::GpuMat(mat1), cv::gpu::GpuMat(mat2), gpu_res);

        gpu_res.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-4);
}

INSTANTIATE_TEST_CASE_P(Arithm, Magnitude, testing::ValuesIn(devices()));

////////////////////////////////////////////////////////////////////////////////
// phase

struct Phase : testing::TestWithParam<cv::gpu::DeviceInfo>
{
    cv::gpu::DeviceInfo devInfo;

    cv::Size size;
    cv::Mat mat1, mat2;

    cv::Mat dst_gold;

    virtual void SetUp() 
    {
        devInfo = GetParam();

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = cvtest::TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));

        mat1 = cvtest::randomMat(rng, size, CV_32FC1, 0.0, 100.0, false);
        mat2 = cvtest::randomMat(rng, size, CV_32FC1, 0.0, 100.0, false);       

        cv::phase(mat1, mat2, dst_gold);
    }
};

TEST_P(Phase, Accuracy) 
{
    PRINT_PARAM(devInfo);
    PRINT_PARAM(size);

    cv::Mat dst;
    
    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpu_res;

        cv::gpu::phase(cv::gpu::GpuMat(mat1), cv::gpu::GpuMat(mat2), gpu_res);

        gpu_res.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-3);
}

INSTANTIATE_TEST_CASE_P(Arithm, Phase, testing::ValuesIn(devices()));

////////////////////////////////////////////////////////////////////////////////
// cartToPolar

struct CartToPolar : testing::TestWithParam<cv::gpu::DeviceInfo>
{
    cv::gpu::DeviceInfo devInfo;

    cv::Size size;
    cv::Mat mat1, mat2;

    cv::Mat mag_gold;
    cv::Mat angle_gold;

    virtual void SetUp() 
    {
        devInfo = GetParam();

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = cvtest::TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));

        mat1 = cvtest::randomMat(rng, size, CV_32FC1, -100.0, 100.0, false);
        mat2 = cvtest::randomMat(rng, size, CV_32FC1, -100.0, 100.0, false);       

        cv::cartToPolar(mat1, mat2, mag_gold, angle_gold);
    }
};

TEST_P(CartToPolar, Accuracy) 
{
    PRINT_PARAM(devInfo);
    PRINT_PARAM(size);

    cv::Mat mag, angle;
    
    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpuMag;
        cv::gpu::GpuMat gpuAngle;

        cv::gpu::cartToPolar(cv::gpu::GpuMat(mat1), cv::gpu::GpuMat(mat2), gpuMag, gpuAngle);

        gpuMag.download(mag);
        gpuAngle.download(angle);
    );

    EXPECT_MAT_NEAR(mag_gold, mag, 1e-4);
    EXPECT_MAT_NEAR(angle_gold, angle, 1e-3);
}

INSTANTIATE_TEST_CASE_P(Arithm, CartToPolar, testing::ValuesIn(devices()));

////////////////////////////////////////////////////////////////////////////////
// polarToCart

struct PolarToCart : testing::TestWithParam<cv::gpu::DeviceInfo>
{
    cv::gpu::DeviceInfo devInfo;

    cv::Size size;
    cv::Mat mag;
    cv::Mat angle;

    cv::Mat x_gold;
    cv::Mat y_gold;

    virtual void SetUp() 
    {
        devInfo = GetParam();

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = cvtest::TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));

        mag = cvtest::randomMat(rng, size, CV_32FC1, -100.0, 100.0, false);
        angle = cvtest::randomMat(rng, size, CV_32FC1, 0.0, 2.0 * CV_PI, false);       

        cv::polarToCart(mag, angle, x_gold, y_gold);
    }
};

TEST_P(PolarToCart, Accuracy) 
{
    PRINT_PARAM(devInfo);
    PRINT_PARAM(size);

    cv::Mat x, y;
    
    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpuX;
        cv::gpu::GpuMat gpuY;

        cv::gpu::polarToCart(cv::gpu::GpuMat(mag), cv::gpu::GpuMat(angle), gpuX, gpuY);

        gpuX.download(x);
        gpuY.download(y);
    );

    EXPECT_MAT_NEAR(x_gold, x, 1e-4);
    EXPECT_MAT_NEAR(y_gold, y, 1e-4);
}

INSTANTIATE_TEST_CASE_P(Arithm, PolarToCart, testing::ValuesIn(devices()));

////////////////////////////////////////////////////////////////////////////////
// minMax

struct MinMax : testing::TestWithParam< std::tr1::tuple<cv::gpu::DeviceInfo, int> >
{
    cv::gpu::DeviceInfo devInfo;
    int type;

    cv::Size size;
    cv::Mat mat;
    cv::Mat mask;

    double minVal_gold;
    double maxVal_gold;

    virtual void SetUp() 
    {
        devInfo = std::tr1::get<0>(GetParam());
        type = std::tr1::get<1>(GetParam());

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = cvtest::TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));

        mat = cvtest::randomMat(rng, size, type, 0.0, 127.0, false);
        mask = cvtest::randomMat(rng, size, CV_8UC1, 0, 2, false);

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
    if (type == CV_64F && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
        return;

    PRINT_PARAM(devInfo);
    PRINT_TYPE(type)
    PRINT_PARAM(size);

    double minVal, maxVal;
    
    ASSERT_NO_THROW(
        cv::gpu::minMax(cv::gpu::GpuMat(mat), &minVal, &maxVal, cv::gpu::GpuMat(mask));
    );

    EXPECT_DOUBLE_EQ(minVal_gold, minVal);
    EXPECT_DOUBLE_EQ(maxVal_gold, maxVal);
}

INSTANTIATE_TEST_CASE_P(Arithm, MinMax, testing::Combine(
                        testing::ValuesIn(devices()),
                        testing::Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F)));

////////////////////////////////////////////////////////////////////////////////
// minMaxLoc

struct MinMaxLoc : testing::TestWithParam< std::tr1::tuple<cv::gpu::DeviceInfo, int> >
{
    cv::gpu::DeviceInfo devInfo;
    int type;

    cv::Size size;
    cv::Mat mat;
    cv::Mat mask;

    double minVal_gold;
    double maxVal_gold;
    cv::Point minLoc_gold;
    cv::Point maxLoc_gold;

    virtual void SetUp() 
    {
        devInfo = std::tr1::get<0>(GetParam());
        type = std::tr1::get<1>(GetParam());

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = cvtest::TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));

        mat = cvtest::randomMat(rng, size, type, 0.0, 127.0, false);
        mask = cvtest::randomMat(rng, size, CV_8UC1, 0, 2, false);

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

    PRINT_PARAM(devInfo);
    PRINT_TYPE(type)
    PRINT_PARAM(size);

    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    
    ASSERT_NO_THROW(
        cv::gpu::minMaxLoc(cv::gpu::GpuMat(mat), &minVal, &maxVal, &minLoc, &maxLoc, cv::gpu::GpuMat(mask));
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

INSTANTIATE_TEST_CASE_P(Arithm, MinMaxLoc, testing::Combine(
                        testing::ValuesIn(devices()),
                        testing::Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F)));

////////////////////////////////////////////////////////////////////////////
// countNonZero

struct CountNonZero : testing::TestWithParam< std::tr1::tuple<cv::gpu::DeviceInfo, int> >
{
    cv::gpu::DeviceInfo devInfo;
    int type;

    cv::Size size;
    cv::Mat mat;

    int n_gold;

    virtual void SetUp() 
    {
        devInfo = std::tr1::get<0>(GetParam());
        type = std::tr1::get<1>(GetParam());

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = cvtest::TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));

        cv::Mat matBase = cvtest::randomMat(rng, size, CV_8U, 0.0, 1.0, false);
        matBase.convertTo(mat, type);

        n_gold = cv::countNonZero(mat);
    }
};

TEST_P(CountNonZero, Accuracy) 
{
    if (type == CV_64F && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
        return;

    PRINT_PARAM(devInfo);
    PRINT_TYPE(type)
    PRINT_PARAM(size);

    int n;
    
    ASSERT_NO_THROW(
        n = cv::gpu::countNonZero(cv::gpu::GpuMat(mat));
    );

    ASSERT_EQ(n_gold, n);
}

INSTANTIATE_TEST_CASE_P(Arithm, CountNonZero, testing::Combine(
                        testing::ValuesIn(devices()),
                        testing::Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F)));

//////////////////////////////////////////////////////////////////////////////
// sum

struct Sum : testing::TestWithParam< std::tr1::tuple<cv::gpu::DeviceInfo, int> >
{
    cv::gpu::DeviceInfo devInfo;
    int type;

    cv::Size size;
    cv::Mat mat;

    cv::Scalar sum_gold;

    virtual void SetUp() 
    {
        devInfo = std::tr1::get<0>(GetParam());
        type = std::tr1::get<1>(GetParam());

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = cvtest::TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));

        mat = cvtest::randomMat(rng, size, CV_8U, 0.0, 10.0, false);

        sum_gold = cv::sum(mat);
    }
};

TEST_P(Sum, Accuracy) 
{
    if (type == CV_64F && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
        return;

    PRINT_PARAM(devInfo);
    PRINT_TYPE(type)
    PRINT_PARAM(size);

    cv::Scalar sum;
    
    ASSERT_NO_THROW(
        sum = cv::gpu::sum(cv::gpu::GpuMat(mat));
    );

    EXPECT_NEAR(sum[0], sum_gold[0], mat.size().area() * 1e-5);
    EXPECT_NEAR(sum[1], sum_gold[1], mat.size().area() * 1e-5);
    EXPECT_NEAR(sum[2], sum_gold[2], mat.size().area() * 1e-5);
    EXPECT_NEAR(sum[3], sum_gold[3], mat.size().area() * 1e-5);
}

INSTANTIATE_TEST_CASE_P(Arithm, Sum, testing::Combine(
                        testing::ValuesIn(devices()),
                        testing::Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F)));

struct AbsSum : testing::TestWithParam< std::tr1::tuple<cv::gpu::DeviceInfo, int> >
{
    cv::gpu::DeviceInfo devInfo;
    int type;

    cv::Size size;
    cv::Mat mat;

    cv::Scalar sum_gold;

    virtual void SetUp() 
    {
        devInfo = std::tr1::get<0>(GetParam());
        type = std::tr1::get<1>(GetParam());

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = cvtest::TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));

        mat = cvtest::randomMat(rng, size, CV_8U, 0.0, 10.0, false);

        sum_gold = cv::norm(mat, cv::NORM_L1);
    }
};

TEST_P(AbsSum, Accuracy) 
{
    if (type == CV_64F && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
        return;

    PRINT_PARAM(devInfo);
    PRINT_TYPE(type)
    PRINT_PARAM(size);

    cv::Scalar sum;
    
    ASSERT_NO_THROW(
        sum = cv::gpu::absSum(cv::gpu::GpuMat(mat));
    );

    EXPECT_NEAR(sum[0], sum_gold[0], mat.size().area() * 1e-5);
    EXPECT_NEAR(sum[1], sum_gold[1], mat.size().area() * 1e-5);
    EXPECT_NEAR(sum[2], sum_gold[2], mat.size().area() * 1e-5);
    EXPECT_NEAR(sum[3], sum_gold[3], mat.size().area() * 1e-5);
}

INSTANTIATE_TEST_CASE_P(Arithm, AbsSum, testing::Combine(
                        testing::ValuesIn(devices()),
                        testing::Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F)));

struct SqrSum : testing::TestWithParam< std::tr1::tuple<cv::gpu::DeviceInfo, int> >
{
    cv::gpu::DeviceInfo devInfo;
    int type;

    cv::Size size;
    cv::Mat mat;

    cv::Scalar sum_gold;

    virtual void SetUp() 
    {
        devInfo = std::tr1::get<0>(GetParam());
        type = std::tr1::get<1>(GetParam());

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = cvtest::TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));

        mat = cvtest::randomMat(rng, size, CV_8U, 0.0, 10.0, false);
 
        cv::Mat sqrmat;
        cv::multiply(mat, mat, sqrmat);
        sum_gold = cv::sum(sqrmat);
    }
};

TEST_P(SqrSum, Accuracy) 
{
    if (type == CV_64F && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
        return;

    PRINT_PARAM(devInfo);
    PRINT_TYPE(type)
    PRINT_PARAM(size);

    cv::Scalar sum;
    
    ASSERT_NO_THROW(
        sum = cv::gpu::sqrSum(cv::gpu::GpuMat(mat));
    );

    EXPECT_NEAR(sum[0], sum_gold[0], mat.size().area() * 1e-5);
    EXPECT_NEAR(sum[1], sum_gold[1], mat.size().area() * 1e-5);
    EXPECT_NEAR(sum[2], sum_gold[2], mat.size().area() * 1e-5);
    EXPECT_NEAR(sum[3], sum_gold[3], mat.size().area() * 1e-5);
}

INSTANTIATE_TEST_CASE_P(Arithm, SqrSum, testing::Combine(
                        testing::ValuesIn(devices()),
                        testing::Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F)));

//////////////////////////////////////////////////////////////////////////////
// bitwise

struct BitwiseNot : testing::TestWithParam< std::tr1::tuple<cv::gpu::DeviceInfo, int> >
{
    cv::gpu::DeviceInfo devInfo;
    int type;

    cv::Size size;
    cv::Mat mat;

    cv::Mat dst_gold;

    virtual void SetUp() 
    {
        devInfo = std::tr1::get<0>(GetParam());
        type = std::tr1::get<1>(GetParam());

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = cvtest::TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));

        mat.create(size, type);
        
        for (int i = 0; i < mat.rows; ++i)
        {
            cv::Mat row(1, static_cast<int>(mat.cols * mat.elemSize()), CV_8U, (void*)mat.ptr(i));
            rng.fill(row, cv::RNG::UNIFORM, cv::Scalar(0), cv::Scalar(255));
        }

        dst_gold = ~mat;
    }
};

TEST_P(BitwiseNot, Accuracy) 
{
    if (mat.depth() == CV_64F && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
        return;

    PRINT_PARAM(devInfo);
    PRINT_TYPE(type)
    PRINT_PARAM(size);

    cv::Mat dst;
    
    ASSERT_NO_THROW(
        cv::gpu::GpuMat dev_dst;

        cv::gpu::bitwise_not(cv::gpu::GpuMat(mat), dev_dst);

        dev_dst.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(Arithm, BitwiseNot, testing::Combine(
                        testing::ValuesIn(devices()),
                        testing::ValuesIn(all_types())));

struct BitwiseOr : testing::TestWithParam< std::tr1::tuple<cv::gpu::DeviceInfo, int> >
{
    cv::gpu::DeviceInfo devInfo;
    int type;

    cv::Size size;
    cv::Mat mat1;
    cv::Mat mat2;

    cv::Mat dst_gold;

    virtual void SetUp() 
    {
        devInfo = std::tr1::get<0>(GetParam());
        type = std::tr1::get<1>(GetParam());

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

        dst_gold = mat1 | mat2;
    }
};

TEST_P(BitwiseOr, Accuracy) 
{
    if (mat1.depth() == CV_64F && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
        return;

    PRINT_PARAM(devInfo);
    PRINT_TYPE(type)
    PRINT_PARAM(size);

    cv::Mat dst;
    
    ASSERT_NO_THROW(
        cv::gpu::GpuMat dev_dst;

        cv::gpu::bitwise_or(cv::gpu::GpuMat(mat1), cv::gpu::GpuMat(mat2), dev_dst);

        dev_dst.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(Arithm, BitwiseOr, testing::Combine(
                        testing::ValuesIn(devices()),
                        testing::ValuesIn(all_types())));

struct BitwiseAnd : testing::TestWithParam< std::tr1::tuple<cv::gpu::DeviceInfo, int> >
{
    cv::gpu::DeviceInfo devInfo;
    int type;

    cv::Size size;
    cv::Mat mat1;
    cv::Mat mat2;

    cv::Mat dst_gold;

    virtual void SetUp() 
    {
        devInfo = std::tr1::get<0>(GetParam());
        type = std::tr1::get<1>(GetParam());

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

        dst_gold = mat1 & mat2;
    }
};

TEST_P(BitwiseAnd, Accuracy) 
{
    if (mat1.depth() == CV_64F && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
        return;

    PRINT_PARAM(devInfo);
    PRINT_TYPE(type)
    PRINT_PARAM(size);

    cv::Mat dst;
    
    ASSERT_NO_THROW(
        cv::gpu::GpuMat dev_dst;

        cv::gpu::bitwise_and(cv::gpu::GpuMat(mat1), cv::gpu::GpuMat(mat2), dev_dst);

        dev_dst.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(Arithm, BitwiseAnd, testing::Combine(
                        testing::ValuesIn(devices()),
                        testing::ValuesIn(all_types())));

struct BitwiseXor : testing::TestWithParam< std::tr1::tuple<cv::gpu::DeviceInfo, int> >
{
    cv::gpu::DeviceInfo devInfo;
    int type;

    cv::Size size;
    cv::Mat mat1;
    cv::Mat mat2;

    cv::Mat dst_gold;

    virtual void SetUp() 
    {
        devInfo = std::tr1::get<0>(GetParam());
        type = std::tr1::get<1>(GetParam());

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

        dst_gold = mat1 ^ mat2;
    }
};

TEST_P(BitwiseXor, Accuracy) 
{
    if (mat1.depth() == CV_64F && !supportFeature(devInfo, cv::gpu::NATIVE_DOUBLE))
        return;

    PRINT_PARAM(devInfo);
    PRINT_TYPE(type)
    PRINT_PARAM(size);

    cv::Mat dst;
    
    ASSERT_NO_THROW(
        cv::gpu::GpuMat dev_dst;

        cv::gpu::bitwise_xor(cv::gpu::GpuMat(mat1), cv::gpu::GpuMat(mat2), dev_dst);

        dev_dst.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(Arithm, BitwiseXor, testing::Combine(
                        testing::ValuesIn(devices()),
                        testing::ValuesIn(all_types())));

#endif // HAVE_CUDA
