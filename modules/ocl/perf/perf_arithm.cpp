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
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Fangfang Bai, fangfang@multicorewareinc.com
//    Jin Ma,       jin@multicorewareinc.com
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other oclMaterials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors as is and
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

#include "precomp.hpp"
///////////// Lut ////////////////////////
PERFTEST(lut)
{
    Mat src, lut, dst;
    ocl::oclMat d_src, d_lut, d_dst;

    int all_type[] = {CV_8UC1, CV_8UC3};
    std::string type_name[] = {"CV_8UC1", "CV_8UC3"};

    for (int size = Min_Size; size <= Max_Size; size *= Multiple)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j];

            gen(src, size, size, all_type[j], 0, 256);
            gen(lut, 1, 256, CV_8UC1, 0, 1);
            dst = src;

            LUT(src, lut, dst);

            CPU_ON;
            LUT(src, lut, dst);
            CPU_OFF;

            d_src.upload(src);
            d_lut.upload(lut);

            WARMUP_ON;
            ocl::LUT(d_src, d_lut, d_dst);
            WARMUP_OFF;

            cv::Mat ocl_mat_dst;
            d_dst.download(ocl_mat_dst);

            TestSystem::instance().setAccurate(ExpectedMatNear(ocl_mat_dst, dst, 0));

            GPU_ON;
            ocl::LUT(d_src, d_lut, d_dst);
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            d_lut.upload(lut);
            ocl::LUT(d_src, d_lut, d_dst);
            d_dst.download(dst);
            GPU_FULL_OFF;

        }

    }
}

///////////// Exp ////////////////////////
PERFTEST(Exp)
{
    Mat src, dst;
    ocl::oclMat d_src, d_dst;

    for (int size = Min_Size; size <= Max_Size; size *= Multiple)
    {
        SUBTEST << size << 'x' << size << "; CV_32FC1";

        gen(src, size, size, CV_32FC1, 5, 16);

        exp(src, dst);

        CPU_ON;
        exp(src, dst);
        CPU_OFF;
        d_src.upload(src);

        WARMUP_ON;
        ocl::exp(d_src, d_dst);
        WARMUP_OFF;

        cv::Mat ocl_mat_dst;
        d_dst.download(ocl_mat_dst);

        TestSystem::instance().setAccurate(ExpectedMatNear(ocl_mat_dst, dst, 2));

        GPU_ON;
        ocl::exp(d_src, d_dst);
        GPU_OFF;

        GPU_FULL_ON;
        d_src.upload(src);
        ocl::exp(d_src, d_dst);
        d_dst.download(dst);
        GPU_FULL_OFF;
    }
}

///////////// LOG ////////////////////////
PERFTEST(Log)
{
    Mat src, dst;
    ocl::oclMat d_src, d_dst;

    for (int size = Min_Size; size <= Max_Size; size *= Multiple)
    {
        SUBTEST << size << 'x' << size << "; 32F";

        gen(src, size, size, CV_32F, 1, 10);

        log(src, dst);

        CPU_ON;
        log(src, dst);
        CPU_OFF;
        d_src.upload(src);

        WARMUP_ON;
        ocl::log(d_src, d_dst);
        WARMUP_OFF;

        cv::Mat ocl_mat_dst;
        d_dst.download(ocl_mat_dst);

        TestSystem::instance().setAccurate(ExpectedMatNear(ocl_mat_dst, dst, 1));

        GPU_ON;
        ocl::log(d_src, d_dst);
        GPU_OFF;

        GPU_FULL_ON;
        d_src.upload(src);
        ocl::log(d_src, d_dst);
        d_dst.download(dst);
        GPU_FULL_OFF;
    }
}

///////////// Add ////////////////////////
PERFTEST(Add)
{
    Mat src1, src2, dst;
    ocl::oclMat d_src1, d_src2, d_dst;

    int all_type[] = {CV_8UC1, CV_32FC1};
    std::string type_name[] = {"CV_8UC1", "CV_32FC1"};

    for (int size = Min_Size; size <= Max_Size; size *= Multiple)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j];

            gen(src1, size, size, all_type[j], 0, 1);
            gen(src2, size, size, all_type[j], 0, 1);

            add(src1, src2, dst);

            CPU_ON;
            add(src1, src2, dst);
            CPU_OFF;
            d_src1.upload(src1);
            d_src2.upload(src2);

            WARMUP_ON;
            ocl::add(d_src1, d_src2, d_dst);
            WARMUP_OFF;

            cv::Mat ocl_mat_dst;
            d_dst.download(ocl_mat_dst);

            TestSystem::instance().setAccurate(ExpectedMatNear(ocl_mat_dst, dst, 0.0));

            GPU_ON;
            ocl::add(d_src1, d_src2, d_dst);
            GPU_OFF;

            GPU_FULL_ON;
            d_src1.upload(src1);
            d_src2.upload(src2);
            ocl::add(d_src1, d_src2, d_dst);
            d_dst.download(dst);
            GPU_FULL_OFF;
        }

    }
}

///////////// Mul ////////////////////////
PERFTEST(Mul)
{
    Mat src1, src2, dst;
    ocl::oclMat d_src1, d_src2, d_dst;

    int all_type[] = {CV_8UC1, CV_8UC4};
    std::string type_name[] = {"CV_8UC1", "CV_8UC4"};

    for (int size = Min_Size; size <= Max_Size; size *= Multiple)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] ;

            gen(src1, size, size, all_type[j], 0, 256);
            gen(src2, size, size, all_type[j], 0, 256);
            dst = src1;
            dst.setTo(0);

            multiply(src1, src2, dst);

            CPU_ON;
            multiply(src1, src2, dst);
            CPU_OFF;
            d_src1.upload(src1);
            d_src2.upload(src2);

            WARMUP_ON;
            ocl::multiply(d_src1, d_src2, d_dst);
            WARMUP_OFF;

            cv::Mat ocl_mat_dst;
            d_dst.download(ocl_mat_dst);

            TestSystem::instance().setAccurate(ExpectedMatNear(ocl_mat_dst, dst, 0.0));

            GPU_ON;
            ocl::multiply(d_src1, d_src2, d_dst);
            GPU_OFF;

            GPU_FULL_ON;
            d_src1.upload(src1);
            d_src2.upload(src2);
            ocl::multiply(d_src1, d_src2, d_dst);
            d_dst.download(dst);
            GPU_FULL_OFF;
        }

    }
}

///////////// Div ////////////////////////
PERFTEST(Div)
{
    Mat src1, src2, dst;
    ocl::oclMat d_src1, d_src2, d_dst;
    int all_type[] = {CV_8UC1, CV_8UC4};
    std::string type_name[] = {"CV_8UC1", "CV_8UC4"};

    for (int size = Min_Size; size <= Max_Size; size *= Multiple)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j];

            gen(src1, size, size, all_type[j], 0, 256);
            gen(src2, size, size, all_type[j], 0, 256);
            dst = src1;
            dst.setTo(0);

            divide(src1, src2, dst);

            CPU_ON;
            divide(src1, src2, dst);
            CPU_OFF;
            d_src1.upload(src1);
            d_src2.upload(src2);

            WARMUP_ON;
            ocl::divide(d_src1, d_src2, d_dst);
            WARMUP_OFF;

            cv::Mat ocl_mat_dst;
            d_dst.download(ocl_mat_dst);

            TestSystem::instance().setAccurate(ExpectedMatNear(ocl_mat_dst, dst, 1));

            GPU_ON;
            ocl::divide(d_src1, d_src2, d_dst);
            GPU_OFF;

            GPU_FULL_ON;
            d_src1.upload(src1);
            d_src2.upload(src2);
            ocl::divide(d_src1, d_src2, d_dst);
            d_dst.download(dst);
            GPU_FULL_OFF;
        }

    }
}

///////////// Absdiff ////////////////////////
PERFTEST(Absdiff)
{
    Mat src1, src2, dst;
    ocl::oclMat d_src1, d_src2, d_dst;

    int all_type[] = {CV_8UC1, CV_8UC4};
    std::string type_name[] = {"CV_8UC1", "CV_8UC4"};

    for (int size = Min_Size; size <= Max_Size; size *= Multiple)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] ;

            gen(src1, size, size, all_type[j], 0, 256);
            gen(src2, size, size, all_type[j], 0, 256);
            gen(dst, size, size, all_type[j], 0, 256);

            absdiff(src1, src2, dst);

            CPU_ON;
            absdiff(src1, src2, dst);
            CPU_OFF;
            d_src1.upload(src1);
            d_src2.upload(src2);

            WARMUP_ON;
            ocl::absdiff(d_src1, d_src2, d_dst);
            WARMUP_OFF;

            cv::Mat ocl_mat_dst;
            d_dst.download(ocl_mat_dst);

            TestSystem::instance().setAccurate(ExpectedMatNear(ocl_mat_dst, dst, 0.0));

            GPU_ON;
            ocl::absdiff(d_src1, d_src2, d_dst);
            GPU_OFF;

            GPU_FULL_ON;
            d_src1.upload(src1);
            d_src2.upload(src2);
            ocl::absdiff(d_src1, d_src2, d_dst);
            d_dst.download(dst);
            GPU_FULL_OFF;
        }

    }
}

///////////// CartToPolar ////////////////////////
PERFTEST(CartToPolar)
{
    Mat src1, src2, dst, dst1;
    ocl::oclMat d_src1, d_src2, d_dst, d_dst1;

    int all_type[] = {CV_32FC1};
    std::string type_name[] = {"CV_32FC1"};

    for (int size = Min_Size; size <= Max_Size; size *= Multiple)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j];

            gen(src1, size, size, all_type[j], 0, 256);
            gen(src2, size, size, all_type[j], 0, 256);
            gen(dst, size, size, all_type[j], 0, 256);
            gen(dst1, size, size, all_type[j], 0, 256);


            cartToPolar(src1, src2, dst, dst1, 1);

            CPU_ON;
            cartToPolar(src1, src2, dst, dst1, 1);
            CPU_OFF;
            d_src1.upload(src1);
            d_src2.upload(src2);

            WARMUP_ON;
            ocl::cartToPolar(d_src1, d_src2, d_dst, d_dst1, 1);
            WARMUP_OFF;

            cv::Mat ocl_mat_dst;
            d_dst.download(ocl_mat_dst);

            cv::Mat ocl_mat_dst1;
            d_dst1.download(ocl_mat_dst1);

            TestSystem::instance().setAccurate(ExpectedMatNear(ocl_mat_dst1, dst1, 0.5)&&ExpectedMatNear(ocl_mat_dst, dst, 0.5));

            GPU_ON;
            ocl::cartToPolar(d_src1, d_src2, d_dst, d_dst1, 1);
            GPU_OFF;

            GPU_FULL_ON;
            d_src1.upload(src1);
            d_src2.upload(src2);
            ocl::cartToPolar(d_src1, d_src2, d_dst, d_dst1, 1);
            d_dst.download(dst);
            d_dst1.download(dst1);
            GPU_FULL_OFF;
        }

    }
}

///////////// PolarToCart ////////////////////////
PERFTEST(PolarToCart)
{
    Mat src1, src2, dst, dst1;
    ocl::oclMat d_src1, d_src2, d_dst, d_dst1;

    int all_type[] = {CV_32FC1};
    std::string type_name[] = {"CV_32FC1"};

    for (int size = Min_Size; size <= Max_Size; size *= Multiple)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] ;

            gen(src1, size, size, all_type[j], 0, 256);
            gen(src2, size, size, all_type[j], 0, 256);
            gen(dst, size, size, all_type[j], 0, 256);
            gen(dst1, size, size, all_type[j], 0, 256);


            polarToCart(src1, src2, dst, dst1, 1);

            CPU_ON;
            polarToCart(src1, src2, dst, dst1, 1);
            CPU_OFF;
            d_src1.upload(src1);
            d_src2.upload(src2);

            WARMUP_ON;
            ocl::polarToCart(d_src1, d_src2, d_dst, d_dst1, 1);
            WARMUP_OFF;

            cv::Mat ocl_mat_dst;
            d_dst.download(ocl_mat_dst);

            cv::Mat ocl_mat_dst1;
            d_dst1.download(ocl_mat_dst1);

            TestSystem::instance().setAccurate(ExpectedMatNear(ocl_mat_dst1, dst1, 0.5)&&ExpectedMatNear(ocl_mat_dst, dst, 0.5));

            GPU_ON;
            ocl::polarToCart(d_src1, d_src2, d_dst, d_dst1, 1);
            GPU_OFF;

            GPU_FULL_ON;
            d_src1.upload(src1);
            d_src2.upload(src2);
            ocl::polarToCart(d_src1, d_src2, d_dst, d_dst1, 1);
            d_dst.download(dst);
            d_dst1.download(dst1);
            GPU_FULL_OFF;
        }

    }
}

///////////// Magnitude ////////////////////////
PERFTEST(magnitude)
{
    Mat x, y, mag;
    ocl::oclMat d_x, d_y, d_mag;

    int all_type[] = {CV_32FC1};
    std::string type_name[] = {"CV_32FC1"};

    for (int size = Min_Size; size <= Max_Size; size *= Multiple)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j];

            gen(x, size, size, all_type[j], 0, 1);
            gen(y, size, size, all_type[j], 0, 1);

            magnitude(x, y, mag);

            CPU_ON;
            magnitude(x, y, mag);
            CPU_OFF;
            d_x.upload(x);
            d_y.upload(y);

            WARMUP_ON;
            ocl::magnitude(d_x, d_y, d_mag);
            WARMUP_OFF;

            cv::Mat ocl_mat_dst;
            d_mag.download(ocl_mat_dst);

            TestSystem::instance().setAccurate(ExpectedMatNear(ocl_mat_dst, mag, 1e-5));

            GPU_ON;
            ocl::magnitude(d_x, d_y, d_mag);
            GPU_OFF;

            GPU_FULL_ON;
            d_x.upload(x);
            d_y.upload(y);
            ocl::magnitude(d_x, d_y, d_mag);
            d_mag.download(mag);
            GPU_FULL_OFF;
        }

    }
}

///////////// Transpose ////////////////////////
PERFTEST(Transpose)
{
    Mat src, dst;
    ocl::oclMat d_src, d_dst;

    int all_type[] = {CV_8UC1, CV_8UC4};
    std::string type_name[] = {"CV_8UC1", "CV_8UC4"};

    for (int size = Min_Size; size <= Max_Size; size *= Multiple)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j];

            gen(src, size, size, all_type[j], 0, 256);
            gen(dst, size, size, all_type[j], 0, 256);

            transpose(src, dst);

            CPU_ON;
            transpose(src, dst);
            CPU_OFF;
            d_src.upload(src);

            WARMUP_ON;
            ocl::transpose(d_src, d_dst);
            WARMUP_OFF;

            cv::Mat ocl_mat_dst;
            d_dst.download(ocl_mat_dst);

            TestSystem::instance().setAccurate(ExpectedMatNear(ocl_mat_dst, dst, 1e-5));

            GPU_ON;
            ocl::transpose(d_src, d_dst);
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            ocl::transpose(d_src, d_dst);
            d_dst.download(dst);
            GPU_FULL_OFF;
        }

    }
}

///////////// Flip ////////////////////////
PERFTEST(Flip)
{
    Mat src, dst;
    ocl::oclMat d_src, d_dst;

    int all_type[] = {CV_8UC1, CV_8UC4};
    std::string type_name[] = {"CV_8UC1", "CV_8UC4"};

    for (int size = Min_Size; size <= Max_Size; size *= Multiple)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] << " ; FLIP_BOTH";

            gen(src, size, size, all_type[j], 0, 256);
            gen(dst, size, size, all_type[j], 0, 256);

            flip(src, dst, 0);

            CPU_ON;
            flip(src, dst, 0);
            CPU_OFF;
            d_src.upload(src);

            WARMUP_ON;
            ocl::flip(d_src, d_dst, 0);
            WARMUP_OFF;

            cv::Mat ocl_mat_dst;
            d_dst.download(ocl_mat_dst);

            TestSystem::instance().setAccurate(ExpectedMatNear(ocl_mat_dst, dst, 1e-5));

            GPU_ON;
            ocl::flip(d_src, d_dst, 0);
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            ocl::flip(d_src, d_dst, 0);
            d_dst.download(dst);
            GPU_FULL_OFF;
        }

    }
}

///////////// minMax ////////////////////////
PERFTEST(minMax)
{
    Mat src;
    ocl::oclMat d_src;

    double min_val = 0.0, max_val = 0.0;
    double min_val_ = 0.0, max_val_ = 0.0;
    Point min_loc, max_loc;
    int all_type[] = {CV_8UC1, CV_32FC1};
    std::string type_name[] = {"CV_8UC1", "CV_32FC1"};

    for (int size = Min_Size; size <= Max_Size; size *= Multiple)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j];

            gen(src, size, size, all_type[j], 0, 256);

            CPU_ON;
            minMaxLoc(src, &min_val, &max_val, &min_loc, &max_loc);
            CPU_OFF;
            d_src.upload(src);

            WARMUP_ON;
            ocl::minMax(d_src, &min_val_, &max_val_);
            WARMUP_OFF;

            TestSystem::instance().setAccurate(EeceptDoubleEQ<double>(max_val_, max_val)&&EeceptDoubleEQ<double>(min_val_, min_val));

            GPU_ON;
            ocl::minMax(d_src, &min_val, &max_val);
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            ocl::minMax(d_src, &min_val, &max_val);
            GPU_FULL_OFF;

        }

    }
}

///////////// minMaxLoc ////////////////////////
PERFTEST(minMaxLoc)
{
    Mat src;
    ocl::oclMat d_src;

    double min_val = 0.0, max_val = 0.0;
    double min_val_ = 0.0, max_val_ = 0.0;
    Point min_loc, max_loc;
    Point min_loc_, max_loc_;
    int all_type[] = {CV_8UC1, CV_32FC1};
    std::string type_name[] = {"CV_8UC1", "CV_32FC1"};

    for (int size = Min_Size; size <= Max_Size; size *= Multiple)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] ;

            gen(src, size, size, all_type[j], 0, 1);

            CPU_ON;
            minMaxLoc(src, &min_val, &max_val, &min_loc, &max_loc);
            CPU_OFF;
            d_src.upload(src);

            WARMUP_ON;
            ocl::minMaxLoc(d_src, &min_val_, &max_val_, &min_loc_, &max_loc_);
            WARMUP_OFF;

            double error0 = 0., error1 = 0., minlocVal = 0., minlocVal_ = 0., maxlocVal = 0., maxlocVal_ = 0.;
            if(src.depth() == 0)
            {
                minlocVal = src.at<unsigned char>(min_loc);
                minlocVal_ = src.at<unsigned char>(min_loc_);
                maxlocVal = src.at<unsigned char>(max_loc);
                maxlocVal_ = src.at<unsigned char>(max_loc_);
                error0 = ::abs(src.at<unsigned char>(min_loc_) - src.at<unsigned char>(min_loc));
                error1 = ::abs(src.at<unsigned char>(max_loc_) - src.at<unsigned char>(max_loc));
            }
            if(src.depth() == 1)
            {
                minlocVal = src.at<signed char>(min_loc);
                minlocVal_ = src.at<signed char>(min_loc_);
                maxlocVal = src.at<signed char>(max_loc);
                maxlocVal_ = src.at<signed char>(max_loc_);
                error0 = ::abs(src.at<signed char>(min_loc_) - src.at<signed char>(min_loc));
                error1 = ::abs(src.at<signed char>(max_loc_) - src.at<signed char>(max_loc));
            }
            if(src.depth() == 2)
            {
                minlocVal = src.at<unsigned short>(min_loc);
                minlocVal_ = src.at<unsigned short>(min_loc_);
                maxlocVal = src.at<unsigned short>(max_loc);
                maxlocVal_ = src.at<unsigned short>(max_loc_);
                error0 = ::abs(src.at<unsigned short>(min_loc_) - src.at<unsigned short>(min_loc));
                error1 = ::abs(src.at<unsigned short>(max_loc_) - src.at<unsigned short>(max_loc));
            }
            if(src.depth() == 3)
            {
                minlocVal = src.at<signed short>(min_loc);
                minlocVal_ = src.at<signed short>(min_loc_);
                maxlocVal = src.at<signed short>(max_loc);
                maxlocVal_ = src.at<signed short>(max_loc_);
                error0 = ::abs(src.at<signed short>(min_loc_) - src.at<signed short>(min_loc));
                error1 = ::abs(src.at<signed short>(max_loc_) - src.at<signed short>(max_loc));
            }
            if(src.depth() == 4)
            {
                minlocVal = src.at<int>(min_loc);
                minlocVal_ = src.at<int>(min_loc_);
                maxlocVal = src.at<int>(max_loc);
                maxlocVal_ = src.at<int>(max_loc_);
                error0 = ::abs(src.at<int>(min_loc_) - src.at<int>(min_loc));
                error1 = ::abs(src.at<int>(max_loc_) - src.at<int>(max_loc));
            }
            if(src.depth() == 5)
            {
                minlocVal = src.at<float>(min_loc);
                minlocVal_ = src.at<float>(min_loc_);
                maxlocVal = src.at<float>(max_loc);
                maxlocVal_ = src.at<float>(max_loc_);
                error0 = ::abs(src.at<float>(min_loc_) - src.at<float>(min_loc));
                error1 = ::abs(src.at<float>(max_loc_) - src.at<float>(max_loc));
            }
            if(src.depth() == 6)
            {
                minlocVal = src.at<double>(min_loc);
                minlocVal_ = src.at<double>(min_loc_);
                maxlocVal = src.at<double>(max_loc);
                maxlocVal_ = src.at<double>(max_loc_);
                error0 = ::abs(src.at<double>(min_loc_) - src.at<double>(min_loc));
                error1 = ::abs(src.at<double>(max_loc_) - src.at<double>(max_loc));
            }

            TestSystem::instance().setAccurate(EeceptDoubleEQ<double>(error1, 0.0)
                &&EeceptDoubleEQ<double>(error0, 0.0)
                &&EeceptDoubleEQ<double>(maxlocVal_, maxlocVal)
                &&EeceptDoubleEQ<double>(minlocVal_, minlocVal)
                &&EeceptDoubleEQ<double>(max_val_, max_val)
                &&EeceptDoubleEQ<double>(min_val_, min_val));

            GPU_ON;
            ocl::minMaxLoc(d_src, &min_val, &max_val, &min_loc, &max_loc);
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            ocl::minMaxLoc(d_src, &min_val, &max_val, &min_loc, &max_loc);
            GPU_FULL_OFF;
        }

    }
}

///////////// Sum ////////////////////////
PERFTEST(Sum)
{
    Mat src;
    Scalar cpures, gpures;
    ocl::oclMat d_src;

    int all_type[] = {CV_8UC1, CV_32SC1};
    std::string type_name[] = {"CV_8UC1", "CV_32SC1"};

    for (int size = Min_Size; size <= Max_Size; size *= Multiple)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] ;

            gen(src, size, size, all_type[j], 0, 256);

            cpures = sum(src);

            CPU_ON;
            cpures = sum(src);
            CPU_OFF;
            d_src.upload(src);

            WARMUP_ON;
            gpures = ocl::sum(d_src);
            WARMUP_OFF;

            TestSystem::instance().setAccurate(ExceptDoubleNear(cpures[3], gpures[3], 0.1)
                &&ExceptDoubleNear(cpures[2], gpures[2], 0.1)
                &&ExceptDoubleNear(cpures[1], gpures[1], 0.1)
                &&ExceptDoubleNear(cpures[0], gpures[0], 0.1));


            GPU_ON;
            gpures = ocl::sum(d_src);
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            gpures = ocl::sum(d_src);
            GPU_FULL_OFF;
        }

    }
}

///////////// countNonZero ////////////////////////
PERFTEST(countNonZero)
{
    Mat src;
    ocl::oclMat d_src;

    int all_type[] = {CV_8UC1, CV_32FC1};
    std::string type_name[] = {"CV_8UC1", "CV_32FC1"};

    for (int size = Min_Size; size <= Max_Size; size *= Multiple)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] ;

            gen(src, size, size, all_type[j], 0, 256);

            countNonZero(src);

            int cpures = 0, gpures = 0;
            CPU_ON;
            cpures = countNonZero(src);
            CPU_OFF;
            d_src.upload(src);

            WARMUP_ON;
            gpures = ocl::countNonZero(d_src);
            WARMUP_OFF;

            TestSystem::instance().setAccurate((EeceptDoubleEQ<double>((double)cpures, (double)gpures)));

            GPU_ON;
            ocl::countNonZero(d_src);
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            ocl::countNonZero(d_src);
            GPU_FULL_OFF;
        }

    }
}

///////////// Phase ////////////////////////
PERFTEST(Phase)
{
    Mat src1, src2, dst;
    ocl::oclMat d_src1, d_src2, d_dst;

    int all_type[] = {CV_32FC1};
    std::string type_name[] = {"CV_32FC1"};

    for (int size = Min_Size; size <= Max_Size; size *= Multiple)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] ;

            gen(src1, size, size, all_type[j], 0, 256);
            gen(src2, size, size, all_type[j], 0, 256);
            gen(dst, size, size, all_type[j], 0, 256);


            phase(src1, src2, dst, 1);

            CPU_ON;
            phase(src1, src2, dst, 1);
            CPU_OFF;
            d_src1.upload(src1);
            d_src2.upload(src2);

            WARMUP_ON;
            ocl::phase(d_src1, d_src2, d_dst, 1);
            WARMUP_OFF;

            cv::Mat ocl_mat_dst;
            d_dst.download(ocl_mat_dst);

            TestSystem::instance().setAccurate(ExpectedMatNear(ocl_mat_dst, dst, 1e-2));

            GPU_ON;
            ocl::phase(d_src1, d_src2, d_dst, 1);
            GPU_OFF;

            GPU_FULL_ON;
            d_src1.upload(src1);
            d_src2.upload(src2);
            ocl::phase(d_src1, d_src2, d_dst, 1);
            d_dst.download(dst);
            GPU_FULL_OFF;
        }

    }
}

///////////// bitwise_and////////////////////////
PERFTEST(bitwise_and)
{
    Mat src1, src2, dst;
    ocl::oclMat d_src1, d_src2, d_dst;

    int all_type[] = {CV_8UC1, CV_32SC1};
    std::string type_name[] = {"CV_8UC1", "CV_32SC1"};

    for (int size = Min_Size; size <= Max_Size; size *= Multiple)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] ;

            gen(src1, size, size, all_type[j], 0, 256);
            gen(src2, size, size, all_type[j], 0, 256);
            gen(dst, size, size, all_type[j], 0, 256);


            bitwise_and(src1, src2, dst);

            CPU_ON;
            bitwise_and(src1, src2, dst);
            CPU_OFF;
            d_src1.upload(src1);
            d_src2.upload(src2);

            WARMUP_ON;
            ocl::bitwise_and(d_src1, d_src2, d_dst);
            WARMUP_OFF;

            cv::Mat ocl_mat_dst;
            d_dst.download(ocl_mat_dst);

            TestSystem::instance().setAccurate(ExpectedMatNear(ocl_mat_dst, dst, 0.0));

            GPU_ON;
            ocl::bitwise_and(d_src1, d_src2, d_dst);
            GPU_OFF;

            GPU_FULL_ON;
            d_src1.upload(src1);
            d_src2.upload(src2);
            ocl::bitwise_and(d_src1, d_src2, d_dst);
            d_dst.download(dst);
            GPU_FULL_OFF;
        }

    }
}

///////////// bitwise_not////////////////////////
PERFTEST(bitwise_not)
{
    Mat src1, dst;
    ocl::oclMat d_src1, d_dst;

    int all_type[] = {CV_8UC1, CV_32SC1};
    std::string type_name[] = {"CV_8UC1", "CV_32SC1"};

    for (int size = Min_Size; size <= Max_Size; size *= Multiple)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] ;

            gen(src1, size, size, all_type[j], 0, 256);
            gen(dst, size, size, all_type[j], 0, 256);


            bitwise_not(src1, dst);

            CPU_ON;
            bitwise_not(src1, dst);
            CPU_OFF;
            d_src1.upload(src1);

            WARMUP_ON;
            ocl::bitwise_not(d_src1, d_dst);
            WARMUP_OFF;

            cv::Mat ocl_mat_dst;
            d_dst.download(ocl_mat_dst);

            TestSystem::instance().setAccurate(ExpectedMatNear(ocl_mat_dst, dst, 0.0));

            GPU_ON;
            ocl::bitwise_not(d_src1, d_dst);
            GPU_OFF;

            GPU_FULL_ON;
            d_src1.upload(src1);
            ocl::bitwise_not(d_src1, d_dst);
            d_dst.download(dst);
            GPU_FULL_OFF;
        }

    }
}

///////////// compare////////////////////////
PERFTEST(compare)
{
    Mat src1, src2, dst;
    ocl::oclMat d_src1, d_src2, d_dst;

    int CMP_EQ = 0;
    int all_type[] = {CV_8UC1, CV_32FC1};
    std::string type_name[] = {"CV_8UC1", "CV_32FC1"};

    for (int size = Min_Size; size <= Max_Size; size *= Multiple)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] ;

            gen(src1, size, size, all_type[j], 0, 256);
            gen(src2, size, size, all_type[j], 0, 256);
            gen(dst, size, size, all_type[j], 0, 256);


            compare(src1, src2, dst, CMP_EQ);

            CPU_ON;
            compare(src1, src2, dst, CMP_EQ);
            CPU_OFF;
            d_src1.upload(src1);
            d_src2.upload(src2);

            WARMUP_ON;
            ocl::compare(d_src1, d_src2, d_dst, CMP_EQ);
            WARMUP_OFF;

            cv::Mat ocl_mat_dst;
            d_dst.download(ocl_mat_dst);

            TestSystem::instance().setAccurate(ExpectedMatNear(ocl_mat_dst, dst, 0.0));

            GPU_ON;
            ocl::compare(d_src1, d_src2, d_dst, CMP_EQ);
            GPU_OFF;

            GPU_FULL_ON;
            d_src1.upload(src1);
            d_src2.upload(src2);
            ocl::compare(d_src1, d_src2, d_dst, CMP_EQ);
            d_dst.download(dst);
            GPU_FULL_OFF;
        }

    }
}

///////////// pow ////////////////////////
PERFTEST(pow)
{
    Mat src, dst;
    ocl::oclMat d_src, d_dst;

    int all_type[] = {CV_32FC1};
    std::string type_name[] = {"CV_32FC1"};

    for (int size = Min_Size; size <= Max_Size; size *= Multiple)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] ;

            gen(src, size, size, all_type[j], 5, 16);

            pow(src, -2.0, dst);

            CPU_ON;
            pow(src, -2.0, dst);
            CPU_OFF;
            d_src.upload(src);
            d_dst.upload(dst);

            WARMUP_ON;
            ocl::pow(d_src, -2.0, d_dst);
            WARMUP_OFF;

            cv::Mat ocl_mat_dst;
            d_dst.download(ocl_mat_dst);

            TestSystem::instance().setAccurate(ExpectedMatNear(ocl_mat_dst, dst, 1.0));

            GPU_ON;
            ocl::pow(d_src, -2.0, d_dst);
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            ocl::pow(d_src, -2.0, d_dst);
            d_dst.download(dst);
            GPU_FULL_OFF;
        }

    }
}

///////////// MagnitudeSqr////////////////////////
PERFTEST(MagnitudeSqr)
{
    Mat src1, src2, dst;
    ocl::oclMat d_src1, d_src2, d_dst;

    int all_type[] = {CV_32FC1};
    std::string type_name[] = {"CV_32FC1"};

    for (int size = Min_Size; size <= Max_Size; size *= Multiple)
    {
        for (size_t t = 0; t < sizeof(all_type) / sizeof(int); t++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[t];

            gen(src1, size, size, all_type[t], 0, 256);
            gen(src2, size, size, all_type[t], 0, 256);
            gen(dst, size, size, all_type[t], 0, 256);


            for (int i = 0; i < src1.rows; ++i)

                for (int j = 0; j < src1.cols; ++j)
                {
                    float val1 = src1.at<float>(i, j);
                    float val2 = src2.at<float>(i, j);

                    ((float *)(dst.data))[i * dst.step / 4 + j] = val1 * val1 + val2 * val2;

                }

                CPU_ON;

                for (int i = 0; i < src1.rows; ++i)
                    for (int j = 0; j < src1.cols; ++j)
                    {
                        float val1 = src1.at<float>(i, j);
                        float val2 = src2.at<float>(i, j);

                        ((float *)(dst.data))[i * dst.step / 4 + j] = val1 * val1 + val2 * val2;

                    }

                    CPU_OFF;
                    d_src1.upload(src1);
                    d_src2.upload(src2);

                    WARMUP_ON;
                    ocl::magnitudeSqr(d_src1, d_src2, d_dst);
                    WARMUP_OFF;

                    cv::Mat ocl_mat_dst;
                    d_dst.download(ocl_mat_dst);

                    TestSystem::instance().setAccurate(ExpectedMatNear(ocl_mat_dst, dst, 1.0));

                    GPU_ON;
                    ocl::magnitudeSqr(d_src1, d_src2, d_dst);
                    GPU_OFF;

                    GPU_FULL_ON;
                    d_src1.upload(src1);
                    d_src2.upload(src2);
                    ocl::magnitudeSqr(d_src1, d_src2, d_dst);
                    d_dst.download(dst);
                    GPU_FULL_OFF;
        }

    }
}

///////////// AddWeighted////////////////////////
PERFTEST(AddWeighted)
{
    Mat src1, src2, dst;
    ocl::oclMat d_src1, d_src2, d_dst;

    double alpha = 2.0, beta = 1.0, gama = 3.0;
    int all_type[] = {CV_8UC1, CV_32FC1};
    std::string type_name[] = {"CV_8UC1", "CV_32FC1"};

    for (int size = Min_Size; size <= Max_Size; size *= Multiple)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] ;

            gen(src1, size, size, all_type[j], 0, 256);
            gen(src2, size, size, all_type[j], 0, 256);
            gen(dst, size, size, all_type[j], 0, 256);


            addWeighted(src1, alpha, src2, beta, gama, dst);

            CPU_ON;
            addWeighted(src1, alpha, src2, beta, gama, dst);
            CPU_OFF;
            d_src1.upload(src1);
            d_src2.upload(src2);

            WARMUP_ON;
            ocl::addWeighted(d_src1, alpha, d_src2, beta, gama, d_dst);
            WARMUP_OFF;

            cv::Mat ocl_mat_dst;
            d_dst.download(ocl_mat_dst);

            TestSystem::instance().setAccurate(ExpectedMatNear(ocl_mat_dst, dst, 1e-5));

            GPU_ON;
            ocl::addWeighted(d_src1, alpha, d_src2, beta, gama, d_dst);
            GPU_OFF;

            GPU_FULL_ON;
            d_src1.upload(src1);
            d_src2.upload(src2);
            ocl::addWeighted(d_src1, alpha, d_src2, beta, gama, d_dst);
            d_dst.download(dst);
            GPU_FULL_OFF;
        }

    }
}