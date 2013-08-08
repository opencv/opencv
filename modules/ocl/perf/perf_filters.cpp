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
#include "perf_precomp.hpp"

///////////// Blur////////////////////////
PERFTEST(Blur)
{
    Mat src1, dst, ocl_dst;
    ocl::oclMat d_src1, d_dst;

    Size ksize = Size(3, 3);
    int bordertype = BORDER_CONSTANT;
    int all_type[] = {CV_8UC1, CV_8UC4};
    std::string type_name[] = {"CV_8UC1", "CV_8UC4"};

    for (int size = Min_Size; size <= Max_Size; size *= Multiple)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] ;

            gen(src1, size, size, all_type[j], 0, 256);
            gen(dst, size, size, all_type[j], 0, 256);

            blur(src1, dst, ksize, Point(-1, -1), bordertype);

            CPU_ON;
            blur(src1, dst, ksize, Point(-1, -1), bordertype);
            CPU_OFF;

            d_src1.upload(src1);

            WARMUP_ON;
            ocl::blur(d_src1, d_dst, ksize, Point(-1, -1), bordertype);
            WARMUP_OFF;

            GPU_ON;
            ocl::blur(d_src1, d_dst, ksize, Point(-1, -1), bordertype);
            GPU_OFF;

            GPU_FULL_ON;
            d_src1.upload(src1);
            ocl::blur(d_src1, d_dst, ksize, Point(-1, -1), bordertype);
            d_dst.download(ocl_dst);
            GPU_FULL_OFF;

            TestSystem::instance().ExpectedMatNear(ocl_dst, dst, 1.0);
        }

    }
}
///////////// Laplacian////////////////////////
PERFTEST(Laplacian)
{
    Mat src1, dst, ocl_dst;
    ocl::oclMat d_src1, d_dst;

    int ksize = 3;
    int all_type[] = {CV_8UC1, CV_8UC4};
    std::string type_name[] = {"CV_8UC1", "CV_8UC4"};

    for (int size = Min_Size; size <= Max_Size; size *= Multiple)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] ;

            gen(src1, size, size, all_type[j], 0, 256);
            gen(dst, size, size, all_type[j], 0, 256);

            Laplacian(src1, dst, -1, ksize, 1);

            CPU_ON;
            Laplacian(src1, dst, -1, ksize, 1);
            CPU_OFF;

            d_src1.upload(src1);

            WARMUP_ON;
            ocl::Laplacian(d_src1, d_dst, -1, ksize, 1);
            WARMUP_OFF;

            GPU_ON;
            ocl::Laplacian(d_src1, d_dst, -1, ksize, 1);
            GPU_OFF;

            GPU_FULL_ON;
            d_src1.upload(src1);
            ocl::Laplacian(d_src1, d_dst, -1, ksize, 1);
            d_dst.download(ocl_dst);
            GPU_FULL_OFF;

            TestSystem::instance().ExpectedMatNear(ocl_dst, dst, 1e-5);
        }

    }
}

///////////// Erode ////////////////////
PERFTEST(Erode)
{
    Mat src, dst, ker, ocl_dst;
    ocl::oclMat d_src, d_dst;

    int all_type[] = {CV_8UC1, CV_8UC4, CV_32FC1, CV_32FC4};
    std::string type_name[] = {"CV_8UC1", "CV_8UC4", "CV_32FC1", "CV_32FC4"};

    for (int size = Min_Size; size <= Max_Size; size *= Multiple)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] ;

            gen(src, size, size, all_type[j], Scalar::all(0), Scalar::all(256));
            ker = getStructuringElement(MORPH_RECT, Size(3, 3));

            erode(src, dst, ker);

            CPU_ON;
            erode(src, dst, ker);
            CPU_OFF;

            d_src.upload(src);

            WARMUP_ON;
            ocl::erode(d_src, d_dst, ker);
            WARMUP_OFF;

            GPU_ON;
            ocl::erode(d_src, d_dst, ker);
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            ocl::erode(d_src, d_dst, ker);
            d_dst.download(ocl_dst);
            GPU_FULL_OFF;

            TestSystem::instance().ExpectedMatNear(ocl_dst, dst, 1e-5);
        }

    }
}

///////////// Sobel ////////////////////////
PERFTEST(Sobel)
{
    Mat src, dst, ocl_dst;
    ocl::oclMat d_src, d_dst;

    int dx = 1;
    int dy = 1;
    int all_type[] = {CV_8UC1, CV_8UC4};
    std::string type_name[] = {"CV_8UC1", "CV_8UC4"};

    for (int size = Min_Size; size <= Max_Size; size *= Multiple)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] ;

            gen(src, size, size, all_type[j], 0, 256);

            Sobel(src, dst, -1, dx, dy);

            CPU_ON;
            Sobel(src, dst, -1, dx, dy);
            CPU_OFF;

            d_src.upload(src);

            WARMUP_ON;
            ocl::Sobel(d_src, d_dst, -1, dx, dy);
            WARMUP_OFF;

            GPU_ON;
            ocl::Sobel(d_src, d_dst, -1, dx, dy);
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            ocl::Sobel(d_src, d_dst, -1, dx, dy);
            d_dst.download(ocl_dst);
            GPU_FULL_OFF;

            TestSystem::instance().ExpectedMatNear(ocl_dst, dst, 1);
        }

    }
}
///////////// Scharr ////////////////////////
PERFTEST(Scharr)
{
    Mat src, dst, ocl_dst;
    ocl::oclMat d_src, d_dst;

    int dx = 1;
    int dy = 0;
    int all_type[] = {CV_8UC1, CV_8UC4};
    std::string type_name[] = {"CV_8UC1", "CV_8UC4"};

    for (int size = Min_Size; size <= Max_Size; size *= Multiple)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] ;

            gen(src, size, size, all_type[j], 0, 256);

            Scharr(src, dst, -1, dx, dy);

            CPU_ON;
            Scharr(src, dst, -1, dx, dy);
            CPU_OFF;

            d_src.upload(src);

            WARMUP_ON;
            ocl::Scharr(d_src, d_dst, -1, dx, dy);
            WARMUP_OFF;

            GPU_ON;
            ocl::Scharr(d_src, d_dst, -1, dx, dy);
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            ocl::Scharr(d_src, d_dst, -1, dx, dy);
            d_dst.download(ocl_dst);
            GPU_FULL_OFF;

            TestSystem::instance().ExpectedMatNear(ocl_dst, dst, 1);
        }

    }
}

///////////// GaussianBlur ////////////////////////
PERFTEST(GaussianBlur)
{
    Mat src, dst, ocl_dst;
    int all_type[] = {CV_8UC1, CV_8UC4, CV_32FC1, CV_32FC4};
    std::string type_name[] = {"CV_8UC1", "CV_8UC4", "CV_32FC1", "CV_32FC4"};
    const int ksize = 7;	

    for (int size = Min_Size; size <= Max_Size; size *= Multiple)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] ;

            gen(src, size, size, all_type[j], 0, 256);

            GaussianBlur(src, dst, Size(ksize, ksize), 0);

            CPU_ON;
            GaussianBlur(src, dst, Size(ksize, ksize), 0);
            CPU_OFF;

            ocl::oclMat d_src(src);
            ocl::oclMat d_dst;

            WARMUP_ON;
            ocl::GaussianBlur(d_src, d_dst, Size(ksize, ksize), 0);
            WARMUP_OFF;

            GPU_ON;
            ocl::GaussianBlur(d_src, d_dst, Size(ksize, ksize), 0);
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            ocl::GaussianBlur(d_src, d_dst, Size(ksize, ksize), 0);
            d_dst.download(ocl_dst);
            GPU_FULL_OFF;

            TestSystem::instance().ExpectedMatNear(ocl_dst, dst, 1.0);
        }

    }
}

///////////// filter2D////////////////////////
PERFTEST(filter2D)
{
    Mat src;

    for (int size = Min_Size; size <= Max_Size; size *= Multiple)
    {
        int all_type[] = {CV_8UC1, CV_8UC4};
        std::string type_name[] = {"CV_8UC1", "CV_8UC4"};

        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            gen(src, size, size, all_type[j], 0, 256);

            const int ksize = 3;

            SUBTEST << "ksize = " << ksize << "; " << size << 'x' << size << "; " << type_name[j] ;

            Mat kernel;
            gen(kernel, ksize, ksize, CV_32SC1, -3.0, 3.0);

            Mat dst, ocl_dst;

            cv::filter2D(src, dst, -1, kernel);

            CPU_ON;
            cv::filter2D(src, dst, -1, kernel);
            CPU_OFF;

            ocl::oclMat d_src(src), d_dst;

            WARMUP_ON;
            ocl::filter2D(d_src, d_dst, -1, kernel);
            WARMUP_OFF;

            GPU_ON;
            ocl::filter2D(d_src, d_dst, -1, kernel);
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            ocl::filter2D(d_src, d_dst, -1, kernel);
            d_dst.download(ocl_dst);
            GPU_FULL_OFF;

            TestSystem::instance().ExpectedMatNear(ocl_dst, dst, 1e-5);

        }


    }
}