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

///////////// equalizeHist ////////////////////////
PERFTEST(equalizeHist)
{
    Mat src, dst, ocl_dst;
    int all_type[] = {CV_8UC1};
    std::string type_name[] = {"CV_8UC1"};

    for (int size = Min_Size; size <= Max_Size; size *= Multiple)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] ;

            gen(src, size, size, all_type[j], 0, 256);

            equalizeHist(src, dst);

            CPU_ON;
            equalizeHist(src, dst);
            CPU_OFF;

            ocl::oclMat d_src(src);
            ocl::oclMat d_dst;
            ocl::oclMat d_hist;
            ocl::oclMat d_buf;

            WARMUP_ON;
            ocl::equalizeHist(d_src, d_dst);
            WARMUP_OFF;

            GPU_ON;
            ocl::equalizeHist(d_src, d_dst);
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            ocl::equalizeHist(d_src, d_dst);
            d_dst.download(ocl_dst);
            GPU_FULL_OFF;

            TestSystem::instance().ExpectedMatNear(dst, ocl_dst, 1.1);
        }

    }
}
/////////// CopyMakeBorder //////////////////////
PERFTEST(CopyMakeBorder)
{
    Mat src, dst, ocl_dst;
    ocl::oclMat d_dst;

    int bordertype = BORDER_CONSTANT;
    int all_type[] = {CV_8UC1, CV_8UC4};
    std::string type_name[] = {"CV_8UC1", "CV_8UC4"};

    for (int size = Min_Size; size <= Max_Size; size *= Multiple)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] ;


            gen(src, size, size, all_type[j], 0, 256);

            copyMakeBorder(src, dst, 7, 5, 5, 7, bordertype, cv::Scalar(1.0));

            CPU_ON;
            copyMakeBorder(src, dst, 7, 5, 5, 7, bordertype, cv::Scalar(1.0));
            CPU_OFF;

            ocl::oclMat d_src(src);

            WARMUP_ON;
            ocl::copyMakeBorder(d_src, d_dst, 7, 5, 5, 7, bordertype, cv::Scalar(1.0));
            WARMUP_OFF;

            GPU_ON;
            ocl::copyMakeBorder(d_src, d_dst, 7, 5, 5, 7, bordertype, cv::Scalar(1.0));
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            ocl::copyMakeBorder(d_src, d_dst, 7, 5, 5, 7, bordertype, cv::Scalar(1.0));
            d_dst.download(ocl_dst);
            GPU_FULL_OFF;

            TestSystem::instance().ExpectedMatNear(dst, ocl_dst, 0.0);
        }

    }
}
///////////// cornerMinEigenVal ////////////////////////
PERFTEST(cornerMinEigenVal)
{
    Mat src, dst, ocl_dst;
    ocl::oclMat d_dst;

    int blockSize = 7, apertureSize = 1 + 2 * (rand() % 4);
    int borderType = BORDER_REFLECT;
    int all_type[] = {CV_8UC1, CV_32FC1};
    std::string type_name[] = {"CV_8UC1", "CV_32FC1"};

    for (int size = Min_Size; size <= Max_Size; size *= Multiple)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] ;

            gen(src, size, size, all_type[j], 0, 256);

            cornerMinEigenVal(src, dst, blockSize, apertureSize, borderType);

            CPU_ON;
            cornerMinEigenVal(src, dst, blockSize, apertureSize, borderType);
            CPU_OFF;

            ocl::oclMat d_src(src);

            WARMUP_ON;
            ocl::cornerMinEigenVal(d_src, d_dst, blockSize, apertureSize, borderType);
            WARMUP_OFF;

            GPU_ON;
            ocl::cornerMinEigenVal(d_src, d_dst, blockSize, apertureSize, borderType);
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            ocl::cornerMinEigenVal(d_src, d_dst, blockSize, apertureSize, borderType);
            d_dst.download(ocl_dst);
            GPU_FULL_OFF;

            TestSystem::instance().ExpectedMatNear(dst, ocl_dst, 1.0);
        }

    }
}
///////////// cornerHarris ////////////////////////
PERFTEST(cornerHarris)
{
    Mat src, dst, ocl_dst;
    ocl::oclMat d_src, d_dst;

    int all_type[] = {CV_8UC1, CV_32FC1};
    std::string type_name[] = {"CV_8UC1", "CV_32FC1"};

    for (int size = Min_Size; size <= Max_Size; size *= Multiple)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] << " ; BORDER_REFLECT";

            gen(src, size, size, all_type[j], 0, 1);

            cornerHarris(src, dst, 5, 7, 0.1, BORDER_REFLECT);

            CPU_ON;
            cornerHarris(src, dst, 5, 7, 0.1, BORDER_REFLECT);
            CPU_OFF;

            d_src.upload(src);

            WARMUP_ON;
            ocl::cornerHarris(d_src, d_dst, 5, 7, 0.1, BORDER_REFLECT);
            WARMUP_OFF;

            GPU_ON;
            ocl::cornerHarris(d_src, d_dst, 5, 7, 0.1, BORDER_REFLECT);
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            ocl::cornerHarris(d_src, d_dst, 5, 7, 0.1, BORDER_REFLECT);
            d_dst.download(ocl_dst);
            GPU_FULL_OFF;

            TestSystem::instance().ExpectedMatNear(dst, ocl_dst, 1.0);
        }


    }
}
///////////// integral ////////////////////////
PERFTEST(integral)
{
    Mat src, sum, ocl_sum;
    ocl::oclMat d_src, d_sum, d_buf;

    int all_type[] = {CV_8UC1};
    std::string type_name[] = {"CV_8UC1"};

    for (int size = Min_Size; size <= Max_Size; size *= Multiple)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j]  ;

            gen(src, size, size, all_type[j], 0, 256);

            integral(src, sum);

            CPU_ON;
            integral(src, sum);
            CPU_OFF;

            d_src.upload(src);

            WARMUP_ON;
            ocl::integral(d_src, d_sum);
            WARMUP_OFF;

            GPU_ON;
            ocl::integral(d_src, d_sum);
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            ocl::integral(d_src, d_sum);
            d_sum.download(ocl_sum);
            GPU_FULL_OFF;

            if(sum.type() == ocl_sum.type()) //we won't test accuracy when cpu function overlow
                TestSystem::instance().ExpectedMatNear(sum, ocl_sum, 0.0);

        }

    }
}
///////////// WarpAffine ////////////////////////
PERFTEST(WarpAffine)
{
    Mat src, dst, ocl_dst;
    ocl::oclMat d_src, d_dst;

    static const double coeffs[2][3] =
    {
        {cos(CV_PI / 6), -sin(CV_PI / 6), 100.0},
        {sin(CV_PI / 6), cos(CV_PI / 6), -100.0}
    };
    Mat M(2, 3, CV_64F, (void *)coeffs);
    int interpolation = INTER_NEAREST;

    int all_type[] = {CV_8UC1, CV_8UC4};
    std::string type_name[] = {"CV_8UC1", "CV_8UC4"};


    for (int size = Min_Size; size <= Max_Size; size *= Multiple)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] ;

            gen(src, size, size, all_type[j], 0, 256);
            gen(dst, size, size, all_type[j], 0, 256);
            Size size1 = Size(size, size);

            warpAffine(src, dst, M, size1, interpolation);

            CPU_ON;
            warpAffine(src, dst, M, size1, interpolation);
            CPU_OFF;

            d_src.upload(src);

            WARMUP_ON;
            ocl::warpAffine(d_src, d_dst, M, size1, interpolation);
            WARMUP_OFF;

            GPU_ON;
            ocl::warpAffine(d_src, d_dst, M, size1, interpolation);
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            ocl::warpAffine(d_src, d_dst, M, size1, interpolation);
            d_dst.download(ocl_dst);
            GPU_FULL_OFF;

            TestSystem::instance().ExpectedMatNear(dst, ocl_dst, 1.0);
        }

    }
}
///////////// WarpPerspective ////////////////////////
PERFTEST(WarpPerspective)
{
    Mat src, dst, ocl_dst;
    ocl::oclMat d_src, d_dst;

    static const double coeffs[3][3] =
    {
        {cos(CV_PI / 6), -sin(CV_PI / 6), 100.0},
        {sin(CV_PI / 6), cos(CV_PI / 6), -100.0},
        {0.0, 0.0, 1.0}
    };
    Mat M(3, 3, CV_64F, (void *)coeffs);
    int interpolation = INTER_LINEAR;

    int all_type[] = {CV_8UC1, CV_8UC4};
    std::string type_name[] = {"CV_8UC1", "CV_8UC4"};

    for (int size = Min_Size; size <= Max_Size; size *= Multiple)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] ;

            gen(src, size, size, all_type[j], 0, 256);
            gen(dst, size, size, all_type[j], 0, 256);
            Size size1 = Size(size, size);

            warpPerspective(src, dst, M, size1, interpolation);

            CPU_ON;
            warpPerspective(src, dst, M, size1, interpolation);
            CPU_OFF;

            d_src.upload(src);

            WARMUP_ON;
            ocl::warpPerspective(d_src, d_dst, M, size1, interpolation);
            WARMUP_OFF;

            GPU_ON;
            ocl::warpPerspective(d_src, d_dst, M, size1, interpolation);
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            ocl::warpPerspective(d_src, d_dst, M, size1, interpolation);
            d_dst.download(ocl_dst);
            GPU_FULL_OFF;

            TestSystem::instance().ExpectedMatNear(dst, ocl_dst, 1.0);
        }

    }
}

///////////// resize ////////////////////////
PERFTEST(resize)
{
    Mat src, dst, ocl_dst;
    ocl::oclMat d_src, d_dst;


    int all_type[] = {CV_8UC1, CV_8UC4};
    std::string type_name[] = {"CV_8UC1", "CV_8UC4"};

    for (int size = Min_Size; size <= Max_Size; size *= Multiple)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] << " ; up";

            gen(src, size, size, all_type[j], 0, 256);

            resize(src, dst, Size(), 2.0, 2.0);

            CPU_ON;
            resize(src, dst, Size(), 2.0, 2.0);
            CPU_OFF;

            d_src.upload(src);

            WARMUP_ON;
            ocl::resize(d_src, d_dst, Size(), 2.0, 2.0);
            WARMUP_OFF;

            GPU_ON;
            ocl::resize(d_src, d_dst, Size(), 2.0, 2.0);
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            ocl::resize(d_src, d_dst, Size(), 2.0, 2.0);
            d_dst.download(ocl_dst);
            GPU_FULL_OFF;

            TestSystem::instance().ExpectedMatNear(dst, ocl_dst, 1.0);
        }

    }

    for (int size = Min_Size; size <= Max_Size; size *= Multiple)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] << " ; down";

            gen(src, size, size, all_type[j], 0, 256);

            resize(src, dst, Size(), 0.5, 0.5);

            CPU_ON;
            resize(src, dst, Size(), 0.5, 0.5);
            CPU_OFF;

            d_src.upload(src);

            WARMUP_ON;
            ocl::resize(d_src, d_dst, Size(), 0.5, 0.5);
            WARMUP_OFF;

            GPU_ON;
            ocl::resize(d_src, d_dst, Size(), 0.5, 0.5);
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            ocl::resize(d_src, d_dst, Size(), 0.5, 0.5);
            d_dst.download(ocl_dst);
            GPU_FULL_OFF;

            TestSystem::instance().ExpectedMatNear(dst, ocl_dst, 1.0);
        }

    }
}
///////////// threshold////////////////////////
PERFTEST(threshold)
{
    Mat src, dst, ocl_dst;
    ocl::oclMat d_src, d_dst;

    for (int size = Min_Size; size <= Max_Size; size *= Multiple)
    {
        SUBTEST << size << 'x' << size << "; 8UC1; THRESH_BINARY";

        gen(src, size, size, CV_8U, 0, 100);

        threshold(src, dst, 50.0, 0.0, THRESH_BINARY);

        CPU_ON;
        threshold(src, dst, 50.0, 0.0, THRESH_BINARY);
        CPU_OFF;

        d_src.upload(src);

        WARMUP_ON;
        ocl::threshold(d_src, d_dst, 50.0, 0.0, THRESH_BINARY);
        WARMUP_OFF;

        GPU_ON;
        ocl::threshold(d_src, d_dst, 50.0, 0.0, THRESH_BINARY);
        GPU_OFF;

        GPU_FULL_ON;
        d_src.upload(src);
        ocl::threshold(d_src, d_dst, 50.0, 0.0, THRESH_BINARY);
        d_dst.download(ocl_dst);
        GPU_FULL_OFF;

        TestSystem::instance().ExpectedMatNear(dst, ocl_dst, 1.0);
    }

    for (int size = Min_Size; size <= Max_Size; size *= Multiple)
    {
        SUBTEST << size << 'x' << size << "; 32FC1; THRESH_TRUNC [NPP]";

        gen(src, size, size, CV_32FC1, 0, 100);

        threshold(src, dst, 50.0, 0.0, THRESH_TRUNC);

        CPU_ON;
        threshold(src, dst, 50.0, 0.0, THRESH_TRUNC);
        CPU_OFF;

        d_src.upload(src);

        WARMUP_ON;
        ocl::threshold(d_src, d_dst, 50.0, 0.0, THRESH_TRUNC);
        WARMUP_OFF;

        GPU_ON;
        ocl::threshold(d_src, d_dst, 50.0, 0.0, THRESH_TRUNC);
        GPU_OFF;

        GPU_FULL_ON;
        d_src.upload(src);
        ocl::threshold(d_src, d_dst, 50.0, 0.0, THRESH_TRUNC);
        d_dst.download(ocl_dst);
        GPU_FULL_OFF;

        TestSystem::instance().ExpectedMatNear(dst, ocl_dst, 1.0);
    }
}
///////////// meanShiftFiltering////////////////////////
COOR do_meanShift(int x0, int y0, uchar *sptr, uchar *dptr, int sstep, cv::Size size, int sp, int sr, int maxIter, float eps, int *tab)
{

    int isr2 = sr * sr;
    int c0, c1, c2, c3;
    int iter;
    uchar *ptr = NULL;
    uchar *pstart = NULL;
    int revx = 0, revy = 0;
    c0 = sptr[0];
    c1 = sptr[1];
    c2 = sptr[2];
    c3 = sptr[3];
    // iterate meanshift procedure
    for(iter = 0; iter < maxIter; iter++ )
    {
        int count = 0;
        int s0 = 0, s1 = 0, s2 = 0, sx = 0, sy = 0;

        //mean shift: process pixels in window (p-sigmaSp)x(p+sigmaSp)
        int minx = x0 - sp;
        int miny = y0 - sp;
        int maxx = x0 + sp;
        int maxy = y0 + sp;

        //deal with the image boundary
        if(minx < 0) minx = 0;
        if(miny < 0) miny = 0;
        if(maxx >= size.width) maxx = size.width - 1;
        if(maxy >= size.height) maxy = size.height - 1;
        if(iter == 0)
        {
            pstart = sptr;
        }
        else
        {
            pstart = pstart + revy * sstep + (revx << 2); //point to the new position
        }
        ptr = pstart;
        ptr = ptr + (miny - y0) * sstep + ((minx - x0) << 2); //point to the start in the row

        for( int y = miny; y <= maxy; y++, ptr += sstep - ((maxx - minx + 1) << 2))
        {
            int rowCount = 0;
            int x = minx;
#if CV_ENABLE_UNROLLED
            for( ; x + 4 <= maxx; x += 4, ptr += 16)
            {
                int t0, t1, t2;
                t0 = ptr[0], t1 = ptr[1], t2 = ptr[2];
                if(tab[t0 - c0 + 255] + tab[t1 - c1 + 255] + tab[t2 - c2 + 255] <= isr2)
                {
                    s0 += t0;
                    s1 += t1;
                    s2 += t2;
                    sx += x;
                    rowCount++;
                }
                t0 = ptr[4], t1 = ptr[5], t2 = ptr[6];
                if(tab[t0 - c0 + 255] + tab[t1 - c1 + 255] + tab[t2 - c2 + 255] <= isr2)
                {
                    s0 += t0;
                    s1 += t1;
                    s2 += t2;
                    sx += x + 1;
                    rowCount++;
                }
                t0 = ptr[8], t1 = ptr[9], t2 = ptr[10];
                if(tab[t0 - c0 + 255] + tab[t1 - c1 + 255] + tab[t2 - c2 + 255] <= isr2)
                {
                    s0 += t0;
                    s1 += t1;
                    s2 += t2;
                    sx += x + 2;
                    rowCount++;
                }
                t0 = ptr[12], t1 = ptr[13], t2 = ptr[14];
                if(tab[t0 - c0 + 255] + tab[t1 - c1 + 255] + tab[t2 - c2 + 255] <= isr2)
                {
                    s0 += t0;
                    s1 += t1;
                    s2 += t2;
                    sx += x + 3;
                    rowCount++;
                }
            }
#endif
            for(; x <= maxx; x++, ptr += 4)
            {
                int t0 = ptr[0], t1 = ptr[1], t2 = ptr[2];
                if(tab[t0 - c0 + 255] + tab[t1 - c1 + 255] + tab[t2 - c2 + 255] <= isr2)
                {
                    s0 += t0;
                    s1 += t1;
                    s2 += t2;
                    sx += x;
                    rowCount++;
                }
            }
            if(rowCount == 0)
                continue;
            count += rowCount;
            sy += y * rowCount;
        }

        if( count == 0 )
            break;

        int x1 = sx / count;
        int y1 = sy / count;
        s0 = s0 / count;
        s1 = s1 / count;
        s2 = s2 / count;

        bool stopFlag = (x0 == x1 && y0 == y1) || (abs(x1 - x0) + abs(y1 - y0) +
            tab[s0 - c0 + 255] + tab[s1 - c1 + 255] + tab[s2 - c2 + 255] <= eps);

        //revise the pointer corresponding to the new (y0,x0)
        revx = x1 - x0;
        revy = y1 - y0;

        x0 = x1;
        y0 = y1;
        c0 = s0;
        c1 = s1;
        c2 = s2;

        if( stopFlag )
            break;
    } //for iter

    dptr[0] = (uchar)c0;
    dptr[1] = (uchar)c1;
    dptr[2] = (uchar)c2;
    dptr[3] = (uchar)c3;

    COOR coor;
    coor.x = static_cast<short>(x0);
    coor.y = static_cast<short>(y0);
    return coor;
}

static void meanShiftFiltering_(const Mat &src_roi, Mat &dst_roi, int sp, int sr, cv::TermCriteria crit)
{
    if( src_roi.empty() )
        CV_Error( Error::StsBadArg, "The input image is empty" );

    if( src_roi.depth() != CV_8U || src_roi.channels() != 4 )
        CV_Error( Error::StsUnsupportedFormat, "Only 8-bit, 4-channel images are supported" );

    dst_roi.create(src_roi.size(), src_roi.type());

    CV_Assert( (src_roi.cols == dst_roi.cols) && (src_roi.rows == dst_roi.rows) );
    CV_Assert( !(dst_roi.step & 0x3) );

    if( !(crit.type & cv::TermCriteria::MAX_ITER) )
        crit.maxCount = 5;
    int maxIter = std::min(std::max(crit.maxCount, 1), 100);
    float eps;
    if( !(crit.type & cv::TermCriteria::EPS) )
        eps = 1.f;
    eps = (float)std::max(crit.epsilon, 0.0);

    int tab[512];
    for(int i = 0; i < 512; i++)
        tab[i] = (i - 255) * (i - 255);
    uchar *sptr = src_roi.data;
    uchar *dptr = dst_roi.data;
    int sstep = (int)src_roi.step;
    int dstep = (int)dst_roi.step;
    cv::Size size = src_roi.size();

    for(int i = 0; i < size.height; i++, sptr += sstep - (size.width << 2),
        dptr += dstep - (size.width << 2))
    {
        for(int j = 0; j < size.width; j++, sptr += 4, dptr += 4)
        {
            do_meanShift(j, i, sptr, dptr, sstep, size, sp, sr, maxIter, eps, tab);
        }
    }
}

PERFTEST(meanShiftFiltering)
{
    int sp = 5, sr = 6;
    Mat src, dst, ocl_dst;

    ocl::oclMat d_src, d_dst;

    for (int size = Min_Size; size <= Max_Size; size *= Multiple)
    {
        SUBTEST << size << 'x' << size << "; 8UC3 vs 8UC4";

        gen(src, size, size, CV_8UC4, Scalar::all(0), Scalar::all(256));

        cv::TermCriteria crit(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 5, 1);

        meanShiftFiltering_(src, dst, sp, sr, crit);

        CPU_ON;
        meanShiftFiltering_(src, dst, sp, sr, crit);
        CPU_OFF;

        d_src.upload(src);

        WARMUP_ON;
        ocl::meanShiftFiltering(d_src, d_dst, sp, sr, crit);
        WARMUP_OFF;

        GPU_ON;
        ocl::meanShiftFiltering(d_src, d_dst, sp, sr, crit);
        GPU_OFF;

        GPU_FULL_ON;
        d_src.upload(src);
        ocl::meanShiftFiltering(d_src, d_dst, sp, sr, crit);
        d_dst.download(ocl_dst);
        GPU_FULL_OFF;

        TestSystem::instance().ExpectedMatNear(dst, ocl_dst, 0.0);
    }
}

void meanShiftProc_(const Mat &src_roi, Mat &dst_roi, Mat &dstCoor_roi, int sp, int sr, cv::TermCriteria crit)
{
    if (src_roi.empty())
    {
        CV_Error(Error::StsBadArg, "The input image is empty");
    }
    if (src_roi.depth() != CV_8U || src_roi.channels() != 4)
    {
        CV_Error(Error::StsUnsupportedFormat, "Only 8-bit, 4-channel images are supported");
    }

    dst_roi.create(src_roi.size(), src_roi.type());
    dstCoor_roi.create(src_roi.size(), CV_16SC2);

    CV_Assert((src_roi.cols == dst_roi.cols) && (src_roi.rows == dst_roi.rows) &&
              (src_roi.cols == dstCoor_roi.cols) && (src_roi.rows == dstCoor_roi.rows));
    CV_Assert(!(dstCoor_roi.step & 0x3));

    if (!(crit.type & cv::TermCriteria::MAX_ITER))
    {
        crit.maxCount = 5;
    }

    int maxIter = std::min(std::max(crit.maxCount, 1), 100);
    float eps;

    if (!(crit.type & cv::TermCriteria::EPS))
    {
        eps = 1.f;
    }

    eps = (float)std::max(crit.epsilon, 0.0);

    int tab[512];

    for (int i = 0; i < 512; i++)
    {
        tab[i] = (i - 255) * (i - 255);
    }

    uchar *sptr = src_roi.data;
    uchar *dptr = dst_roi.data;
    short *dCoorptr = (short *)dstCoor_roi.data;
    int sstep = (int)src_roi.step;
    int dstep = (int)dst_roi.step;
    int dCoorstep = (int)dstCoor_roi.step >> 1;
    cv::Size size = src_roi.size();

    for (int i = 0; i < size.height; i++, sptr += sstep - (size.width << 2),
            dptr += dstep - (size.width << 2), dCoorptr += dCoorstep - (size.width << 1))
    {
        for (int j = 0; j < size.width; j++, sptr += 4, dptr += 4, dCoorptr += 2)
        {
            *((COOR *)dCoorptr) = do_meanShift(j, i, sptr, dptr, sstep, size, sp, sr, maxIter, eps, tab);
        }
    }

}
PERFTEST(meanShiftProc)
{
    Mat src;
    vector<Mat> dst(2), ocl_dst(2);
    ocl::oclMat d_src, d_dst, d_dstCoor;

    TermCriteria crit(TermCriteria::COUNT + TermCriteria::EPS, 5, 1);

    for (int size = Min_Size; size <= Max_Size; size *= Multiple)
    {
        SUBTEST << size << 'x' << size << "; 8UC4 and CV_16SC2 ";

        gen(src, size, size, CV_8UC4, Scalar::all(0), Scalar::all(256));

        meanShiftProc_(src, dst[0], dst[1], 5, 6, crit);

        CPU_ON;
        meanShiftProc_(src, dst[0], dst[1], 5, 6, crit);
        CPU_OFF;

        d_src.upload(src);

        WARMUP_ON;
        ocl::meanShiftProc(d_src, d_dst, d_dstCoor, 5, 6, crit);
        WARMUP_OFF;

        GPU_ON;
        ocl::meanShiftProc(d_src, d_dst, d_dstCoor, 5, 6, crit);
        GPU_OFF;

        GPU_FULL_ON;
        d_src.upload(src);
        ocl::meanShiftProc(d_src, d_dst, d_dstCoor, 5, 6, crit);
        d_dst.download(ocl_dst[0]);
        d_dstCoor.download(ocl_dst[1]);
        GPU_FULL_OFF;

        vector<double> eps(2, 0.);
        TestSystem::instance().ExpectMatsNear(dst, ocl_dst, eps);
    }
}

///////////// remap////////////////////////
PERFTEST(remap)
{
    Mat src, dst, xmap, ymap, ocl_dst;
    ocl::oclMat d_src, d_dst, d_xmap, d_ymap;

    int all_type[] = {CV_8UC1, CV_8UC4};
    std::string type_name[] = {"CV_8UC1", "CV_8UC4"};

    int interpolation = INTER_LINEAR;
    int borderMode = BORDER_CONSTANT;

    for (int size = Min_Size; size <= Max_Size; size *= Multiple)
    {
        for (size_t t = 0; t < sizeof(all_type) / sizeof(int); t++)
        {
            SUBTEST << size << 'x' << size << "; src " << type_name[t] << "; map CV_32FC1";

            gen(src, size, size, all_type[t], 0, 256);

            xmap.create(size, size, CV_32FC1);
            dst.create(size, size, CV_32FC1);
            ymap.create(size, size, CV_32FC1);

            for (int i = 0; i < size; ++i)
            {
                float *xmap_row = xmap.ptr<float>(i);
                float *ymap_row = ymap.ptr<float>(i);

                for (int j = 0; j < size; ++j)
                {
                    xmap_row[j] = (j - size * 0.5f) * 0.75f + size * 0.5f;
                    ymap_row[j] = (i - size * 0.5f) * 0.75f + size * 0.5f;
                }
            }

            remap(src, dst, xmap, ymap, interpolation, borderMode);

            CPU_ON;
            remap(src, dst, xmap, ymap, interpolation, borderMode);
            CPU_OFF;

            d_src.upload(src);
            d_dst.upload(dst);
            d_xmap.upload(xmap);
            d_ymap.upload(ymap);

            WARMUP_ON;
            ocl::remap(d_src, d_dst, d_xmap, d_ymap, interpolation, borderMode);
            WARMUP_OFF;

            GPU_ON;
            ocl::remap(d_src, d_dst, d_xmap, d_ymap, interpolation, borderMode);
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            ocl::remap(d_src, d_dst, d_xmap, d_ymap, interpolation, borderMode);
            d_dst.download(ocl_dst);
            GPU_FULL_OFF;

            TestSystem::instance().ExpectedMatNear(dst, ocl_dst, 2.0);
        }

    }
}
///////////// CLAHE ////////////////////////
PERFTEST(CLAHE)
{
    Mat src, dst, ocl_dst;
    cv::ocl::oclMat d_src, d_dst;
    int all_type[] = {CV_8UC1};
    std::string type_name[] = {"CV_8UC1"};

    double clipLimit = 40.0;

    cv::Ptr<cv::CLAHE> clahe   = cv::createCLAHE(clipLimit);
    cv::Ptr<cv::CLAHE> d_clahe = cv::ocl::createCLAHE(clipLimit);

    for (int size = Min_Size; size <= Max_Size; size *= Multiple)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] ;

            gen(src, size, size, all_type[j], 0, 256);

            CPU_ON;
            clahe->apply(src, dst);
            CPU_OFF;

            d_src.upload(src);

            WARMUP_ON;
            d_clahe->apply(d_src, d_dst);
            WARMUP_OFF;

            ocl_dst = d_dst;

            TestSystem::instance().ExpectedMatNear(dst, ocl_dst, 1.0);

            GPU_ON;
            d_clahe->apply(d_src, d_dst);
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            d_clahe->apply(d_src, d_dst);
            d_dst.download(dst);
            GPU_FULL_OFF;
        }
    }
}

///////////// columnSum////////////////////////
PERFTEST(columnSum)
{
    Mat src, dst, ocl_dst;
    ocl::oclMat d_src, d_dst;

    for (int size = Min_Size; size <= Max_Size; size *= Multiple)
    {
        SUBTEST << size << 'x' << size << "; CV_32FC1";

        gen(src, size, size, CV_32FC1, 0, 256);

        CPU_ON;
        dst.create(src.size(), src.type());
        for (int j = 0; j < src.cols; j++)
            dst.at<float>(0, j) = src.at<float>(0, j);

        for (int i = 1; i < src.rows; ++i)
            for (int j = 0; j < src.cols; ++j)
                dst.at<float>(i, j) = dst.at<float>(i - 1 , j) + src.at<float>(i , j);
        CPU_OFF;

        d_src.upload(src);

        WARMUP_ON;
        ocl::columnSum(d_src, d_dst);
        WARMUP_OFF;

        GPU_ON;
        ocl::columnSum(d_src, d_dst);
        GPU_OFF;

        GPU_FULL_ON;
        d_src.upload(src);
        ocl::columnSum(d_src, d_dst);
        d_dst.download(ocl_dst);
        GPU_FULL_OFF;

        TestSystem::instance().ExpectedMatNear(dst, ocl_dst, 5e-1);
    }
}
