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

using namespace perf;
using std::tr1::tuple;
using std::tr1::get;

///////////// equalizeHist ////////////////////////

typedef TestBaseWithParam<Size> equalizeHistFixture;

PERF_TEST_P(equalizeHistFixture, equalizeHist, OCL_TYPICAL_MAT_SIZES)
{
    const Size srcSize = GetParam();

    Mat src(srcSize, CV_8UC1), dst(srcSize, CV_8UC1);
    declare.in(src, WARMUP_RNG).out(dst);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src), oclDst(srcSize, src.type());

        OCL_TEST_CYCLE() cv::ocl::equalizeHist(oclSrc, oclDst);

        oclDst.download(dst);

        SANITY_CHECK(dst, 1 + DBL_EPSILON);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::equalizeHist(src, dst);

        SANITY_CHECK(dst, 1 + DBL_EPSILON);
    }
    else
        OCL_PERF_ELSE
}

/////////// CopyMakeBorder //////////////////////

typedef Size_MatType CopyMakeBorderFixture;

PERF_TEST_P(CopyMakeBorderFixture, CopyMakeBorder,
            ::testing::Combine(OCL_TYPICAL_MAT_SIZES,
                               OCL_PERF_ENUM(CV_8UC1, CV_8UC4)))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params), borderType = BORDER_CONSTANT;

    Mat src(srcSize, type), dst;
    const Size dstSize = srcSize + Size(12, 12);
    dst.create(dstSize, type);
    declare.in(src, WARMUP_RNG).out(dst);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src), oclDst(dstSize, type);

        OCL_TEST_CYCLE() cv::ocl::copyMakeBorder(oclSrc, oclDst, 7, 5, 5, 7, borderType, cv::Scalar(1.0));

        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::copyMakeBorder(src, dst, 7, 5, 5, 7, borderType, cv::Scalar(1.0));

        SANITY_CHECK(dst);
    }
    else
        OCL_PERF_ELSE
}

///////////// cornerMinEigenVal ////////////////////////

typedef Size_MatType cornerMinEigenValFixture;

PERF_TEST_P(cornerMinEigenValFixture, cornerMinEigenVal,
            ::testing::Combine(OCL_TYPICAL_MAT_SIZES,
                               OCL_PERF_ENUM(CV_8UC1, CV_32FC1)))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params), borderType = BORDER_REFLECT;
    const int blockSize = 7, apertureSize = 1 + 2 * 3;

    Mat src(srcSize, type), dst(srcSize, CV_32FC1);
    declare.in(src, WARMUP_RNG).out(dst)
            .time(srcSize == OCL_SIZE_4000 ? 20 : srcSize == OCL_SIZE_2000 ? 5 : 3);

    const int depth = CV_MAT_DEPTH(type);
    const ERROR_TYPE errorType = depth == CV_8U ? ERROR_ABSOLUTE : ERROR_RELATIVE;

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src), oclDst(srcSize, CV_32FC1);

        OCL_TEST_CYCLE() cv::ocl::cornerMinEigenVal(oclSrc, oclDst, blockSize, apertureSize, borderType);

        oclDst.download(dst);

        SANITY_CHECK(dst, 1e-6, errorType);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::cornerMinEigenVal(src, dst, blockSize, apertureSize, borderType);

        SANITY_CHECK(dst, 1e-6, errorType);
    }
    else
        OCL_PERF_ELSE
}

///////////// cornerHarris ////////////////////////

typedef Size_MatType cornerHarrisFixture;

PERF_TEST_P(cornerHarrisFixture, cornerHarris,
            ::testing::Combine(OCL_TYPICAL_MAT_SIZES,
                               OCL_PERF_ENUM(CV_8UC1, CV_32FC1)))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params), borderType = BORDER_REFLECT;

    Mat src(srcSize, type), dst(srcSize, CV_32FC1);
    randu(src, 0, 1);
    declare.in(src).out(dst)
            .time(srcSize == OCL_SIZE_4000 ? 20 : srcSize == OCL_SIZE_2000 ? 5 : 3);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src), oclDst(srcSize, CV_32FC1);

        OCL_TEST_CYCLE() cv::ocl::cornerHarris(oclSrc, oclDst, 5, 7, 0.1, borderType);

        oclDst.download(dst);

        SANITY_CHECK(dst, 3e-5);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::cornerHarris(src, dst, 5, 7, 0.1, borderType);

        SANITY_CHECK(dst, 3e-5);
    }
    else
        OCL_PERF_ELSE
}

///////////// integral ////////////////////////

typedef TestBaseWithParam<Size> integralFixture;

PERF_TEST_P(integralFixture, integral, OCL_TYPICAL_MAT_SIZES)
{
    const Size srcSize = GetParam();

    Mat src(srcSize, CV_8UC1), dst;
    declare.in(src, WARMUP_RNG);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src), oclDst;

        OCL_TEST_CYCLE() cv::ocl::integral(oclSrc, oclDst);

        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::integral(src, dst);

        SANITY_CHECK(dst);
    }
    else
        OCL_PERF_ELSE
}

///////////// WarpAffine ////////////////////////

typedef Size_MatType WarpAffineFixture;

PERF_TEST_P(WarpAffineFixture, WarpAffine,
            ::testing::Combine(OCL_TYPICAL_MAT_SIZES,
                               OCL_PERF_ENUM(CV_8UC1, CV_8UC4)))
{
    static const double coeffs[2][3] =
    {
        { cos(CV_PI / 6), -sin(CV_PI / 6), 100.0 },
        { sin(CV_PI / 6), cos(CV_PI / 6), -100.0 }
    };
    Mat M(2, 3, CV_64F, (void *)coeffs);
    const int interpolation = INTER_NEAREST;

    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    Mat src(srcSize, type), dst(srcSize, type);
    declare.in(src, WARMUP_RNG).out(dst);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src), oclDst(srcSize, type);

        OCL_TEST_CYCLE() cv::ocl::warpAffine(oclSrc, oclDst, M, srcSize, interpolation);

        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::warpAffine(src, dst, M, srcSize, interpolation);

        SANITY_CHECK(dst);
    }
    else
        OCL_PERF_ELSE
}

///////////// WarpPerspective ////////////////////////

typedef Size_MatType WarpPerspectiveFixture;

PERF_TEST_P(WarpPerspectiveFixture, WarpPerspective,
            ::testing::Combine(OCL_TYPICAL_MAT_SIZES,
                               OCL_PERF_ENUM(CV_8UC1, CV_8UC4)))
{
    static const double coeffs[3][3] =
    {
        {cos(CV_PI / 6), -sin(CV_PI / 6), 100.0},
        {sin(CV_PI / 6), cos(CV_PI / 6), -100.0},
        {0.0, 0.0, 1.0}
    };
    Mat M(3, 3, CV_64F, (void *)coeffs);
    const int interpolation = INTER_LINEAR;

    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    Mat src(srcSize, type), dst(srcSize, type);
    declare.in(src, WARMUP_RNG).out(dst)
            .time(srcSize == OCL_SIZE_4000 ? 18 : srcSize == OCL_SIZE_2000 ? 5 : 2);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src), oclDst(srcSize, type);

        OCL_TEST_CYCLE() cv::ocl::warpPerspective(oclSrc, oclDst, M, srcSize, interpolation);

        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::warpPerspective(src, dst, M, srcSize, interpolation);

        SANITY_CHECK(dst);
    }
    else
        OCL_PERF_ELSE
}

///////////// resize ////////////////////////

CV_ENUM(resizeInterType, INTER_NEAREST, INTER_LINEAR)

typedef tuple<Size, MatType, resizeInterType, double> resizeParams;
typedef TestBaseWithParam<resizeParams> resizeFixture;

PERF_TEST_P(resizeFixture, resize,
            ::testing::Combine(OCL_TYPICAL_MAT_SIZES,
                               OCL_PERF_ENUM(CV_8UC1, CV_8UC4),
                               resizeInterType::all(),
                               ::testing::Values(0.5, 2.0)))
{
    const resizeParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params), interType = get<2>(params);
    double scale = get<3>(params);

    Mat src(srcSize, type), dst;
    const Size dstSize(cvRound(srcSize.width * scale), cvRound(srcSize.height * scale));
    dst.create(dstSize, type);
    declare.in(src, WARMUP_RNG).out(dst);
    if (interType == INTER_LINEAR && type == CV_8UC4 && OCL_SIZE_4000 == srcSize)
        declare.time(11);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src), oclDst(dstSize, type);

        OCL_TEST_CYCLE() cv::ocl::resize(oclSrc, oclDst, Size(), scale, scale, interType);

        oclDst.download(dst);

        SANITY_CHECK(dst, 1 + DBL_EPSILON);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::resize(src, dst, Size(), scale, scale, interType);

        SANITY_CHECK(dst, 1 + DBL_EPSILON);
    }
    else
        OCL_PERF_ELSE
}

///////////// threshold////////////////////////

CV_ENUM(ThreshType, THRESH_BINARY, THRESH_TRUNC)

typedef tuple<Size, ThreshType> ThreshParams;
typedef TestBaseWithParam<ThreshParams> ThreshFixture;

PERF_TEST_P(ThreshFixture, threshold,
            ::testing::Combine(OCL_TYPICAL_MAT_SIZES,
                               ThreshType::all()))
{
    const ThreshParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int threshType = get<1>(params);

    Mat src(srcSize, CV_8U), dst(srcSize, CV_8U);
    randu(src, 0, 100);
    declare.in(src).out(dst);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src), oclDst(srcSize, CV_8U);

        OCL_TEST_CYCLE() cv::ocl::threshold(oclSrc, oclDst, 50.0, 0.0, threshType);

        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::threshold(src, dst, 50.0, 0.0, threshType);

        SANITY_CHECK(dst);
    }
    else
        OCL_PERF_ELSE
}

///////////// meanShiftFiltering////////////////////////

typedef struct _COOR
{
    short x;
    short y;
} COOR;

static COOR do_meanShift(int x0, int y0, uchar *sptr, uchar *dptr, int sstep, cv::Size size, int sp, int sr, int maxIter, float eps, int *tab)
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

typedef TestBaseWithParam<Size> meanShiftFilteringFixture;

PERF_TEST_P(meanShiftFilteringFixture, meanShiftFiltering,
            OCL_TYPICAL_MAT_SIZES)
{
    const Size srcSize = GetParam();
    const int sp = 5, sr = 6;
    cv::TermCriteria crit(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 5, 1);

    Mat src(srcSize, CV_8UC4), dst(srcSize, CV_8UC4);
    declare.in(src, WARMUP_RNG).out(dst)
            .time(srcSize == OCL_SIZE_4000 ?
                      56 : srcSize == OCL_SIZE_2000 ? 15 : 3.8);

    if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() meanShiftFiltering_(src, dst, sp, sr, crit);

        SANITY_CHECK(dst);
    }
    else if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src), oclDst(srcSize, CV_8UC4);

        OCL_TEST_CYCLE() ocl::meanShiftFiltering(oclSrc, oclDst, sp, sr, crit);

        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else
        OCL_PERF_ELSE
}

static void meanShiftProc_(const Mat &src_roi, Mat &dst_roi, Mat &dstCoor_roi, int sp, int sr, cv::TermCriteria crit)
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

typedef TestBaseWithParam<Size> meanShiftProcFixture;

PERF_TEST_P(meanShiftProcFixture, meanShiftProc,
            OCL_TYPICAL_MAT_SIZES)
{
    const Size srcSize = GetParam();
    TermCriteria crit(TermCriteria::COUNT + TermCriteria::EPS, 5, 1);

    Mat src(srcSize, CV_8UC4), dst1(srcSize, CV_8UC4),
            dst2(srcSize, CV_16SC2);
    declare.in(src, WARMUP_RNG).out(dst1, dst2)
            .time(srcSize == OCL_SIZE_4000 ?
                      56 : srcSize == OCL_SIZE_2000 ? 15 : 3.8);;

    if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() meanShiftProc_(src, dst1, dst2, 5, 6, crit);

        SANITY_CHECK(dst1);
        SANITY_CHECK(dst2);
    }
    else if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src), oclDst1(srcSize, CV_8UC4),
                oclDst2(srcSize, CV_16SC2);

        OCL_TEST_CYCLE() ocl::meanShiftProc(oclSrc, oclDst1, oclDst2, 5, 6, crit);

        oclDst1.download(dst1);
        oclDst2.download(dst2);

        SANITY_CHECK(dst1);
        SANITY_CHECK(dst2);
    }
    else
        OCL_PERF_ELSE
}

///////////// remap////////////////////////

CV_ENUM(RemapInterType, INTER_NEAREST, INTER_LINEAR)

typedef tuple<Size, MatType, RemapInterType> remapParams;
typedef TestBaseWithParam<remapParams> remapFixture;

PERF_TEST_P(remapFixture, remap,
            ::testing::Combine(OCL_TYPICAL_MAT_SIZES,
                               OCL_PERF_ENUM(CV_8UC1, CV_8UC4),
                               RemapInterType::all()))
{
    const remapParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params), interpolation = get<2>(params);

    Mat src(srcSize, type), dst(srcSize, type);
    declare.in(src, WARMUP_RNG).out(dst);

    if (srcSize == OCL_SIZE_4000 && interpolation == INTER_LINEAR)
        declare.time(9);

    Mat xmap, ymap;
    xmap.create(srcSize, CV_32FC1);
    ymap.create(srcSize, CV_32FC1);

    for (int i = 0; i < srcSize.height; ++i)
    {
        float * const xmap_row = xmap.ptr<float>(i);
        float * const ymap_row = ymap.ptr<float>(i);

        for (int j = 0; j < srcSize.width; ++j)
        {
            xmap_row[j] = (j - srcSize.width * 0.5f) * 0.75f + srcSize.width * 0.5f;
            ymap_row[j] = (i - srcSize.height * 0.5f) * 0.75f + srcSize.height * 0.5f;
        }
    }

    const int borderMode = BORDER_CONSTANT;

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src), oclDst(srcSize, type);
        ocl::oclMat oclXMap(xmap), oclYMap(ymap);

        OCL_TEST_CYCLE() cv::ocl::remap(oclSrc, oclDst, oclXMap, oclYMap, interpolation, borderMode);

        oclDst.download(dst);

        SANITY_CHECK(dst, 1 + DBL_EPSILON);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::remap(src, dst, xmap, ymap, interpolation, borderMode);

        SANITY_CHECK(dst, 1 + DBL_EPSILON);
    }
    else
        OCL_PERF_ELSE
}

///////////// CLAHE ////////////////////////

typedef TestBaseWithParam<Size> CLAHEFixture;

PERF_TEST_P(CLAHEFixture, CLAHE, OCL_TYPICAL_MAT_SIZES)
{
    const Size srcSize = GetParam();
    const string impl = getSelectedImpl();

    Mat src(srcSize, CV_8UC1), dst;
    const double clipLimit = 40.0;
    declare.in(src, WARMUP_RNG);

    if (srcSize == OCL_SIZE_4000)
        declare.time(11);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src), oclDst;
        cv::Ptr<cv::CLAHE> oclClahe = cv::ocl::createCLAHE(clipLimit);

        OCL_TEST_CYCLE() oclClahe->apply(oclSrc, oclDst);

        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (RUN_PLAIN_IMPL)
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clipLimit);
        TEST_CYCLE() clahe->apply(src, dst);

        SANITY_CHECK(dst);
    }
    else
        OCL_PERF_ELSE
}

///////////// columnSum////////////////////////

typedef TestBaseWithParam<Size> columnSumFixture;

static void columnSumPerfTest(const Mat & src, Mat & dst)
{
    for (int j = 0; j < src.cols; j++)
        dst.at<float>(0, j) = src.at<float>(0, j);

    for (int i = 1; i < src.rows; ++i)
        for (int j = 0; j < src.cols; ++j)
            dst.at<float>(i, j) = dst.at<float>(i - 1 , j) + src.at<float>(i , j);
}

PERF_TEST_P(columnSumFixture, columnSum, OCL_TYPICAL_MAT_SIZES)
{
    const Size srcSize = GetParam();

    Mat src(srcSize, CV_32FC1), dst(srcSize, CV_32FC1);
    declare.in(src, WARMUP_RNG).out(dst);

    if (srcSize == OCL_SIZE_4000)
        declare.time(5);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src), oclDst(srcSize, CV_32FC1);

        OCL_TEST_CYCLE() cv::ocl::columnSum(oclSrc, oclDst);

        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() columnSumPerfTest(src, dst);

        SANITY_CHECK(dst);
    }
    else
        OCL_PERF_ELSE
}
