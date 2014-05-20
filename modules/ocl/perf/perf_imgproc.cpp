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
//     and/or other materials provided with the distribution.
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

typedef TestBaseWithParam<Size> EqualizeHistFixture;

OCL_PERF_TEST_P(EqualizeHistFixture, EqualizeHist, OCL_TEST_SIZES)
{
    const Size srcSize = GetParam();
    const double eps = 1 + DBL_EPSILON;

    Mat src(srcSize, CV_8UC1), dst(srcSize, CV_8UC1);
    declare.in(src, WARMUP_RNG).out(dst);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src), oclDst(srcSize, src.type());

        OCL_TEST_CYCLE() cv::ocl::equalizeHist(oclSrc, oclDst);

        oclDst.download(dst);

        SANITY_CHECK(dst, eps);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::equalizeHist(src, dst);

        SANITY_CHECK(dst, eps);
    }
    else
        OCL_PERF_ELSE
}

///////////// CalcHist ////////////////////////

typedef TestBaseWithParam<Size> CalcHistFixture;

OCL_PERF_TEST_P(CalcHistFixture, CalcHist, OCL_TEST_SIZES)
{
    const Size srcSize = GetParam();
    const std::vector<int> channels(1, 0);
    std::vector<float> ranges(2);
    std::vector<int> histSize(1, 256);
    ranges[0] = 0;
    ranges[1] = 256;

    Mat src(srcSize, CV_8UC1), dst(srcSize, CV_32FC1);
    declare.in(src, WARMUP_RNG).out(dst);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src), oclDst(srcSize, CV_32SC1);

        OCL_TEST_CYCLE() cv::ocl::calcHist(oclSrc, oclDst);

        oclDst.download(dst);
        SANITY_CHECK(dst);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::calcHist(std::vector<Mat>(1, src), channels,
                                  noArray(), dst, histSize, ranges, false);

        dst.convertTo(dst, CV_32S);
        dst = dst.reshape(1, 1);

        SANITY_CHECK(dst);
    }
    else
        OCL_PERF_ELSE
}

/////////// CopyMakeBorder //////////////////////

CV_ENUM(Border, BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT, BORDER_WRAP, BORDER_REFLECT_101)

typedef tuple<Size, MatType, Border> CopyMakeBorderParamType;
typedef TestBaseWithParam<CopyMakeBorderParamType> CopyMakeBorderFixture;

OCL_PERF_TEST_P(CopyMakeBorderFixture, CopyMakeBorder,
            ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES, Border::all()))
{
    const CopyMakeBorderParamType params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params), borderType = get<2>(params);

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

typedef Size_MatType CornerMinEigenValFixture;

OCL_PERF_TEST_P(CornerMinEigenValFixture, CornerMinEigenVal,
            ::testing::Combine(OCL_TEST_SIZES, OCL_PERF_ENUM(CV_8UC1, CV_32FC1)))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params), borderType = BORDER_REFLECT;
    const int blockSize = 7, apertureSize = 1 + 2 * 3;

    Mat src(srcSize, type), dst(srcSize, CV_32FC1);
    declare.in(src, WARMUP_RNG).out(dst);

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

typedef Size_MatType CornerHarrisFixture;

OCL_PERF_TEST_P(CornerHarrisFixture, CornerHarris,
            ::testing::Combine(OCL_TEST_SIZES, OCL_PERF_ENUM(CV_8UC1, CV_32FC1)))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params), borderType = BORDER_REFLECT;

    Mat src(srcSize, type), dst(srcSize, CV_32FC1);
    randu(src, 0, 1);
    declare.in(src).out(dst);

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

typedef tuple<Size, MatDepth> IntegralParams;
typedef TestBaseWithParam<IntegralParams> IntegralFixture;

OCL_PERF_TEST_P(IntegralFixture, DISABLED_Integral1, ::testing::Combine(OCL_TEST_SIZES, OCL_PERF_ENUM(CV_32S, CV_32F)))
{
    const IntegralParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int sdepth = get<1>(params);

    Mat src(srcSize, CV_8UC1), dst;
    declare.in(src, WARMUP_RNG);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src), oclDst;

//        OCL_TEST_CYCLE() cv::ocl::integral(oclSrc, oclDst, sdepth);

        oclDst.download(dst);

        SANITY_CHECK(dst, 1e-6, ERROR_RELATIVE);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::integral(src, dst, sdepth);

        SANITY_CHECK(dst, 1e-6, ERROR_RELATIVE);
    }
    else
        OCL_PERF_ELSE
}

///////////// threshold////////////////////////

CV_ENUM(ThreshType, THRESH_BINARY, THRESH_BINARY_INV, THRESH_TRUNC, THRESH_TOZERO_INV)

typedef tuple<Size, MatType, ThreshType> ThreshParams;
typedef TestBaseWithParam<ThreshParams> ThreshFixture;

OCL_PERF_TEST_P(ThreshFixture, Threshold,
            ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES, ThreshType::all()))
{
    const ThreshParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int srcType = get<1>(params);
    const int threshType = get<2>(params);
    const double maxValue = 220.0, threshold = 50;

    Mat src(srcSize, srcType), dst(srcSize, srcType);
    randu(src, 0, 100);
    declare.in(src).out(dst);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src), oclDst(srcSize, CV_8U);

        OCL_TEST_CYCLE() cv::ocl::threshold(oclSrc, oclDst, threshold, maxValue, threshType);

        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::threshold(src, dst, threshold, maxValue, threshType);

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
        CV_Error( CV_StsBadArg, "The input image is empty" );

    if( src_roi.depth() != CV_8U || src_roi.channels() != 4 )
        CV_Error( CV_StsUnsupportedFormat, "Only 8-bit, 4-channel images are supported" );

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

typedef TestBaseWithParam<Size> MeanShiftFilteringFixture;

PERF_TEST_P(MeanShiftFilteringFixture, MeanShiftFiltering,
            OCL_TYPICAL_MAT_SIZES)
{
    const Size srcSize = GetParam();
    const int sp = 5, sr = 6;
    cv::TermCriteria crit(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 5, 1);

    Mat src(srcSize, CV_8UC4), dst(srcSize, CV_8UC4);
    declare.in(src, WARMUP_RNG).out(dst);

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
        CV_Error(CV_StsBadArg, "The input image is empty");
    }
    if (src_roi.depth() != CV_8U || src_roi.channels() != 4)
    {
        CV_Error(CV_StsUnsupportedFormat, "Only 8-bit, 4-channel images are supported");
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

typedef TestBaseWithParam<Size> MeanShiftProcFixture;

PERF_TEST_P(MeanShiftProcFixture, MeanShiftProc,
            OCL_TYPICAL_MAT_SIZES)
{
    const Size srcSize = GetParam();
    TermCriteria crit(TermCriteria::COUNT + TermCriteria::EPS, 5, 1);

    Mat src(srcSize, CV_8UC4), dst1(srcSize, CV_8UC4),
            dst2(srcSize, CV_16SC2);
    declare.in(src, WARMUP_RNG).out(dst1, dst2);

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

///////////// CLAHE ////////////////////////

typedef TestBaseWithParam<Size> CLAHEFixture;

OCL_PERF_TEST_P(CLAHEFixture, CLAHE, OCL_TEST_SIZES)
{
    const Size srcSize = GetParam();

    Mat src(srcSize, CV_8UC1), dst;
    const double clipLimit = 40.0;
    declare.in(src, WARMUP_RNG);

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

///////////// ColumnSum////////////////////////

typedef TestBaseWithParam<Size> ColumnSumFixture;

static void columnSumPerfTest(const Mat & src, Mat & dst)
{
    for (int j = 0; j < src.cols; j++)
        dst.at<float>(0, j) = src.at<float>(0, j);

    for (int i = 1; i < src.rows; ++i)
        for (int j = 0; j < src.cols; ++j)
            dst.at<float>(i, j) = dst.at<float>(i - 1 , j) + src.at<float>(i , j);
}

PERF_TEST_P(ColumnSumFixture, ColumnSum, OCL_TYPICAL_MAT_SIZES)
{
    const Size srcSize = GetParam();

    Mat src(srcSize, CV_32FC1), dst(srcSize, CV_32FC1);
    declare.in(src, WARMUP_RNG).out(dst);

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

//////////////////////////////distanceToCenters////////////////////////////////////////////////

CV_ENUM(DistType, NORM_L1, NORM_L2SQR)

typedef tuple<Size, DistType> DistanceToCentersParams;
typedef TestBaseWithParam<DistanceToCentersParams> DistanceToCentersFixture;

static void distanceToCentersPerfTest(Mat& src, Mat& centers, Mat& dists, Mat& labels, int distType)
{
    Mat batch_dists;
    cv::batchDistance(src, centers, batch_dists, CV_32FC1, noArray(), distType);

    std::vector<float> dists_v;
    std::vector<int> labels_v;

    for (int i = 0; i < batch_dists.rows; i++)
    {
        Mat r = batch_dists.row(i);
        double mVal;
        Point mLoc;

        minMaxLoc(r, &mVal, NULL, &mLoc, NULL);
        dists_v.push_back(static_cast<float>(mVal));
        labels_v.push_back(mLoc.x);
    }

    Mat(dists_v).copyTo(dists);
    Mat(labels_v).copyTo(labels);
}

PERF_TEST_P(DistanceToCentersFixture, DistanceToCenters, ::testing::Combine(::testing::Values(cv::Size(256,256), cv::Size(512,512)), DistType::all()) )
{
    const DistanceToCentersParams params = GetParam();
    Size size = get<0>(params);
    int distType = get<1>(params);

    Mat src(size, CV_32FC1), centers(size, CV_32FC1);
    Mat dists(src.rows, 1, CV_32FC1), labels(src.rows, 1, CV_32SC1);

    declare.in(src, centers, WARMUP_RNG).out(dists, labels);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat ocl_src(src), ocl_centers(centers);

        OCL_TEST_CYCLE() ocl::distanceToCenters(ocl_src, ocl_centers, dists, labels, distType);

        SANITY_CHECK(dists, 1e-6, ERROR_RELATIVE);
        SANITY_CHECK(labels);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() distanceToCentersPerfTest(src, centers, dists, labels, distType);

        SANITY_CHECK(dists, 1e-6, ERROR_RELATIVE);
        SANITY_CHECK(labels);
    }
    else
        OCL_PERF_ELSE
}
