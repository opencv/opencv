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
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Niko Li, newlife20080214@gmail.com
//    Jia Haipeng, jiahaipeng95@gmail.com
//    Shengen Yan, yanshengen@gmail.com
//    Jiang Liyuan, lyuan001.good@163.com
//    Rock Li, Rock.Li@amd.com
//    Wu Zailong, bullet@yeah.net
//    Xu Pang, pangxu010@163.com
//    Sen Liu, swjtuls1987@126.com
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

#ifdef HAVE_OPENCL

using namespace testing;
using namespace std;
using namespace cv;

typedef struct
{
    short x;
    short y;
} COOR;

COOR do_meanShift(int x0, int y0, uchar *sptr, uchar *dptr, int sstep, Size size, int sp, int sr, int maxIter, float eps, int *tab)
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
    coor.x = (short)x0;
    coor.y = (short)y0;
    return coor;
}

void meanShiftFiltering_(const Mat &src_roi, Mat &dst_roi, int sp, int sr, TermCriteria crit)
{
    if( src_roi.empty() )
        CV_Error( CV_StsBadArg, "The input image is empty" );

    if( src_roi.depth() != CV_8U || src_roi.channels() != 4 )
        CV_Error( CV_StsUnsupportedFormat, "Only 8-bit, 4-channel images are supported" );

    CV_Assert( (src_roi.cols == dst_roi.cols) && (src_roi.rows == dst_roi.rows) );
    CV_Assert( !(dst_roi.step & 0x3) );

    if( !(crit.type & TermCriteria::MAX_ITER) )
        crit.maxCount = 5;
    int maxIter = std::min(std::max(crit.maxCount, 1), 100);
    float eps;
    if( !(crit.type & TermCriteria::EPS) )
        eps = 1.f;
    eps = (float)std::max(crit.epsilon, 0.0);

    int tab[512];
    for(int i = 0; i < 512; i++)
        tab[i] = (i - 255) * (i - 255);
    uchar *sptr = src_roi.data;
    uchar *dptr = dst_roi.data;
    int sstep = (int)src_roi.step;
    int dstep = (int)dst_roi.step;
    Size size = src_roi.size();

    for(int i = 0; i < size.height; i++, sptr += sstep - (size.width << 2),
            dptr += dstep - (size.width << 2))
    {
        for(int j = 0; j < size.width; j++, sptr += 4, dptr += 4)
        {
            do_meanShift(j, i, sptr, dptr, sstep, size, sp, sr, maxIter, eps, tab);
        }
    }
}

void meanShiftProc_(const Mat &src_roi, Mat &dst_roi, Mat &dstCoor_roi, int sp, int sr, TermCriteria crit)
{
    if( src_roi.empty() )
        CV_Error( CV_StsBadArg, "The input image is empty" );
    if( src_roi.depth() != CV_8U || src_roi.channels() != 4 )
        CV_Error( CV_StsUnsupportedFormat, "Only 8-bit, 4-channel images are supported" );
    CV_Assert( (src_roi.cols == dst_roi.cols) && (src_roi.rows == dst_roi.rows) &&
               (src_roi.cols == dstCoor_roi.cols) && (src_roi.rows == dstCoor_roi.rows));
    CV_Assert( !(dstCoor_roi.step & 0x3) );

    if( !(crit.type & TermCriteria::MAX_ITER) )
        crit.maxCount = 5;
    int maxIter = std::min(std::max(crit.maxCount, 1), 100);
    float eps;
    if( !(crit.type & TermCriteria::EPS) )
        eps = 1.f;
    eps = (float)std::max(crit.epsilon, 0.0);

    int tab[512];
    for(int i = 0; i < 512; i++)
        tab[i] = (i - 255) * (i - 255);
    uchar *sptr = src_roi.data;
    uchar *dptr = dst_roi.data;
    short *dCoorptr = (short *)dstCoor_roi.data;
    int sstep = (int)src_roi.step;
    int dstep = (int)dst_roi.step;
    int dCoorstep = (int)dstCoor_roi.step >> 1;
    Size size = src_roi.size();

    for(int i = 0; i < size.height; i++, sptr += sstep - (size.width << 2),
            dptr += dstep - (size.width << 2), dCoorptr += dCoorstep - (size.width << 1))
    {
        for(int j = 0; j < size.width; j++, sptr += 4, dptr += 4, dCoorptr += 2)
        {
            *((COOR *)dCoorptr) = do_meanShift(j, i, sptr, dptr, sstep, size, sp, sr, maxIter, eps, tab);
        }
    }

}

//////////////////////////////// meanShift //////////////////////////////////////////

PARAM_TEST_CASE(meanShiftTestBase, MatType, MatType, int, int, TermCriteria, bool)
{
    int type, typeCoor;
    int sp, sr;
    TermCriteria crit;
    bool useRoi;

    // src mat
    Mat src, src_roi;
    Mat dst, dst_roi;
    Mat dstCoor, dstCoor_roi;

    // ocl dst mat
    ocl::oclMat gsrc, gsrc_roi;
    ocl::oclMat gdst, gdst_roi;
    ocl::oclMat gdstCoor, gdstCoor_roi;

    virtual void SetUp()
    {
        type = GET_PARAM(0);
        typeCoor = GET_PARAM(1);
        sp = GET_PARAM(2);
        sr = GET_PARAM(3);
        crit = GET_PARAM(4);
        useRoi = GET_PARAM(5);
    }

    void random_roi()
    {
        Size roiSize = randomSize(1, MAX_VALUE);
        Border srcBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(src, src_roi, roiSize, srcBorder, type, 5, 256);
        generateOclMat(gsrc, gsrc_roi, src, roiSize, srcBorder);

        Border dstBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(dst, dst_roi, roiSize, dstBorder, type, 5, 256);
        generateOclMat(gdst, gdst_roi, dst, roiSize, dstBorder);

        randomSubMat(dstCoor, dstCoor_roi, roiSize, dstBorder, typeCoor, 5, 256);
        generateOclMat(gdstCoor, gdstCoor_roi, dstCoor, roiSize, dstBorder);
    }

    void Near(double threshold = 0.0)
    {
        Mat whole, roi;
        gdst.download(whole);
        gdst_roi.download(roi);

        EXPECT_MAT_NEAR(dst, whole, threshold);
        EXPECT_MAT_NEAR(dst_roi, roi, threshold);
    }

    void Near1(double threshold = 0.0)
    {
        Mat whole, roi;
        gdstCoor.download(whole);
        gdstCoor_roi.download(roi);

        EXPECT_MAT_NEAR(dstCoor, whole, threshold);
        EXPECT_MAT_NEAR(dstCoor_roi, roi, threshold);
    }
};

/////////////////////////meanShiftFiltering/////////////////////////////

typedef meanShiftTestBase meanShiftFiltering;

OCL_TEST_P(meanShiftFiltering, Mat)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        meanShiftFiltering_(src_roi, dst_roi, sp, sr, crit);
        ocl::meanShiftFiltering(gsrc_roi, gdst_roi, sp, sr, crit);

        Near();
    }
}

///////////////////////////meanShiftProc//////////////////////////////////

typedef meanShiftTestBase meanShiftProc;

OCL_TEST_P(meanShiftProc, Mat)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        meanShiftProc_(src_roi, dst_roi, dstCoor_roi, sp, sr, crit);
        ocl::meanShiftProc(gsrc_roi, gdst_roi, gdstCoor_roi, sp, sr, crit);

        Near();
        Near1();
    }
}

/////////////////////////////////////////////////////////////////////////////////////

INSTANTIATE_TEST_CASE_P(Imgproc, meanShiftFiltering, Combine(
                            Values((MatType)CV_8UC4),
                            Values((MatType)CV_16SC2),
                            Values(5),
                            Values(6),
                            Values(TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 5, 1)),
                            Bool()
                        ));

INSTANTIATE_TEST_CASE_P(Imgproc, meanShiftProc, Combine(
                            Values((MatType)CV_8UC4),
                            Values((MatType)CV_16SC2),
                            Values(5),
                            Values(6),
                            Values(TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 5, 1)),
                            Bool()
                        ));

#endif // HAVE_OPENCL
