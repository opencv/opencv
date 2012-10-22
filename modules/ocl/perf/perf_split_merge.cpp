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
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Jia Haipeng, jiahaipeng95@gmail.com
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

#include "precomp.hpp"

#ifdef HAVE_OPENCL

using namespace cvtest;
using namespace testing;
using namespace std;
using namespace cv::ocl;
PARAM_TEST_CASE(MergeTestBase, MatType, int)
{
    int type;
    int channels;

    //src mat
    cv::Mat mat1;
    cv::Mat mat2;
    cv::Mat mat3;
    cv::Mat mat4;

    //dst mat
    cv::Mat dst;

    // set up roi
    int roicols;
    int roirows;
    int src1x;
    int src1y;
    int src2x;
    int src2y;
    int src3x;
    int src3y;
    int src4x;
    int src4y;
    int dstx;
    int dsty;

    //src mat with roi
    cv::Mat mat1_roi;
    cv::Mat mat2_roi;
    cv::Mat mat3_roi;
    cv::Mat mat4_roi;

    //dst mat with roi
    cv::Mat dst_roi;
    //std::vector<cv::ocl::Info> oclinfo;
    //ocl dst mat for testing
    cv::ocl::oclMat gdst_whole;

    //ocl mat with roi
    cv::ocl::oclMat gmat1;
    cv::ocl::oclMat gmat2;
    cv::ocl::oclMat gmat3;
    cv::ocl::oclMat gmat4;
    cv::ocl::oclMat gdst;

    virtual void SetUp()
    {
        type = GET_PARAM(0);
        channels = GET_PARAM(1);

        cv::RNG &rng = TS::ptr()->get_rng();
        cv::Size size(MWIDTH, MHEIGHT);

        mat1 = randomMat(rng, size, CV_MAKETYPE(type, 1), 5, 16, false);
        mat2 = randomMat(rng, size, CV_MAKETYPE(type, 1), 5, 16, false);
        mat3 = randomMat(rng, size, CV_MAKETYPE(type, 1), 5, 16, false);
        mat4 = randomMat(rng, size, CV_MAKETYPE(type, 1), 5, 16, false);
        dst  = randomMat(rng, size, CV_MAKETYPE(type, channels), 5, 16, false);
        //int devnums = getDevice(oclinfo);
        //CV_Assert(devnums > 0);
        ////if you want to use undefault device, set it here
        ////setDevice(oclinfo[0]);
        //setBinpath(CLBINPATH);
    }
    void Has_roi(int b)
    {
        //cv::RNG& rng = TS::ptr()->get_rng();
        if(b)
        {
            //randomize ROI
            roicols =  mat1.cols - 1; //start
            roirows = mat1.rows - 1;
            src1x   = 1;
            src1y   = 1;
            src2x   = 1;
            src2y   = 1;
            src3x   = 1;
            src3y   = 1;
            src4x   = 1;
            src4y   = 1;
            dstx    = 1;
            dsty    = 1;

        }
        else
        {
            roicols = mat1.cols;
            roirows = mat1.rows;
            src1x   = 0;
            src1y   = 0;
            src2x   = 0;
            src2y   = 0;
            src3x   = 0;
            src3y   = 0;
            src4x   = 0;
            src4y   = 0;
            dstx    = 0;
            dsty    = 0;
        };

        mat1_roi = mat1(Rect(src1x, src1y, roicols, roirows));
        mat2_roi = mat2(Rect(src2x, src2y, roicols, roirows));
        mat3_roi = mat3(Rect(src3x, src3y, roicols, roirows));
        mat4_roi = mat4(Rect(src4x, src4y, roicols, roirows));


        dst_roi = dst(Rect(dstx, dsty, roicols, roirows));
    }

};

struct Merge : MergeTestBase {};

TEST_P(Merge, Accuracy)
{
#ifndef PRINT_KERNEL_RUN_TIME
    double totalcputick = 0;
    double totalgputick = 0;
    double totalgputick_kernel = 0;
    double t0 = 0;
    double t1 = 0;
    double t2 = 0;
    for(int k = LOOPROISTART; k < LOOPROIEND; k++)
    {
        totalcputick = 0;
        totalgputick = 0;
        totalgputick_kernel = 0;
        for(int j = 0; j < LOOP_TIMES + 1; j ++)
        {
            Has_roi(k);
            std::vector<cv::Mat> dev_src;
            dev_src.push_back(mat1_roi);
            dev_src.push_back(mat2_roi);
            dev_src.push_back(mat3_roi);
            dev_src.push_back(mat4_roi);
            t0 = (double)cvGetTickCount();//cpu start
            cv::merge(dev_src, dst_roi);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1	]
            gmat1 = mat1_roi;
            gmat2 = mat2_roi;
            gmat3 = mat3_roi;
            gmat4 = mat4_roi;
            gdst_whole = dst;
            gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));
            std::vector<cv::ocl::oclMat> dev_gsrc;
            dev_gsrc.push_back(gmat1);
            dev_gsrc.push_back(gmat2);
            dev_gsrc.push_back(gmat3);
            dev_gsrc.push_back(gmat4);
            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::merge(dev_gsrc, gdst);
            t2 = (double)cvGetTickCount() - t2;//kernel
            cv::Mat cpu_dst;
            gdst_whole.download (cpu_dst);//download
            t1 = (double)cvGetTickCount() - t1;//gpu end1

            if(j == 0)
                continue;

            totalgputick = t1 + totalgputick;
            totalcputick = t0 + totalcputick;
            totalgputick_kernel = t2 + totalgputick_kernel;

        }
        if(k == 0)
        {
            cout << "no roi\n";
        }
        else
        {
            cout << "with roi\n";
        };
        cout << "average cpu runtime is  " << totalcputick / ((double)cvGetTickFrequency()* LOOP_TIMES * 1000.) << "ms" << endl;
        cout << "average gpu runtime is  " << totalgputick / ((double)cvGetTickFrequency()* LOOP_TIMES * 1000.) << "ms" << endl;
        cout << "average gpu runtime without data transfer is  " << totalgputick_kernel / ((double)cvGetTickFrequency()* LOOP_TIMES * 1000.) << "ms" << endl;
    }
#else
    for(int j = LOOPROISTART; j < LOOPROIEND; j ++)
    {
        Has_roi(j);
        gmat1 = mat1_roi;
        gmat2 = mat2_roi;
        gmat3 = mat3_roi;
        gmat4 = mat4_roi;
        gdst_whole = dst;
        gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));
        std::vector<cv::ocl::oclMat> dev_gsrc;
        dev_gsrc.push_back(gmat1);
        dev_gsrc.push_back(gmat2);
        dev_gsrc.push_back(gmat3);
        dev_gsrc.push_back(gmat4);

        if(j == 0)
        {
            cout << "no roi:";
        }
        else
        {
            cout << "\nwith roi:";
        };
        cv::ocl::merge(dev_gsrc, gdst);
    };
#endif
}


PARAM_TEST_CASE(SplitTestBase, MatType, int)
{
    int type;
    int channels;

    //src mat
    cv::Mat mat;

    //dstmat
    cv::Mat dst1;
    cv::Mat dst2;
    cv::Mat dst3;
    cv::Mat dst4;

    // set up roi
    int roicols;
    int roirows;
    int srcx;
    int srcy;
    int dst1x;
    int dst1y;
    int dst2x;
    int dst2y;
    int dst3x;
    int dst3y;
    int dst4x;
    int dst4y;

    //src mat with roi
    cv::Mat mat_roi;

    //dst mat with roi
    cv::Mat dst1_roi;
    cv::Mat dst2_roi;
    cv::Mat dst3_roi;
    cv::Mat dst4_roi;
    //std::vector<cv::ocl::Info> oclinfo;
    //ocl dst mat for testing
    cv::ocl::oclMat gdst1_whole;
    cv::ocl::oclMat gdst2_whole;
    cv::ocl::oclMat gdst3_whole;
    cv::ocl::oclMat gdst4_whole;

    //ocl mat with roi
    cv::ocl::oclMat gmat;
    cv::ocl::oclMat gdst1;
    cv::ocl::oclMat gdst2;
    cv::ocl::oclMat gdst3;
    cv::ocl::oclMat gdst4;

    virtual void SetUp()
    {
        type = GET_PARAM(0);
        channels = GET_PARAM(1);

        cv::RNG &rng = TS::ptr()->get_rng();
        cv::Size size(MWIDTH, MHEIGHT);

        mat  = randomMat(rng, size, CV_MAKETYPE(type, channels), 5, 16, false);
        dst1 = randomMat(rng, size, CV_MAKETYPE(type, 1), 5, 16, false);
        dst2 = randomMat(rng, size, CV_MAKETYPE(type, 1), 5, 16, false);
        dst3 = randomMat(rng, size, CV_MAKETYPE(type, 1), 5, 16, false);
        dst4 = randomMat(rng, size, CV_MAKETYPE(type, 1), 5, 16, false);
        //int devnums = getDevice(oclinfo);
        //CV_Assert(devnums > 0);
        ////if you want to use undefault device, set it here
        ////setDevice(oclinfo[0]);
        //setBinpath(CLBINPATH);
    }

    void Has_roi(int b)
    {
        //cv::RNG& rng = TS::ptr()->get_rng();
        if(b)
        {
            //randomize ROI
            roicols =  mat.cols - 1; //start
            roirows = mat.rows - 1;
            srcx   = 1;
            srcx   = 1;
            dst1x    = 1;
            dst1y    = 1;
            dst2x    = 1;
            dst2y    = 1;
            dst3x    = 1;
            dst3y    = 1;
            dst4x    = 1;
            dst4y    = 1;
        }
        else
        {
            roicols = mat.cols;
            roirows = mat.rows;
            srcx = 0;
            srcy = 0;
            dst1x = 0;
            dst1y = 0;
            dst2x    = 0;
            dst2y    = 0;
            dst3x    = 0;
            dst3y    = 0;
            dst4x    = 0;
            dst4y    = 0;
        };

        mat_roi = mat(Rect(srcx, srcy, roicols, roirows));

        dst1_roi = dst1(Rect(dst1x, dst1y, roicols, roirows));
        dst2_roi = dst2(Rect(dst2x, dst2y, roicols, roirows));
        dst3_roi = dst3(Rect(dst3x, dst3y, roicols, roirows));
        dst4_roi = dst4(Rect(dst4x, dst4y, roicols, roirows));
    }

};

struct Split : SplitTestBase {};

TEST_P(Split, Accuracy)
{
#ifndef PRINT_KERNEL_RUN_TIME
    double totalcputick = 0;
    double totalgputick = 0;
    double totalgputick_kernel = 0;
    double t0 = 0;
    double t1 = 0;
    double t2 = 0;
    for(int k = LOOPROISTART; k < LOOPROIEND; k++)
    {
        totalcputick = 0;
        totalgputick = 0;
        totalgputick_kernel = 0;
        for(int j = 0; j < LOOP_TIMES + 1; j ++)
        {
            Has_roi(k);
            cv::Mat         dev_dst[4]  = {dst1_roi, dst2_roi, dst3_roi, dst4_roi};
            cv::ocl::oclMat dev_gdst[4] = {gdst1, gdst2, gdst3, gdst4};
            t0 = (double)cvGetTickCount();//cpu start
            cv::split(mat_roi, dev_dst);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            gdst1_whole = dst1;
            gdst1 = gdst1_whole(Rect(dst1x, dst1y, roicols, roirows));

            gdst2_whole = dst2;
            gdst2 = gdst2_whole(Rect(dst2x, dst2y, roicols, roirows));

            gdst3_whole = dst3;
            gdst3 = gdst3_whole(Rect(dst3x, dst3y, roicols, roirows));

            gdst4_whole = dst4;
            gdst4 = gdst4_whole(Rect(dst4x, dst4y, roicols, roirows));

            gmat = mat_roi;
            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::split(gmat, dev_gdst);
            t2 = (double)cvGetTickCount() - t2;//kernel
            cv::Mat cpu_dst1;
            cv::Mat cpu_dst2;
            cv::Mat cpu_dst3;
            cv::Mat cpu_dst4;
            gdst1_whole.download(cpu_dst1);
            gdst2_whole.download(cpu_dst2);
            gdst3_whole.download(cpu_dst3);
            gdst4_whole.download(cpu_dst4);
            t1 = (double)cvGetTickCount() - t1;//gpu end1
            if(j == 0)
                continue;
            totalgputick = t1 + totalgputick;
            totalcputick = t0 + totalcputick;
            totalgputick_kernel = t2 + totalgputick_kernel;

        }
        if(k == 0)
        {
            cout << "no roi\n";
        }
        else
        {
            cout << "with roi\n";
        };
        cout << "average cpu runtime is  " << totalcputick / ((double)cvGetTickFrequency()* LOOP_TIMES * 1000.) << "ms" << endl;
        cout << "average gpu runtime is  " << totalgputick / ((double)cvGetTickFrequency()* LOOP_TIMES * 1000.) << "ms" << endl;
        cout << "average gpu runtime without data transfer is  " << totalgputick_kernel / ((double)cvGetTickFrequency()* LOOP_TIMES * 1000.) << "ms" << endl;
    }
#else
    for(int j = LOOPROISTART; j < LOOPROIEND; j ++)
    {
        Has_roi(j);
        //cv::Mat         dev_dst[4]  = {dst1_roi, dst2_roi, dst3_roi, dst4_roi};
        cv::ocl::oclMat dev_gdst[4] = {gdst1, gdst2, gdst3, gdst4};
        gdst1_whole = dst1;
        gdst1 = gdst1_whole(Rect(dst1x, dst1y, roicols, roirows));

        gdst2_whole = dst2;
        gdst2 = gdst2_whole(Rect(dst2x, dst2y, roicols, roirows));

        gdst3_whole = dst3;
        gdst3 = gdst3_whole(Rect(dst3x, dst3y, roicols, roirows));

        gdst4_whole = dst4;
        gdst4 = gdst4_whole(Rect(dst4x, dst4y, roicols, roirows));
        gmat = mat_roi;
        if(j == 0)
        {
            cout << "no roi:";
        }
        else
        {
            cout << "\nwith roi:";
        };
        cv::ocl::split(gmat, dev_gdst);
    };
#endif
}

//*************test*****************
INSTANTIATE_TEST_CASE_P(SplitMerge, Merge, Combine(
                            Values(CV_8UC4, CV_32FC4), Values(1, 4)));

INSTANTIATE_TEST_CASE_P(SplitMerge, Split , Combine(
                            Values(CV_8U, CV_32S, CV_32F), Values(1, 4)));

#endif // HAVE_OPENCL
