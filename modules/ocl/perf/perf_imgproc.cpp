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
//    Niko Li, newlife20080214@gmail.com
//    Jia Haipeng, jiahaipeng95@gmail.com
//    Shengen Yan, yanshengen@gmail.com
//    Jiang Liyuan, lyuan001.good@163.com
//    Rock Li, Rock.Li@amd.com
//    Zailong Wu, bullet@yeah.net
//    Xu Pang, pangxu010@163.com
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


MatType nulltype = -1;

#define ONE_TYPE(type)  testing::ValuesIn(typeVector(type))
#define NULL_TYPE  testing::ValuesIn(typeVector(nulltype))


vector<MatType> typeVector(MatType type)
{
    vector<MatType> v;
    v.push_back(type);
    return v;
}


PARAM_TEST_CASE(ImgprocTestBase, MatType, MatType, MatType, MatType, MatType, bool)
{
    int type1, type2, type3, type4, type5;
    cv::Scalar val;
    // set up roi
    int roicols;
    int roirows;
    int src1x;
    int src1y;
    int src2x;
    int src2y;
    int dstx;
    int dsty;
    int dst1x;
    int dst1y;
    int maskx;
    int masky;

    //mat
    cv::Mat mat1;
    cv::Mat mat2;
    cv::Mat mask;
    cv::Mat dst;
    cv::Mat dst1; //bak, for two outputs

    //mat with roi
    cv::Mat mat1_roi;
    cv::Mat mat2_roi;
    cv::Mat mask_roi;
    cv::Mat dst_roi;
    cv::Mat dst1_roi; //bak
    //std::vector<cv::ocl::Info> oclinfo;
    //ocl mat
    cv::ocl::oclMat clmat1;
    cv::ocl::oclMat clmat2;
    cv::ocl::oclMat clmask;
    cv::ocl::oclMat cldst;
    cv::ocl::oclMat cldst1; //bak

    //ocl mat with roi
    cv::ocl::oclMat clmat1_roi;
    cv::ocl::oclMat clmat2_roi;
    cv::ocl::oclMat clmask_roi;
    cv::ocl::oclMat cldst_roi;
    cv::ocl::oclMat cldst1_roi;

    virtual void SetUp()
    {
        type1 = GET_PARAM(0);
        type2 = GET_PARAM(1);
        type3 = GET_PARAM(2);
        type4 = GET_PARAM(3);
        type5 = GET_PARAM(4);
        cv::RNG &rng = TS::ptr()->get_rng();
        cv::Size size(MWIDTH, MHEIGHT);
        double min = 1, max = 20;
        //int devnums = getDevice(oclinfo);
        //CV_Assert(devnums>0);
        ////if you want to use undefault device, set it here
        ////setDevice(oclinfo[0]);
        //cv::ocl::setBinpath(CLBINPATH);
        if(type1 != nulltype)
        {
            mat1 = randomMat(rng, size, type1, min, max, false);
            clmat1 = mat1;
        }
        if(type2 != nulltype)
        {
            mat2 = randomMat(rng, size, type2, min, max, false);
            clmat2 = mat2;
        }
        if(type3 != nulltype)
        {
            dst  = randomMat(rng, size, type3, min, max, false);
            cldst = dst;
        }
        if(type4 != nulltype)
        {
            dst1 = randomMat(rng, size, type4, min, max, false);
            cldst1 = dst1;
        }
        if(type5 != nulltype)
        {
            mask = randomMat(rng, size, CV_8UC1, 0, 2,  false);
            cv::threshold(mask, mask, 0.5, 255., type5);
            clmask = mask;
        }
        val = cv::Scalar(rng.uniform(-10.0, 10.0), rng.uniform(-10.0, 10.0), rng.uniform(-10.0, 10.0), rng.uniform(-10.0, 10.0));
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
            src2x   = 1;
            src1y   = 1;
            src2y   = 1;
            dstx    = 1;
            dsty    = 1;
            dst1x    = 1;
            dst1y    = 1;
            maskx	 = 1;
            masky	= 1;
        }
        else
        {
            roicols = mat1.cols;
            roirows = mat1.rows;
            src1x = 0;
            src2x = 0;
            src1y = 0;
            src2y = 0;
            dstx = 0;
            dsty = 0;
            dst1x  = 0;
            dst1y  = 0;
            maskx	 = 0;
            masky	= 0;
        };

        if(type1 != nulltype)
        {
            mat1_roi = mat1(Rect(src1x, src1y, roicols, roirows));
            //clmat1_roi = clmat1(Rect(src1x,src1y,roicols,roirows));
        }
        if(type2 != nulltype)
        {
            mat2_roi = mat2(Rect(src2x, src2y, roicols, roirows));
            //clmat2_roi = clmat2(Rect(src2x,src2y,roicols,roirows));
        }
        if(type3 != nulltype)
        {
            dst_roi  = dst(Rect(dstx, dsty, roicols, roirows));
            //cldst_roi = cldst(Rect(dstx,dsty,roicols,roirows));
        }
        if(type4 != nulltype)
        {
            dst1_roi = dst1(Rect(dst1x, dst1y, roicols, roirows));
            //cldst1_roi = cldst1(Rect(dst1x,dst1y,roicols,roirows));
        }
        if(type5 != nulltype)
        {
            mask_roi = mask(Rect(maskx, masky, roicols, roirows));
            //clmask_roi = clmask(Rect(maskx,masky,roicols,roirows));
        }
    }

    void random_roi()
    {
        cv::RNG &rng = TS::ptr()->get_rng();

        //randomize ROI
        roicols = rng.uniform(1, mat1.cols);
        roirows = rng.uniform(1, mat1.rows);
        src1x   = rng.uniform(0, mat1.cols - roicols);
        src1y   = rng.uniform(0, mat1.rows - roirows);
        src2x   = rng.uniform(0, mat2.cols - roicols);
        src2y   = rng.uniform(0, mat2.rows - roirows);
        dstx    = rng.uniform(0, dst.cols  - roicols);
        dsty    = rng.uniform(0, dst.rows  - roirows);
        dst1x    = rng.uniform(0, dst1.cols  - roicols);
        dst1y    = rng.uniform(0, dst1.rows  - roirows);
        maskx   = rng.uniform(0, mask.cols - roicols);
        masky   = rng.uniform(0, mask.rows - roirows);

        if(type1 != nulltype)
        {
            mat1_roi = mat1(Rect(src1x, src1y, roicols, roirows));
            //clmat1_roi = clmat1(Rect(src1x,src1y,roicols,roirows));
        }
        if(type2 != nulltype)
        {
            mat2_roi = mat2(Rect(src2x, src2y, roicols, roirows));
            //clmat2_roi = clmat2(Rect(src2x,src2y,roicols,roirows));
        }
        if(type3 != nulltype)
        {
            dst_roi  = dst(Rect(dstx, dsty, roicols, roirows));
            //cldst_roi = cldst(Rect(dstx,dsty,roicols,roirows));
        }
        if(type4 != nulltype)
        {
            dst1_roi = dst1(Rect(dst1x, dst1y, roicols, roirows));
            //cldst1_roi = cldst1(Rect(dst1x,dst1y,roicols,roirows));
        }
        if(type5 != nulltype)
        {
            mask_roi = mask(Rect(maskx, masky, roicols, roirows));
            //clmask_roi = clmask(Rect(maskx,masky,roicols,roirows));
        }
    }
};
////////////////////////////////equalizeHist//////////////////////////////////////////

struct equalizeHist : ImgprocTestBase {};

TEST_P(equalizeHist, MatType)
{
    if (mat1.type() != CV_8UC1 || mat1.type() != dst.type())
    {
        cout << "Unsupported type" << endl;
        EXPECT_DOUBLE_EQ(0.0, 0.0);
    }
    else
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

                t0 = (double)cvGetTickCount();//cpu start
                cv::equalizeHist(mat1_roi, dst_roi);
                t0 = (double)cvGetTickCount() - t0;//cpu end

                t1 = (double)cvGetTickCount();//gpu start1
                if(type1 != nulltype)
                {
                    clmat1_roi = clmat1(Rect(src1x, src1y, roicols, roirows));
                }
                cldst_roi = cldst(Rect(dstx, dsty, roicols, roirows));
                t2 = (double)cvGetTickCount(); //kernel
                cv::ocl::equalizeHist(clmat1_roi, cldst_roi);
                t2 = (double)cvGetTickCount() - t2;//kernel
                cv::Mat cpu_cldst;
                //cldst.download(cpu_cldst);//download
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
            if(type1 != nulltype)
            {
                clmat1_roi = clmat1(Rect(src1x, src1y, roicols, roirows));
            }
            if(j == 0)
            {
                cout << "no roi:";
            }
            else
            {
                cout << "\nwith roi:";
            };
            cv::ocl::equalizeHist(clmat1_roi, cldst_roi);
        };
#endif
    }
}


////////////////////////////////bilateralFilter////////////////////////////////////////////

struct bilateralFilter : ImgprocTestBase {};

TEST_P(bilateralFilter, Mat)
{
    double sigmacolor = 50.0;
    int radius = 9;
    int d = 2 * radius + 1;
    double sigmaspace = 20.0;
    int bordertype[] = {cv::BORDER_CONSTANT, cv::BORDER_REPLICATE/*,cv::BORDER_REFLECT,cv::BORDER_WRAP,cv::BORDER_REFLECT_101*/};
    const char *borderstr[] = {"BORDER_CONSTANT", "BORDER_REPLICATE"/*, "BORDER_REFLECT","BORDER_WRAP","BORDER_REFLECT_101"*/};

    if (mat1.depth() != CV_8U || mat1.type() != dst.type())
    {
        cout << "Unsupported type" << endl;
        EXPECT_DOUBLE_EQ(0.0, 0.0);
    }
    else
    {
        for(size_t i = 0; i < sizeof(bordertype) / sizeof(int); i++)
        {
            cout << borderstr[i] << endl;
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
                    if(((bordertype[i] != cv::BORDER_CONSTANT) && (bordertype[i] != cv::BORDER_REPLICATE) && (mat1_roi.cols <= radius)) || (mat1_roi.cols <= radius) || (mat1_roi.rows <= radius) || (mat1_roi.rows <= radius))
                    {
                        continue;
                    }
                    t0 = (double)cvGetTickCount();//cpu start
                    cv::bilateralFilter(mat1_roi, dst_roi, d, sigmacolor, sigmaspace, bordertype[i]);
                    t0 = (double)cvGetTickCount() - t0;//cpu end

                    t1 = (double)cvGetTickCount();//gpu start1
                    if(type1 != nulltype)
                    {
                        clmat1_roi = clmat1(Rect(src1x, src1y, roicols, roirows));
                    }
                    t2 = (double)cvGetTickCount(); //kernel
                    cv::ocl::bilateralFilter(clmat1_roi, cldst_roi, d, sigmacolor, sigmaspace, bordertype[i]);
                    t2 = (double)cvGetTickCount() - t2;//kernel
                    cv::Mat cpu_cldst;
                    cldst.download(cpu_cldst);//download
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
                if(type1 != nulltype)
                {
                    clmat1_roi = clmat1(Rect(src1x, src1y, roicols, roirows));
                };
                if(j == 0)
                {
                    cout << "no roi:";
                }
                else
                {
                    cout << "\nwith roi:";
                };
                cv::ocl::bilateralFilter(clmat1_roi, cldst_roi, d, sigmacolor, sigmaspace, bordertype[i]);
            };

#endif
        };

    }
}

////////////////////////////////copyMakeBorder////////////////////////////////////////////

struct CopyMakeBorder : ImgprocTestBase {};

TEST_P(CopyMakeBorder, Mat)
{
    int bordertype[] = {cv::BORDER_CONSTANT, cv::BORDER_REPLICATE, cv::BORDER_REFLECT, cv::BORDER_WRAP, cv::BORDER_REFLECT_101};
    //const char* borderstr[]={"BORDER_CONSTANT", "BORDER_REPLICATE"/*, "BORDER_REFLECT","BORDER_WRAP","BORDER_REFLECT_101"*/};
    int top = 5;
    int bottom = 5;
    int left = 6;
    int right = 6;
    if (mat1.type() != dst.type())
    {
        cout << "Unsupported type" << endl;
        EXPECT_DOUBLE_EQ(0.0, 0.0);
    }
    else
    {
        for(size_t i = 0; i < sizeof(bordertype) / sizeof(int); i++)
        {
#ifndef PRINT_KERNEL_RUN_TIME
            double totalcputick = 0;
            double totalgputick = 0;
            double totalgputick_kernel = 0;
            double t0 = 0;
            double t1 = 0;
            double t2 = 0;
            for(int k = LOOPROISTART; k < 1; k++) //don't support roi perf test
            {
                totalcputick = 0;
                totalgputick = 0;
                totalgputick_kernel = 0;
                for(int j = 0; j < LOOP_TIMES + 1; j ++)
                {
                    Has_roi(k);

                    t0 = (double)cvGetTickCount();//cpu start
                    cv::copyMakeBorder(mat1_roi, dst_roi, top, bottom, left, right, bordertype[i] | cv::BORDER_ISOLATED, cv::Scalar(1.0));
                    t0 = (double)cvGetTickCount() - t0;//cpu end

                    t1 = (double)cvGetTickCount();//gpu start1
                    if(type1 != nulltype)
                    {
                        clmat1_roi = clmat1(Rect(src1x, src1y, roicols, roirows));
                    }
                    t2 = (double)cvGetTickCount(); //kernel
                    cv::ocl::copyMakeBorder(clmat1_roi, cldst_roi, top, bottom, left, right,  bordertype[i] | cv::BORDER_ISOLATED, cv::Scalar(1.0));
                    t2 = (double)cvGetTickCount() - t2;//kernel
                    cv::Mat cpu_cldst;
                    cldst.download(cpu_cldst);//download
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
                if(type1 != nulltype)
                {
                    clmat1_roi = clmat1(Rect(src1x, src1y, roicols, roirows));
                };
                if(j == 0)
                {
                    cout << "no roi:";
                }
                else
                {
                    cout << "\nwith roi:";
                };
                cv::ocl::copyMakeBorder(clmat1_roi, cldst_roi, top, bottom, left, right,  bordertype[i] | cv::BORDER_ISOLATED, cv::Scalar(1.0));
            };
#endif
        };
    }
}

////////////////////////////////cornerMinEigenVal//////////////////////////////////////////

struct cornerMinEigenVal : ImgprocTestBase {};

TEST_P(cornerMinEigenVal, Mat)
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
            int blockSize = 7, apertureSize = 3; //1 + 2 * (rand() % 4);
            int borderType = cv::BORDER_REFLECT;
            t0 = (double)cvGetTickCount();//cpu start
            cv::cornerMinEigenVal(mat1_roi, dst_roi, blockSize, apertureSize, borderType);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            if(type1 != nulltype)
            {
                clmat1_roi = clmat1(Rect(src1x, src1y, roicols, roirows));
            }
            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::cornerMinEigenVal(clmat1_roi, cldst_roi, blockSize, apertureSize, borderType);
            t2 = (double)cvGetTickCount() - t2;//kernel
            cv::Mat cpu_cldst;
            cldst.download(cpu_cldst);//download
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
        int blockSize = 7, apertureSize = 1 + 2 * (rand() % 4);
        int borderType = cv::BORDER_REFLECT;
        if(type1 != nulltype)
        {
            clmat1_roi = clmat1(Rect(src1x, src1y, roicols, roirows));
        };
        if(j == 0)
        {
            cout << "no roi:";
        }
        else
        {
            cout << "\nwith roi:";
        };
        cv::ocl::cornerMinEigenVal(clmat1_roi, cldst_roi, blockSize, apertureSize, borderType);
    };
#endif
}


////////////////////////////////cornerHarris//////////////////////////////////////////

struct cornerHarris : ImgprocTestBase {};

TEST_P(cornerHarris, Mat)
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
            int blockSize = 7, apertureSize = 3;
            int borderType = cv::BORDER_REFLECT;
            double kk = 2;
            t0 = (double)cvGetTickCount();//cpu start
            cv::cornerHarris(mat1_roi, dst_roi, blockSize, apertureSize, kk, borderType);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            if(type1 != nulltype)
            {
                clmat1_roi = clmat1(Rect(src1x, src1y, roicols, roirows));
            }
            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::cornerHarris(clmat1_roi, cldst_roi, blockSize, apertureSize, kk, borderType);
            t2 = (double)cvGetTickCount() - t2;//kernel
            cv::Mat cpu_cldst;
            cldst.download(cpu_cldst);//download
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
        double kk = 2;
        int blockSize = 7, apertureSize = 3;
        int borderType = cv::BORDER_REFLECT;
        if(type1 != nulltype)
        {
            clmat1_roi = clmat1(Rect(src1x, src1y, roicols, roirows));
        };
        if(j == 0)
        {
            cout << "no roi:";
        }
        else
        {
            cout << "\nwith roi:";
        };
        cv::ocl::cornerHarris(clmat1_roi, cldst_roi, blockSize, apertureSize, kk, borderType);
    };
#endif

}


////////////////////////////////integral/////////////////////////////////////////////////

struct integral : ImgprocTestBase {};

TEST_P(integral, Mat)
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
            t0 = (double)cvGetTickCount();//cpu start
            cv::integral(mat1_roi, dst_roi, dst1_roi);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            if(type1 != nulltype)
            {
                clmat1_roi = clmat1(Rect(src1x, src1y, roicols, roirows));
            }
            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::integral(clmat1_roi, cldst_roi, cldst1_roi);
            t2 = (double)cvGetTickCount() - t2;//kernel
            cv::Mat cpu_cldst;
            cv::Mat cpu_cldst1;
            cldst.download(cpu_cldst);//download
            cldst1.download(cpu_cldst1);
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
        if(type1 != nulltype)
        {
            clmat1_roi = clmat1(Rect(src1x, src1y, roicols, roirows));
        };
        if(j == 0)
        {
            cout << "no roi:";
        }
        else
        {
            cout << "\nwith roi:";
        };
        cv::ocl::integral(clmat1_roi, cldst_roi, cldst1_roi);
    };
#endif
}


/////////////////////////////////////////////////////////////////////////////////////////////////
// warpAffine  & warpPerspective

PARAM_TEST_CASE(WarpTestBase, MatType, int)
{
    int type;
    cv::Size size;
    int interpolation;

    //src mat
    cv::Mat mat1;
    cv::Mat dst;

    // set up roi
    int src_roicols;
    int src_roirows;
    int dst_roicols;
    int dst_roirows;
    int src1x;
    int src1y;
    int dstx;
    int dsty;


    //src mat with roi
    cv::Mat mat1_roi;
    cv::Mat dst_roi;
    //std::vector<cv::ocl::Info> oclinfo;
    //ocl dst mat for testing
    cv::ocl::oclMat gdst_whole;

    //ocl mat with roi
    cv::ocl::oclMat gmat1;
    cv::ocl::oclMat gdst;

    virtual void SetUp()
    {
        type = GET_PARAM(0);
        //dsize = GET_PARAM(1);
        interpolation = GET_PARAM(1);

        cv::RNG &rng = TS::ptr()->get_rng();
        size = cv::Size(MWIDTH, MHEIGHT);

        mat1 = randomMat(rng, size, type, 5, 16, false);
        dst  = randomMat(rng, size, type, 5, 16, false);

        //int devnums = getDevice(oclinfo);
        //CV_Assert(devnums > 0);
        ////if you want to use undefault device, set it here
        ////setDevice(oclinfo[0]);
        //cv::ocl::setBinpath(CLBINPATH);
    }
    void Has_roi(int b)
    {
        //cv::RNG& rng = TS::ptr()->get_rng();
        if(b)
        {
            //randomize ROI
            src_roicols =  mat1.cols - 1; //start
            src_roirows = mat1.rows - 1;
            dst_roicols = dst.cols - 1;
            dst_roirows = dst.rows - 1;
            src1x   = 1;
            src1y   = 1;
            dstx    = 1;
            dsty    = 1;

        }
        else
        {
            src_roicols = mat1.cols;
            src_roirows = mat1.rows;
            dst_roicols = dst.cols;
            dst_roirows = dst.rows;
            src1x = 0;
            src1y = 0;
            dstx = 0;
            dsty = 0;

        };
        mat1_roi = mat1(Rect(src1x, src1y, src_roicols, src_roirows));
        dst_roi  = dst(Rect(dstx, dsty, dst_roicols, dst_roirows));


    }

};

/////warpAffine

struct WarpAffine : WarpTestBase {};

TEST_P(WarpAffine, Mat)
{
    static const double coeffs[2][3] =
    {
        {cos(3.14 / 6), -sin(3.14 / 6), 100.0},
        {sin(3.14 / 6), cos(3.14 / 6), -100.0}
    };
    Mat M(2, 3, CV_64F, (void *)coeffs);

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

            t0 = (double)cvGetTickCount();//cpu start
            cv::warpAffine(mat1_roi, dst_roi, M, size, interpolation);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            gdst_whole = dst;
            gdst = gdst_whole(Rect(dstx, dsty, dst_roicols, dst_roirows));

            gmat1 = mat1_roi;
            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::warpAffine(gmat1, gdst, M, size, interpolation);
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
        gdst_whole = dst;
        gdst = gdst_whole(Rect(dstx, dsty, dst_roicols, dst_roirows));
        gmat1 = mat1_roi;
        if(j == 0)
        {
            cout << "no roi:";
        }
        else
        {
            cout << "\nwith roi:";
        };
        cv::ocl::warpAffine(gmat1, gdst, M, size, interpolation);
    };
#endif

}


// warpPerspective

struct WarpPerspective : WarpTestBase {};

TEST_P(WarpPerspective, Mat)
{
    static const double coeffs[3][3] =
    {
        {cos(3.14 / 6), -sin(3.14 / 6), 100.0},
        {sin(3.14 / 6), cos(3.14 / 6), -100.0},
        {0.0, 0.0, 1.0}
    };
    Mat M(3, 3, CV_64F, (void *)coeffs);

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

            t0 = (double)cvGetTickCount();//cpu start
            cv::warpPerspective(mat1_roi, dst_roi, M, size, interpolation);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            gdst_whole = dst;
            gdst = gdst_whole(Rect(dstx, dsty, dst_roicols, dst_roirows));

            gmat1 = mat1_roi;
            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::warpPerspective(gmat1, gdst, M, size, interpolation);
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
        gdst_whole = dst;
        gdst = gdst_whole(Rect(dstx, dsty, dst_roicols, dst_roirows));
        gmat1 = mat1_roi;
        if(j == 0)
        {
            cout << "no roi:";
        }
        else
        {
            cout << "\nwith roi:";
        };
        cv::ocl::warpPerspective(gmat1, gdst, M, size, interpolation);
    };
#endif

}

/////////////////////////////////////////////////////////////////////////////////////////////////
// remap
//////////////////////////////////////////////////////////////////////////////////////////////////

PARAM_TEST_CASE(Remap, MatType, MatType, MatType, int, int)
{
    int srcType;
    int map1Type;
    int map2Type;
    cv::Scalar val;

    int interpolation;
    int bordertype;

    cv::Mat src;
    cv::Mat dst;
    cv::Mat map1;
    cv::Mat map2;


    int src_roicols;
    int src_roirows;
    int dst_roicols;
    int dst_roirows;
    int map1_roicols;
    int map1_roirows;
    int map2_roicols;
    int map2_roirows;
    int srcx;
    int srcy;
    int dstx;
    int dsty;
    int map1x;
    int map1y;
    int map2x;
    int map2y;

    cv::Mat src_roi;
    cv::Mat dst_roi;
    cv::Mat map1_roi;
    cv::Mat map2_roi;

    //ocl mat for testing
    cv::ocl::oclMat gdst;

    //ocl mat with roi
    cv::ocl::oclMat gsrc_roi;
    cv::ocl::oclMat gdst_roi;
    cv::ocl::oclMat gmap1_roi;
    cv::ocl::oclMat gmap2_roi;

    virtual void SetUp()
    {
        srcType = GET_PARAM(0);
        map1Type = GET_PARAM(1);
        map2Type = GET_PARAM(2);
        interpolation = GET_PARAM(3);
        bordertype = GET_PARAM(4);

        cv::RNG &rng = TS::ptr()->get_rng();
        cv::Size srcSize = cv::Size(MWIDTH, MHEIGHT);
        cv::Size map1Size = cv::Size(MWIDTH, MHEIGHT);
        double min = 5, max = 16;

        if(srcType != nulltype)
        {
            src = randomMat(rng, srcSize, srcType, min, max, false);
        }
        if((map1Type == CV_16SC2 && map2Type == nulltype) || (map1Type == CV_32FC2 && map2Type == nulltype))
        {
            map1 = randomMat(rng, map1Size, map1Type, min, max, false);

        }
        else if (map1Type == CV_32FC1 && map2Type == CV_32FC1)
        {
            map1 = randomMat(rng, map1Size, map1Type, min, max, false);
            map2 = randomMat(rng, map1Size, map1Type, min, max, false);
        }

        else
            cout << "The wrong input type" << endl;

        dst = randomMat(rng, map1Size, srcType, min, max, false);
        switch (src.channels())
        {
        case 1:
            val = cv::Scalar(rng.uniform(0.0, 10.0), 0, 0, 0);
            break;
        case 2:
            val = cv::Scalar(rng.uniform(0.0, 10.0), rng.uniform(0.0, 10.0), 0, 0);
            break;
        case 3:
            val = cv::Scalar(rng.uniform(0.0, 10.0), rng.uniform(0.0, 10.0), rng.uniform(0.0, 10.0), 0);
            break;
        case 4:
            val = cv::Scalar(rng.uniform(0.0, 10.0), rng.uniform(0.0, 10.0), rng.uniform(0.0, 10.0), rng.uniform(0.0, 10.0));
            break;
        }

        //int devnums = getDevice(oclinfo);
        //CV_Assert(devnums > 0);
        //if you want to use undefault device, set it here
        //setDevice(oclinfo[0]);
        //cv::ocl::setBinpath(CLBINPATH);
    }
    void Has_roi(int b)
    {
        if(b)
        {
            //randomize ROI
            dst_roicols = dst.cols - 1;
            dst_roirows = dst.rows - 1;

            src_roicols = src.cols - 1;
            src_roirows = src.rows - 1;


            srcx = 1;
            srcy = 1;
            dstx = 1;
            dsty = 1;
        }
        else
        {
            dst_roicols = dst.cols;
            dst_roirows = dst.rows;

            src_roicols = src.cols;
            src_roirows = src.rows;


            srcx = 0;
            srcy = 0;
            dstx = 0;
            dsty = 0;
        }
        map1_roicols = dst_roicols;
        map1_roirows = dst_roirows;
        map2_roicols = dst_roicols;
        map2_roirows = dst_roirows;
        map1x = dstx;
        map1y = dsty;
        map2x = dstx;
        map2y = dsty;

        if((map1Type == CV_16SC2 && map2Type == nulltype) || (map1Type == CV_32FC2 && map2Type == nulltype))
        {
            map1_roi = map1(Rect(map1x, map1y, map1_roicols, map1_roirows));
            gmap1_roi = map1_roi;
        }

        else if (map1Type == CV_32FC1 && map2Type == CV_32FC1)
        {
            map1_roi = map1(Rect(map1x, map1y, map1_roicols, map1_roirows));
            map2_roi = map2(Rect(map2x, map2y, map2_roicols, map2_roirows));
            gmap1_roi = map1_roi;
            gmap2_roi = map2_roi;
        }
        dst_roi = dst(Rect(dstx, dsty, dst_roicols, dst_roirows));
        src_roi = dst(Rect(srcx, srcy, src_roicols, src_roirows));

    }
};

TEST_P(Remap, Mat)
{
    if((interpolation == 1 && map1Type == CV_16SC2) || (map1Type == CV_32FC1 && map2Type == nulltype) || (map1Type == CV_16SC2 && map2Type == CV_32FC1) || (map1Type == CV_32FC2 && map2Type == CV_32FC1))
    {
        cout << "LINEAR don't support the map1Type and map2Type" << endl;
        return;
    }
    int bordertype[] = {cv::BORDER_CONSTANT, cv::BORDER_REPLICATE/*,BORDER_REFLECT,BORDER_WRAP,BORDER_REFLECT_101*/};
    const char *borderstr[] = {"BORDER_CONSTANT", "BORDER_REPLICATE"/*, "BORDER_REFLECT","BORDER_WRAP","BORDER_REFLECT_101"*/};
    cout << borderstr[0] << endl;
#ifndef PRINT_KERNEL_RUN_TIME
    double totalcputick = 0;
    double totalgputick = 0;
    double totalgputick_kernel = 0;
    double t0 = 0;
    double t1 = 0;
    double t2 = 0;
    for(int k = 0; k < 2; k++)
    {
        totalcputick = 0;
        totalgputick = 0;
        totalgputick_kernel = 0;
        for(int j = 0; j < LOOP_TIMES + 1; j++)
        {
            Has_roi(k);

            t0 = (double)cvGetTickCount();//cpu start
            cv::remap(src_roi, dst_roi, map1_roi, map2_roi, interpolation, bordertype[0], val);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start
            gsrc_roi = src_roi;
            gdst = dst;
            gdst_roi = gdst(Rect(dstx, dsty, dst_roicols, dst_roirows));

            t2 = (double)cvGetTickCount();//kernel
            cv::ocl::remap(gsrc_roi, gdst_roi, gmap1_roi, gmap2_roi, interpolation, bordertype[0], val);
            t2 = (double)cvGetTickCount() - t2;//kernel

            cv::Mat cpu_dst;
            gdst.download(cpu_dst);

            t1 = (double)cvGetTickCount() - t1;//gpu end

            if (j == 0)
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
    for(int j = 0; j < 2; j ++)
    {
        Has_roi(j);
        gdst = dst;
        gdst_roi = gdst(Rect(dstx, dsty, dst_roicols, dst_roirows));
        gsrc_roi = src_roi;
        if(j == 0)
        {
            cout << "no roi:";
        }
        else
        {
            cout << "\nwith roi:";
        };
        cv::ocl::remap(gsrc_roi, gdst_roi, gmap1_roi, gmap2_roi, interpolation, bordertype[0], val);
    };
#endif

}


/////////////////////////////////////////////////////////////////////////////////////////////////
// resize

PARAM_TEST_CASE(Resize, MatType, cv::Size, double, double, int)
{
    int type;
    cv::Size dsize;
    double fx, fy;
    int interpolation;

    //src mat
    cv::Mat mat1;
    cv::Mat dst;

    // set up roi
    int src_roicols;
    int src_roirows;
    int dst_roicols;
    int dst_roirows;
    int src1x;
    int src1y;
    int dstx;
    int dsty;


    //src mat with roi
    cv::Mat mat1_roi;
    cv::Mat dst_roi;
    //std::vector<cv::ocl::Info> oclinfo;
    //ocl dst mat for testing
    cv::ocl::oclMat gdst_whole;

    //ocl mat with roi
    cv::ocl::oclMat gmat1;
    cv::ocl::oclMat gdst;

    virtual void SetUp()
    {
        type = GET_PARAM(0);
        dsize = GET_PARAM(1);
        fx = GET_PARAM(2);
        fy = GET_PARAM(3);
        interpolation = GET_PARAM(4);

        cv::RNG &rng = TS::ptr()->get_rng();
        cv::Size size(MWIDTH, MHEIGHT);

        if(dsize == cv::Size() && !(fx > 0 && fy > 0))
        {
            cout << "invalid dsize and fx fy" << endl;
            return;
        }

        if(dsize == cv::Size())
        {
            dsize.width = (int)(size.width * fx);
            dsize.height = (int)(size.height * fy);
        }

        mat1 = randomMat(rng, size, type, 5, 16, false);
        dst  = randomMat(rng, dsize, type, 5, 16, false);

        //int devnums = getDevice(oclinfo);
        //CV_Assert(devnums > 0);
        ////if you want to use undefault device, set it here
        ////setDevice(oclinfo[0]);
        //cv::ocl::setBinpath(CLBINPATH);
    }
    void Has_roi(int b)
    {
        //cv::RNG& rng = TS::ptr()->get_rng();
        if(b)
        {
            //randomize ROI
            src_roicols =  mat1.cols - 1; //start
            src_roirows = mat1.rows - 1;
            dst_roicols = dst.cols - 1;
            dst_roirows = dst.rows - 1;
            src1x   = 1;
            src1y   = 1;
            dstx    = 1;
            dsty    = 1;

        }
        else
        {
            src_roicols = mat1.cols;
            src_roirows = mat1.rows;
            dst_roicols = dst.cols;
            dst_roirows = dst.rows;
            src1x = 0;
            src1y = 0;
            dstx = 0;
            dsty = 0;

        };
        mat1_roi = mat1(Rect(src1x, src1y, src_roicols, src_roirows));
        dst_roi  = dst(Rect(dstx, dsty, dst_roicols, dst_roirows));


    }

};

TEST_P(Resize, Mat)
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

            t0 = (double)cvGetTickCount();//cpu start
            cv::resize(mat1_roi, dst_roi, dsize, fx, fy, interpolation);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            gdst_whole = dst;
            gdst = gdst_whole(Rect(dstx, dsty, dst_roicols, dst_roirows));

            gmat1 = mat1_roi;
            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::resize(gmat1, gdst, dsize, fx, fy, interpolation);
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
        gdst_whole = dst;
        gdst = gdst_whole(Rect(dstx, dsty, dst_roicols, dst_roirows));
        gmat1 = mat1_roi;
        if(j == 0)
        {
            cout << "no roi:";
        }
        else
        {
            cout << "\nwith roi:";
        };
        cv::ocl::resize(gmat1, gdst, dsize, fx, fy, interpolation);
    };
#endif

}

/////////////////////////////////////////////////////////////////////////////////////////////////
//threshold

PARAM_TEST_CASE(Threshold, MatType, ThreshOp)
{
    int type;
    int threshOp;

    //src mat
    cv::Mat mat1;
    cv::Mat dst;

    // set up roi
    int roicols;
    int roirows;
    int src1x;
    int src1y;
    int dstx;
    int dsty;

    //src mat with roi
    cv::Mat mat1_roi;
    cv::Mat dst_roi;
    //std::vector<cv::ocl::Info> oclinfo;
    //ocl dst mat for testing
    cv::ocl::oclMat gdst_whole;

    //ocl mat with roi
    cv::ocl::oclMat gmat1;
    cv::ocl::oclMat gdst;

    virtual void SetUp()
    {
        type = GET_PARAM(0);
        threshOp = GET_PARAM(1);

        cv::RNG &rng = TS::ptr()->get_rng();
        cv::Size size(MWIDTH, MHEIGHT);

        mat1 = randomMat(rng, size, type, 5, 16, false);
        dst  = randomMat(rng, size, type, 5, 16, false);

        //int devnums = getDevice(oclinfo);
        //CV_Assert(devnums > 0);
        ////if you want to use undefault device, set it here
        ////setDevice(oclinfo[0]);
        //cv::ocl::setBinpath(CLBINPATH);
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
            dstx    = 1;
            dsty    = 1;

        }
        else
        {
            roicols = mat1.cols;
            roirows = mat1.rows;
            src1x = 0;
            src1y = 0;
            dstx = 0;
            dsty = 0;

        };
        mat1_roi = mat1(Rect(src1x, src1y, roicols, roirows));
        dst_roi  = dst(Rect(dstx, dsty, roicols, roirows));


    }
};

TEST_P(Threshold, Mat)
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

            double maxVal = randomDouble(20.0, 127.0);
            double thresh = randomDouble(0.0, maxVal);
            t0 = (double)cvGetTickCount();//cpu start
            cv::threshold(mat1_roi, dst_roi, thresh, maxVal, threshOp);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1

            gdst_whole = dst;
            gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));
            gmat1 = mat1_roi;
            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::threshold(gmat1, gdst, thresh, maxVal, threshOp);
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
        double maxVal = randomDouble(20.0, 127.0);
        double thresh = randomDouble(0.0, maxVal);
        gdst_whole = dst;
        gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));
        gmat1 = mat1_roi;

        if(j == 0)
        {
            cout << "no roi:";
        }
        else
        {
            cout << "\nwith roi:";
        };
        cv::ocl::threshold(gmat1, gdst, thresh, maxVal, threshOp);
    };
#endif

}
///////////////////////////////////////////////////////////////////////////////////////////////////
//meanShift

PARAM_TEST_CASE(meanShiftTestBase, MatType, MatType, int, int, cv::TermCriteria)
{
    int type, typeCoor;
    int sp, sr;
    cv::TermCriteria crit;
    //src mat
    cv::Mat src;
    cv::Mat dst;
    cv::Mat dstCoor;

    //set up roi
    int roicols;
    int roirows;
    int srcx;
    int srcy;
    int dstx;
    int dsty;

    //src mat with roi
    cv::Mat src_roi;
    cv::Mat dst_roi;
    cv::Mat dstCoor_roi;

    //ocl dst mat
    cv::ocl::oclMat gdst;
    cv::ocl::oclMat gdstCoor;

    //std::vector<cv::ocl::Info> oclinfo;
    //ocl mat with roi
    cv::ocl::oclMat gsrc_roi;
    cv::ocl::oclMat gdst_roi;
    cv::ocl::oclMat gdstCoor_roi;

    virtual void SetUp()
    {
        type     = GET_PARAM(0);
        typeCoor = GET_PARAM(1);
        sp       = GET_PARAM(2);
        sr       = GET_PARAM(3);
        crit     = GET_PARAM(4);

        cv::RNG &rng = TS::ptr()->get_rng();

        // MWIDTH=256, MHEIGHT=256. defined in utility.hpp
        cv::Size size = cv::Size(MWIDTH, MHEIGHT);

        src = randomMat(rng, size, type, 5, 16, false);
        dst = randomMat(rng, size, type, 5, 16, false);
        dstCoor = randomMat(rng, size, typeCoor, 5, 16, false);

        //int devnums = getDevice(oclinfo);
        //CV_Assert(devnums > 0);
        ////if you want to use undefault device, set it here
        ////setDevice(oclinfo[0]);
        //cv::ocl::setBinpath(CLBINPATH);
    }

    void Has_roi(int b)
    {
        if(b)
        {
            //randomize ROI
            roicols = src.cols - 1;
            roirows = src.rows - 1;
            srcx = 1;
            srcy = 1;
            dstx = 1;
            dsty = 1;
        }
        else
        {
            roicols = src.cols;
            roirows = src.rows;
            srcx = 0;
            srcy = 0;
            dstx = 0;
            dsty = 0;
        };

        src_roi = src(Rect(srcx, srcy, roicols, roirows));
        dst_roi = dst(Rect(dstx, dsty, roicols, roirows));
        dstCoor_roi = dstCoor(Rect(dstx, dsty, roicols, roirows));

        gdst = dst;
        gdstCoor = dstCoor;
    }
};

/////////////////////////meanShiftFiltering/////////////////////////////
struct meanShiftFiltering : meanShiftTestBase {};

TEST_P(meanShiftFiltering, Mat)
{

#ifndef PRINT_KERNEL_RUN_TIME
    double t1 = 0;
    double t2 = 0;
    for(int k = 0; k < 2; k++)
    {
        double totalgputick = 0;
        double totalgputick_kernel = 0;
        for(int j = 0; j < LOOP_TIMES + 1; j ++)
        {
            Has_roi(k);

            t1 = (double)cvGetTickCount();//gpu start1

            gsrc_roi = src_roi;
            gdst_roi = gdst(Rect(dstx, dsty, roicols, roirows));  //gdst_roi

            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::meanShiftFiltering(gsrc_roi, gdst_roi, sp, sr, crit);
            t2 = (double)cvGetTickCount() - t2;//kernel

            cv::Mat cpu_gdst;
            gdst.download(cpu_gdst);//download

            t1 = (double)cvGetTickCount() - t1;//gpu end1

            if(j == 0)
                continue;

            totalgputick = t1 + totalgputick;
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
        cout << "average gpu runtime is  " << totalgputick / ((double)cvGetTickFrequency()* LOOP_TIMES * 1000.) << "ms" << endl;
        cout << "average gpu runtime without data transfer is  " << totalgputick_kernel / ((double)cvGetTickFrequency()* LOOP_TIMES * 1000.) << "ms" << endl;
    }
#else
    for(int j = LOOPROISTART; j < LOOPROIEND; j ++)
    {
        Has_roi(j);

        gsrc_roi = src_roi;
        gdst_roi = gdst(Rect(dstx, dsty, roicols, roirows));  //gdst_roi

        if(j == 0)
        {
            cout << "no roi:";
        }
        else
        {
            cout << "\nwith roi:";
        };
        cv::ocl::meanShiftFiltering(gsrc_roi, gdst_roi, sp, sr, crit);
    };
#endif

}

///////////////////////////meanShiftProc//////////////////////////////////
struct meanShiftProc : meanShiftTestBase {};

TEST_P(meanShiftProc, Mat)
{

#ifndef PRINT_KERNEL_RUN_TIME
    double t1 = 0;
    double t2 = 0;
    for(int k = 0; k < 2; k++)
    {
        double totalgputick = 0;
        double totalgputick_kernel = 0;
        for(int j = 0; j < LOOP_TIMES + 1; j ++)
        {
            Has_roi(k);

            t1 = (double)cvGetTickCount();//gpu start1

            gsrc_roi = src_roi;
            gdst_roi = gdst(Rect(dstx, dsty, roicols, roirows));  //gdst_roi
            gdstCoor_roi = gdstCoor(Rect(dstx, dsty, roicols, roirows));

            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::meanShiftProc(gsrc_roi, gdst_roi, gdstCoor_roi, sp, sr, crit);
            t2 = (double)cvGetTickCount() - t2;//kernel

            cv::Mat cpu_gdstCoor;
            gdstCoor.download(cpu_gdstCoor);//download

            t1 = (double)cvGetTickCount() - t1;//gpu end1

            if(j == 0)
                continue;

            totalgputick = t1 + totalgputick;
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
        cout << "average gpu runtime is  " << totalgputick / ((double)cvGetTickFrequency()* LOOP_TIMES * 1000.) << "ms" << endl;
        cout << "average gpu runtime without data transfer is  " << totalgputick_kernel / ((double)cvGetTickFrequency()* LOOP_TIMES * 1000.) << "ms" << endl;
    }
#else
    for(int j = LOOPROISTART; j < LOOPROIEND; j ++)
    {
        Has_roi(j);

        gsrc_roi = src_roi;
        gdst_roi = gdst(Rect(dstx, dsty, roicols, roirows));  //gdst_roi
        gdstCoor_roi = gdstCoor(Rect(dstx, dsty, roicols, roirows));

        if(j == 0)
        {
            cout << "no roi:";
        }
        else
        {
            cout << "\nwith roi:";
        };
        cv::ocl::meanShiftProc(gsrc_roi, gdst_roi, gdstCoor_roi, sp, sr, crit);
    };
#endif

}

///////////////////////////////////////////////////////////////////////////////////////////
//hist

void calcHistGold(const cv::Mat &src, cv::Mat &hist)
{
    hist.create(1, 256, CV_32SC1);
    hist.setTo(cv::Scalar::all(0));

    int *hist_row = hist.ptr<int>();
    for (int y = 0; y < src.rows; ++y)
    {
        const uchar *src_row = src.ptr(y);

        for (int x = 0; x < src.cols; ++x)
            ++hist_row[src_row[x]];
    }
}

PARAM_TEST_CASE(histTestBase, MatType, MatType)
{
    int type_src;

    //src mat
    cv::Mat src;
    cv::Mat dst_hist;
    //set up roi
    int roicols;
    int roirows;
    int srcx;
    int srcy;
    //src mat with roi
    cv::Mat src_roi;
    //ocl dst mat, dst_hist and gdst_hist don't have roi
    cv::ocl::oclMat gdst_hist;

    //ocl mat with roi
    cv::ocl::oclMat gsrc_roi;

    //    std::vector<cv::ocl::Info> oclinfo;

    virtual void SetUp()
    {
        type_src   = GET_PARAM(0);

        cv::RNG &rng = TS::ptr()->get_rng();
        cv::Size size = cv::Size(MWIDTH, MHEIGHT);

        src = randomMat(rng, size, type_src, 0, 256, false);

        //        int devnums = getDevice(oclinfo);
        //        CV_Assert(devnums > 0);
        //if you want to use undefault device, set it here
        //setDevice(oclinfo[0]);
    }

    void Has_roi(int b)
    {
        if(b)
        {
            //randomize ROI
            roicols = src.cols - 1;
            roirows = src.rows - 1;
            srcx = 1;
            srcy = 1;
        }
        else
        {
            roicols = src.cols;
            roirows = src.rows;
            srcx = 0;
            srcy = 0;
        };
        src_roi = src(Rect(srcx, srcy, roicols, roirows));
    }
};

///////////////////////////calcHist///////////////////////////////////////
struct calcHist : histTestBase {};

TEST_P(calcHist, Mat)
{
#ifndef PRINT_KERNEL_RUN_TIME
    double t0 = 0;
    double t1 = 0;
    double t2 = 0;
    for(int k = 0; k < 2; k++)
    {
        double totalcputick = 0;
        double totalgputick = 0;
        double totalgputick_kernel = 0;
        for(int j = 0; j < LOOP_TIMES + 1; j ++)
        {
            Has_roi(k);

            t0 = (double)cvGetTickCount();//cpu start
            calcHistGold(src_roi, dst_hist);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1

            gsrc_roi = src_roi;

            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::calcHist(gsrc_roi, gdst_hist);
            t2 = (double)cvGetTickCount() - t2;//kernel

            cv::Mat cpu_hist;
            gdst_hist.download(cpu_hist);//download

            t1 = (double)cvGetTickCount() - t1;//gpu end1

            if(j == 0)
                continue;

            totalcputick = t0 + totalcputick;
            totalgputick = t1 + totalgputick;
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
    for(int j = 0; j < 2; j ++)
    {
        Has_roi(j);

        gsrc_roi = src_roi;

        if(j == 0)
        {
            cout << "no roi:";
        }
        else
        {
            cout << "\nwith roi:";
        };
        cv::ocl::calcHist(gsrc_roi, gdst_hist);
    };
#endif
}


//************test*******************

INSTANTIATE_TEST_CASE_P(ImgprocTestBase, equalizeHist, Combine(
                            ONE_TYPE(CV_8UC1),
                            NULL_TYPE,
                            ONE_TYPE(CV_8UC1),
                            NULL_TYPE,
                            NULL_TYPE,
                            Values(false))); // Values(false) is the reserved parameter

INSTANTIATE_TEST_CASE_P(ImgprocTestBase, bilateralFilter, Combine(
                            Values(CV_8UC1, CV_8UC3),
                            NULL_TYPE,
                            Values(CV_8UC1, CV_8UC3),
                            NULL_TYPE,
                            NULL_TYPE,
                            Values(false))); // Values(false) is the reserved parameter


INSTANTIATE_TEST_CASE_P(ImgprocTestBase, CopyMakeBorder, Combine(
                            Values(CV_8UC1, CV_8UC4/*, CV_32SC1*/),
                            NULL_TYPE,
                            Values(CV_8UC1, CV_8UC4/*,CV_32SC1*/),
                            NULL_TYPE,
                            NULL_TYPE,
                            Values(false))); // Values(false) is the reserved parameter
INSTANTIATE_TEST_CASE_P(ImgprocTestBase, cornerMinEigenVal, Combine(
                            Values(CV_8UC1, CV_32FC1),
                            NULL_TYPE,
                            ONE_TYPE(CV_32FC1),
                            NULL_TYPE,
                            NULL_TYPE,
                            Values(false))); // Values(false) is the reserved parameter

INSTANTIATE_TEST_CASE_P(ImgprocTestBase, cornerHarris, Combine(
                            Values(CV_8UC1, CV_32FC1),
                            NULL_TYPE,
                            ONE_TYPE(CV_32FC1),
                            NULL_TYPE,
                            NULL_TYPE,
                            Values(false))); // Values(false) is the reserved parameter


INSTANTIATE_TEST_CASE_P(ImgprocTestBase, integral, Combine(
                            ONE_TYPE(CV_8UC1),
                            NULL_TYPE,
                            ONE_TYPE(CV_32SC1),
                            ONE_TYPE(CV_32FC1),
                            NULL_TYPE,
                            Values(false))); // Values(false) is the reserved parameter

INSTANTIATE_TEST_CASE_P(Imgproc, WarpAffine, Combine(
                            Values(CV_8UC1, CV_8UC4, CV_32FC1, CV_32FC4),
                            Values((MatType)cv::INTER_NEAREST, (MatType)cv::INTER_LINEAR,
                                   (MatType)cv::INTER_CUBIC, (MatType)(cv::INTER_NEAREST | cv::WARP_INVERSE_MAP),
                                   (MatType)(cv::INTER_LINEAR | cv::WARP_INVERSE_MAP), (MatType)(cv::INTER_CUBIC | cv::WARP_INVERSE_MAP))));


INSTANTIATE_TEST_CASE_P(Imgproc, WarpPerspective, Combine
                        (Values(CV_8UC1, CV_8UC4, CV_32FC1, CV_32FC4),
                         Values((MatType)cv::INTER_NEAREST, (MatType)cv::INTER_LINEAR,
                                (MatType)cv::INTER_CUBIC, (MatType)(cv::INTER_NEAREST | cv::WARP_INVERSE_MAP),
                                (MatType)(cv::INTER_LINEAR | cv::WARP_INVERSE_MAP), (MatType)(cv::INTER_CUBIC | cv::WARP_INVERSE_MAP))));


INSTANTIATE_TEST_CASE_P(Imgproc, Resize, Combine(
                            Values(CV_8UC1, CV_8UC4, CV_32FC1, CV_32FC4),  Values(cv::Size()),
                            Values(0.5/*, 1.5, 2*/), Values(0.5/*, 1.5, 2*/), Values((MatType)cv::INTER_NEAREST, (MatType)cv::INTER_LINEAR)));


INSTANTIATE_TEST_CASE_P(Imgproc, Threshold, Combine(
                            Values(CV_8UC1, CV_32FC1), Values(ThreshOp(cv::THRESH_BINARY),
                                    ThreshOp(cv::THRESH_BINARY_INV), ThreshOp(cv::THRESH_TRUNC),
                                    ThreshOp(cv::THRESH_TOZERO), ThreshOp(cv::THRESH_TOZERO_INV))));

INSTANTIATE_TEST_CASE_P(Imgproc, meanShiftFiltering, Combine(
                            ONE_TYPE(CV_8UC4),
                            ONE_TYPE(CV_16SC2),//it is no use in meanShiftFiltering
                            Values(5),
                            Values(6),
                            Values(cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 5, 1))
                        ));

INSTANTIATE_TEST_CASE_P(Imgproc, meanShiftProc, Combine(
                            ONE_TYPE(CV_8UC4),
                            ONE_TYPE(CV_16SC2),
                            Values(5),
                            Values(6),
                            Values(cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 5, 1))
                        ));

INSTANTIATE_TEST_CASE_P(Imgproc, Remap, Combine(
                            Values(CV_8UC1, CV_8UC4, CV_32FC1, CV_32FC4),
                            Values(CV_32FC1, CV_16SC2, CV_32FC2), Values(-1, CV_32FC1),
                            Values((int)cv::INTER_NEAREST, (int)cv::INTER_LINEAR),
                            Values((int)cv::BORDER_CONSTANT)));

INSTANTIATE_TEST_CASE_P(histTestBase, calcHist, Combine(
                            ONE_TYPE(CV_8UC1),
                            ONE_TYPE(CV_32SC1) //no use
                        ));

#endif // HAVE_OPENCL
