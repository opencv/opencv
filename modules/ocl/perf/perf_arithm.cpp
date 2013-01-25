///////////////////////////////////////////////////////////////////////////////////////
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
//    Jiang Liyuan,jlyuan001.good@163.com
//    Rock Li, Rock.Li@amd.com
//    Zailong Wu, bullet@yeah.net
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
#include <iomanip>

#ifdef HAVE_OPENCL
using namespace cv;
using namespace cv::ocl;
using namespace cvtest;
using namespace testing;
using namespace std;
PARAM_TEST_CASE(ArithmTestBase, MatType, bool)
{
    int type;
    cv::Scalar val;

    //src mat
    cv::Mat mat1;
    cv::Mat mat2;
    cv::Mat mask;
    cv::Mat dst;
    cv::Mat dst1; //bak, for two outputs

    // set up roi
    int roicols;
    int roirows;
    int src1x;
    int src1y;
    int src2x;
    int src2y;
    int dstx;
    int dsty;
    int maskx;
    int masky;


    //src mat with roi
    cv::Mat mat1_roi;
    cv::Mat mat2_roi;
    cv::Mat mask_roi;
    cv::Mat dst_roi;
    cv::Mat dst1_roi; //bak
    //std::vector<cv::ocl::Info> oclinfo;
    //ocl dst mat for testing
    cv::ocl::oclMat gdst_whole;
    cv::ocl::oclMat gdst1_whole; //bak

    //ocl mat with roi
    cv::ocl::oclMat gmat1;
    cv::ocl::oclMat gmat2;
    cv::ocl::oclMat gdst;
    cv::ocl::oclMat gdst1;   //bak
    cv::ocl::oclMat gmask;

    virtual void SetUp()
    {
        type = GET_PARAM(0);

        cv::RNG &rng = TS::ptr()->get_rng();

        cv::Size size(MWIDTH, MHEIGHT);

        mat1 = randomMat(rng, size, type, 5, 16, false);
        //mat2 = randomMat(rng, cv::Size(512,3), type, 5, 16, false);
        mat2 = randomMat(rng, size, type, 5, 16, false);
        dst  = randomMat(rng, size, type, 5, 16, false);
        dst1  = randomMat(rng, size, type, 5, 16, false);
        mask = randomMat(rng, size, CV_8UC1, 0, 2,  false);

        cv::threshold(mask, mask, 0.5, 255., CV_8UC1);

        val = cv::Scalar(rng.uniform(-10.0, 10.0), rng.uniform(-10.0, 10.0), rng.uniform(-10.0, 10.0), rng.uniform(-10.0, 10.0));
        //int devnums = getDevice(oclinfo);
        //CV_Assert(devnums>0);
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
            roicols =  mat1.cols - 1;
            roirows = mat1.rows - 1;
            src1x   = 1;
            src2x   = 1;
            src1y   = 1;
            src2y   = 1;
            dstx    = 1;
            dsty    = 1;
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
            maskx	 = 0;
            masky	= 0;
        };

        mat1_roi = mat1(Rect(src1x, src1y, roicols, roirows));
        //mat2_roi = mat2(Rect(src2x,src2y,256,1));
        mat2_roi = mat2(Rect(src2x, src2y, roicols, roirows));
        mask_roi = mask(Rect(maskx, masky, roicols, roirows));
        dst_roi  = dst(Rect(dstx, dsty, roicols, roirows));
        dst1_roi = dst1(Rect(dstx, dsty, roicols, roirows));

        //gdst_whole = dst;
        //gdst = gdst_whole(Rect(dstx,dsty,roicols,roirows));

        //gdst1_whole = dst1;
        //gdst1 = gdst1_whole(Rect(dstx,dsty,roicols,roirows));

        //gmat1 = mat1_roi;
        //gmat2 = mat2_roi;
        //gmask = mask_roi;
    }

};
////////////////////////////////lut/////////////////////////////////////////////////

struct Lut : ArithmTestBase {};

TEST_P(Lut, Mat)
{

    cv::Mat mat2(3, 512, CV_8UC1);
    cv::RNG &rng = TS::ptr()->get_rng();
    rng.fill(mat2, cv::RNG::UNIFORM, cv::Scalar::all(0), cv::Scalar::all(256));

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
            mat2 = randomMat(rng, cv::Size(512, 3), type, 5, 16, false);
            mat2_roi = mat2(Rect(src2x, src2y, 256, 1));


            t0 = (double)cvGetTickCount();//cpu start
            cv::LUT(mat1_roi, mat2_roi, dst_roi);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            gdst_whole = dst;
            gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));

            gmat1 = mat1_roi;
            gmat2 = mat2_roi;

            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::LUT(gmat1, gmat2, gdst);
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
        // s=GetParam();
        cout << "average cpu runtime is  " << totalcputick / ((double)cvGetTickFrequency()* LOOP_TIMES * 1000.) << "ms" << endl;
        cout << "average gpu runtime is  " << totalgputick / ((double)cvGetTickFrequency()* LOOP_TIMES * 1000.) << "ms" << endl;
        cout << "average gpu runtime without data transfer is  " << totalgputick_kernel / ((double)cvGetTickFrequency()* LOOP_TIMES * 1000.) << "ms" << endl;
    }
#else
    for(int j = LOOPROISTART; j < LOOPROIEND; j ++)
    {
        Has_roi(j);
        //  src2x = rng.uniform( 0,mat2.cols - 256);
        // src2y = rng.uniform (0,mat2.rows - 1);

        // cv::Mat mat2_roi = mat2(Rect(src2x,src2y,256,1));
        mat2 = randomMat(rng, cv::Size(512, 3), type, 5, 16, false);
        mat2_roi = mat2(Rect(src2x, src2y, 256, 1));
        gdst_whole = dst;
        gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));
        //   gdst1_whole = dst1;
        //     gdst1 = gdst1_whole(Rect(dstx,dsty,roicols,roirows));
        gmat1 = mat1_roi;
        gmat2 = mat2_roi;
        //     gmask = mask_roi;

        if(j == 0)
        {
            cout << "no roi:";
        }
        else
        {
            cout << "\nwith roi:";
        };
        cv::ocl::LUT(gmat1, gmat2, gdst);
    };
#endif

}



////////////////////////////////exp/////////////////////////////////////////////////

struct Exp : ArithmTestBase {};

TEST_P(Exp, Mat)
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
            cv::exp(mat1_roi, dst_roi);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1

            gdst_whole = dst;
            gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));
            gmat1 = mat1_roi;

            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::exp(gmat1, gdst);
            t2 = (double)cvGetTickCount() - t2;//kernel
            cv::Mat cpu_dst;
            gdst_whole.download(cpu_dst);
            t1 = (double)cvGetTickCount() - t1;//gpu end1
            if(j == 0)
                continue;
            totalgputick = t1 + totalgputick;
            totalcputick = t0 + totalcputick;
            totalgputick_kernel = t2 + totalgputick_kernel;
            //EXPECT_MAT_NEAR(dst, cpu_dst, 0,"");
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
        cv::ocl::exp(gmat1, gdst);
    };
#endif

}


////////////////////////////////log/////////////////////////////////////////////////

struct Log : ArithmTestBase {};

TEST_P(Log, Mat)
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
            cv::log(mat1_roi, dst_roi);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            gdst_whole = dst;
            gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));

            gmat1 = mat1_roi;
            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::log(gmat1, gdst);
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
        cv::ocl::log(gmat1, gdst);
    };
#endif

}




////////////////////////////////add/////////////////////////////////////////////////

struct Add : ArithmTestBase {};

TEST_P(Add, Mat)
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
            cv::add(mat1_roi, mat2_roi, dst_roi);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            gdst_whole = dst;
            gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));

            gmat1 = mat1_roi;
            gmat2 = mat2_roi;
            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::add(gmat1, gmat2, gdst);
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
        gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));
        gmat1 = mat1_roi;
        gmat2 = mat2_roi;
        if(j == 0)
        {
            cout << "no roi:";
        }
        else
        {
            cout << "\nwith roi:";
        };
        cv::ocl::add(gmat1, gmat2, gdst);
    };
#endif
}

TEST_P(Add, Mat_Mask)
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
            cv::add(mat1_roi, mat2_roi, dst_roi, mask_roi);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            gdst_whole = dst;
            gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));

            gmat1 = mat1_roi;
            gmat2 = mat2_roi;
            gmask = mask_roi;
            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::add(gmat1, gmat2, gdst, gmask);
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
        gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));
        gmat1 = mat1_roi;
        gmat2 = mat2_roi;
        gmask = mask_roi;
        if(j == 0)
        {
            cout << "no roi:";
        }
        else
        {
            cout << "\nwith roi:";
        };
        cv::ocl::add(gmat1, gmat2, gdst, gmask);
    };
#endif
}
TEST_P(Add, Scalar)
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
            cv::add(mat1_roi, val, dst_roi);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            gdst_whole = dst;
            gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));
            gmat1 = mat1_roi;
            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::add(gmat1, val, gdst);
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
        cv::ocl::add(gmat1, val, gdst);
    };
#endif
}

TEST_P(Add, Scalar_Mask)
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
            cv::add(mat1_roi, val, dst_roi, mask_roi);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            gdst_whole = dst;
            gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));
            gmat1 = mat1_roi;
            gmask = mask_roi;
            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::add(gmat1, val, gdst, gmask);
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
        gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));
        gmat1 = mat1_roi;
        gmask = mask_roi;
        if(j == 0)
        {
            cout << "no roi:";
        }
        else
        {
            cout << "\nwith roi:";
        };
        cv::ocl::add(gmat1, val, gdst, gmask);
    };
#endif
}


////////////////////////////////sub/////////////////////////////////////////////////
struct Sub : ArithmTestBase {};

TEST_P(Sub, Mat)
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
            cv::subtract(mat1_roi, mat2_roi, dst_roi);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            gdst_whole = dst;
            gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));

            gmat1 = mat1_roi;
            gmat2 = mat2_roi;
            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::subtract(gmat1, gmat2, gdst);
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
        gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));
        gmat1 = mat1_roi;
        gmat2 = mat2_roi;
        if(j == 0)
        {
            cout << "no roi:";
        }
        else
        {
            cout << "\nwith roi:";
        };
        cv::ocl::subtract(gmat1, gmat2, gdst);
    };
#endif
}

TEST_P(Sub, Mat_Mask)
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
            cv::subtract(mat1_roi, mat2_roi, dst_roi, mask_roi);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            gdst_whole = dst;
            gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));

            gmat1 = mat1_roi;
            gmat2 = mat2_roi;
            gmask = mask_roi;
            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::subtract(gmat1, gmat2, gdst, gmask);
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
        gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));
        gmat1 = mat1_roi;
        gmat2 = mat2_roi;
        gmask = mask_roi;
        if(j == 0)
        {
            cout << "no roi:";
        }
        else
        {
            cout << "\nwith roi:";
        };
        cv::ocl::subtract(gmat1, gmat2, gdst, gmask);
    };
#endif
}
TEST_P(Sub, Scalar)
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
            cv::subtract(mat1_roi, val, dst_roi);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            gdst_whole = dst;
            gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));

            gmat1 = mat1_roi;
            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::subtract(gmat1, val, gdst);
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
        cv::ocl::subtract(gmat1, val, gdst);
    };
#endif
}

TEST_P(Sub, Scalar_Mask)
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
            cv::subtract(mat1_roi, val, dst_roi, mask_roi);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            gdst_whole = dst;
            gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));

            gmat1 = mat1_roi;
            gmask = mask_roi;
            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::subtract(gmat1, val, gdst, gmask);
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
        gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));
        gmat1 = mat1_roi;
        gmask = mask_roi;
        if(j == 0)
        {
            cout << "no roi:";
        }
        else
        {
            cout << "\nwith roi:";
        };
        cv::ocl::subtract(gmat1, val, gdst, gmask);
    };
#endif
}


////////////////////////////////Mul/////////////////////////////////////////////////
struct Mul : ArithmTestBase {};

TEST_P(Mul, Mat)
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
            cv::multiply(mat1_roi, mat2_roi, dst_roi);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            gdst_whole = dst;
            gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));

            gmat1 = mat1_roi;
            gmat2 = mat2_roi;
            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::multiply(gmat1, gmat2, gdst);
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
        gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));
        gmat1 = mat1_roi;
        gmat2 = mat2_roi;
        if(j == 0)
        {
            cout << "no roi:";
        }
        else
        {
            cout << "\nwith roi:";
        };
        cv::ocl::multiply(gmat1, gmat2, gdst);
    };
#endif
}

TEST_P(Mul, Mat_Scalar)
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
            cv::RNG &rng = TS::ptr()->get_rng();
            double s = rng.uniform(-10.0, 10.0);
            t0 = (double)cvGetTickCount();//cpu start
            cv::multiply(mat1_roi, mat2_roi, dst_roi, s);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            gdst_whole = dst;
            gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));

            gmat1 = mat1_roi;
            gmat2 = mat2_roi;
            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::multiply(gmat1, gmat2, gdst, s);
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
        cv::RNG &rng = TS::ptr()->get_rng();
        double s = rng.uniform(-10.0, 10.0);
        gdst_whole = dst;
        gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));
        gmat1 = mat1_roi;
        gmat2 = mat2_roi;
        if(j == 0)
        {
            cout << "no roi:";
        }
        else
        {
            cout << "\nwith roi:";
        };
        cv::ocl::multiply(gmat1, gmat2, gdst, s);
    };
#endif
}


struct Div : ArithmTestBase {};

TEST_P(Div, Mat)
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
            cv::divide(mat1_roi, mat2_roi, dst_roi);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            gdst_whole = dst;
            gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));

            gmat1 = mat1_roi;
            gmat2 = mat2_roi;
            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::divide(gmat1, gmat2, gdst);
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
        gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));
        gmat1 = mat1_roi;
        gmat2 = mat2_roi;
        if(j == 0)
        {
            cout << "no roi:";
        }
        else
        {
            cout << "\nwith roi:";
        };
        cv::ocl::divide(gmat1, gmat2, gdst);
    };
#endif
}

TEST_P(Div, Mat_Scalar)
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
            cv::RNG &rng = TS::ptr()->get_rng();
            double s = rng.uniform(-10.0, 10.0);
            t0 = (double)cvGetTickCount();//cpu start
            cv::divide(mat1_roi, mat2_roi, dst_roi, s);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            gdst_whole = dst;
            gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));

            gmat1 = mat1_roi;
            gmat2 = mat2_roi;
            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::divide(gmat1, gmat2, gdst, s);
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
        cv::RNG &rng = TS::ptr()->get_rng();
        double s = rng.uniform(-10.0, 10.0);
        gdst_whole = dst;
        gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));
        gmat1 = mat1_roi;
        gmat2 = mat2_roi;
        if(j == 0)
        {
            cout << "no roi:";
        }
        else
        {
            cout << "\nwith roi:";
        };
        cv::ocl::divide(gmat1, gmat2, gdst, s);
    };
#endif
}


struct Absdiff : ArithmTestBase {};

TEST_P(Absdiff, Mat)
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
            cv::absdiff(mat1_roi, mat2_roi, dst_roi);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            gdst_whole = dst;
            gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));

            gmat1 = mat1_roi;
            gmat2 = mat2_roi;
            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::absdiff(gmat1, gmat2, gdst);
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
        gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));
        gmat1 = mat1_roi;
        gmat2 = mat2_roi;
        if(j == 0)
        {
            cout << "no roi:";
        }
        else
        {
            cout << "\nwith roi:";
        };
        cv::ocl::absdiff(gmat1, gmat2, gdst);
    };
#endif
}

TEST_P(Absdiff, Mat_Scalar)
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
            cv::absdiff(mat1_roi, val, dst_roi);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            gdst_whole = dst;
            gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));

            gmat1 = mat1_roi;
            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::absdiff(gmat1, val, gdst);
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
        cv::ocl::absdiff(gmat1, val, gdst);
    };
#endif
}



struct CartToPolar : ArithmTestBase {};

TEST_P(CartToPolar, angleInDegree)
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
            cv::cartToPolar(mat1_roi, mat2_roi, dst_roi, dst1_roi, 1);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            gdst_whole = dst;
            gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));

            gmat1 = mat1_roi;
            gmat2 = mat2_roi;
            gdst1_whole = dst1;
            gdst1 = gdst1_whole(Rect(dstx, dsty, roicols, roirows));
            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::cartToPolar(gmat1, gmat2, gdst, gdst1, 1);
            t2 = (double)cvGetTickCount() - t2;//kernel
            cv::Mat cpu_dst;
            gdst_whole.download (cpu_dst);//download
            cv::Mat cpu_dst1;
            gdst1_whole.download(cpu_dst1);
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
        gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));
        gdst1_whole = dst1;
        gdst1 = gdst1_whole(Rect(dstx, dsty, roicols, roirows));
        gmat1 = mat1_roi;
        gmat2 = mat2_roi;
        if(j == 0)
        {
            cout << "no roi:";
        }
        else
        {
            cout << "\nwith roi:";
        };
        cv::ocl::cartToPolar(gmat1, gmat2, gdst, gdst1, 1);
    };
#endif
}

TEST_P(CartToPolar, angleInRadians)
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
            cv::cartToPolar(mat1_roi, mat2_roi, dst_roi, dst1_roi, 0);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            gdst_whole = dst;
            gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));
            gdst1_whole = dst1;
            gdst1 = gdst1_whole(Rect(dstx, dsty, roicols, roirows));
            gmat1 = mat1_roi;
            gmat2 = mat2_roi;
            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::cartToPolar(gmat1, gmat2, gdst, gdst1, 0);
            t2 = (double)cvGetTickCount() - t2;//kernel
            cv::Mat cpu_dst;
            gdst_whole.download (cpu_dst);//download
            cv::Mat cpu_dst1;
            gdst1_whole.download(cpu_dst1);
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
        gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));
        gdst1_whole = dst1;
        gdst1 = gdst1_whole(Rect(dstx, dsty, roicols, roirows));
        gmat1 = mat1_roi;
        gmat2 = mat2_roi;
        if(j == 0)
        {
            cout << "no roi:";
        }
        else
        {
            cout << "\nwith roi:";
        };
        cv::ocl::cartToPolar(gmat1, gmat2, gdst, gdst1, 0);
    };
#endif
}


struct PolarToCart : ArithmTestBase {};

TEST_P(PolarToCart, angleInDegree)
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
            cv::polarToCart(mat1_roi, mat2_roi, dst_roi, dst1_roi, 1);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            gdst_whole = dst;
            gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));

            gmat1 = mat1_roi;
            gmat2 = mat2_roi;
            gdst1_whole = dst1;
            gdst1 = gdst1_whole(Rect(dstx, dsty, roicols, roirows));
            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::polarToCart(gmat1, gmat2, gdst, gdst1, 1);
            t2 = (double)cvGetTickCount() - t2;//kernel
            cv::Mat cpu_dst;
            gdst_whole.download (cpu_dst);//download
            cv::Mat cpu_dst1;
            gdst1_whole.download(cpu_dst1);
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
        gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));
        gdst1_whole = dst1;
        gdst1 = gdst1_whole(Rect(dstx, dsty, roicols, roirows));
        gmat1 = mat1_roi;
        gmat2 = mat2_roi;
        if(j == 0)
        {
            cout << "no roi:";
        }
        else
        {
            cout << "\nwith roi:";
        };
        cv::ocl::polarToCart(gmat1, gmat2, gdst, gdst1, 1);
    };
#endif
}

TEST_P(PolarToCart, angleInRadians)
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
            cv::polarToCart(mat1_roi, mat2_roi, dst_roi, dst1_roi, 0);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            gdst_whole = dst;
            gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));

            gmat1 = mat1_roi;
            gmat2 = mat2_roi;
            gdst1_whole = dst1;
            gdst1 = gdst1_whole(Rect(dstx, dsty, roicols, roirows));
            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::polarToCart(gmat1, gmat2, gdst, gdst1, 0);
            t2 = (double)cvGetTickCount() - t2;//kernel
            cv::Mat cpu_dst;
            gdst_whole.download (cpu_dst);//download
            cv::Mat cpu_dst1;
            gdst1_whole.download(cpu_dst1);
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
        gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));
        gmat1 = mat1_roi;
        gmat2 = mat2_roi;
        gdst1_whole = dst1;
        gdst1 = gdst1_whole(Rect(dstx, dsty, roicols, roirows));
        if(j == 0)
        {
            cout << "no roi:";
        }
        else
        {
            cout << "\nwith roi:";
        };
        cv::ocl::polarToCart(gmat1, gmat2, gdst, gdst1, 0);
    };
#endif
}



struct Magnitude : ArithmTestBase {};

TEST_P(Magnitude, Mat)
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
            cv::magnitude(mat1_roi, mat2_roi, dst_roi);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            gdst_whole = dst;
            gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));

            gmat1 = mat1_roi;
            gmat2 = mat2_roi;
            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::magnitude(gmat1, gmat2, gdst);
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
        gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));
        gmat1 = mat1_roi;
        gmat2 = mat2_roi;
        if(j == 0)
        {
            cout << "no roi:";
        }
        else
        {
            cout << "\nwith roi:";
        };
        cv::ocl::magnitude(gmat1, gmat2, gdst);
    };
#endif
}

struct Transpose : ArithmTestBase {};

TEST_P(Transpose, Mat)
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
            cv::transpose(mat1_roi, dst_roi);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            gdst_whole = dst;
            gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));

            gmat1 = mat1_roi;
            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::transpose(gmat1, gdst);
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
        cv::ocl::transpose(gmat1, gdst);
    };
#endif
}


struct Flip : ArithmTestBase {};

TEST_P(Flip, X)
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
            cv::flip(mat1_roi, dst_roi, 0);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            gdst_whole = dst;
            gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));

            gmat1 = mat1_roi;
            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::flip(gmat1, gdst, 0);
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
        cv::ocl::flip(gmat1, gdst, 0);
    };
#endif
}

TEST_P(Flip, Y)
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
            cv::flip(mat1_roi, dst_roi, 1);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            gdst_whole = dst;
            gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));

            gmat1 = mat1_roi;
            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::flip(gmat1, gdst, 1);
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
        cv::ocl::flip(gmat1, gdst, 1);
    };
#endif
}

TEST_P(Flip, BOTH)
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
            cv::flip(mat1_roi, dst_roi, -1);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            gdst_whole = dst;
            gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));

            gmat1 = mat1_roi;
            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::flip(gmat1, gdst, -1);
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
        cv::ocl::flip(gmat1, gdst, -1);
    };
#endif
}



struct MinMax : ArithmTestBase {};

TEST_P(MinMax, MAT)
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
            double minVal, maxVal;
            cv::Point minLoc, maxLoc;
            t0 = (double)cvGetTickCount();//cpu start
            if (mat1.depth() != CV_8S)
            {
                cv::minMaxLoc(mat1_roi, &minVal, &maxVal, &minLoc, &maxLoc);
            }
            else
            {
                minVal = std::numeric_limits<double>::max();
                maxVal = -std::numeric_limits<double>::max();
                for (int i = 0; i < mat1_roi.rows; ++i)
                    for (int j = 0; j < mat1_roi.cols; ++j)
                    {
                        signed char val = mat1_roi.at<signed char>(i, j);
                        if (val < minVal) minVal = val;
                        if (val > maxVal) maxVal = val;
                    }
            }

            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            gmat1 = mat1_roi;
            double minVal_, maxVal_;
            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::minMax(gmat1, &minVal_, &maxVal_);
            t2 = (double)cvGetTickCount() - t2;//kernel
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
        double minVal_, maxVal_;
        if(j == 0)
        {
            cout << "no roi:";
        }
        else
        {
            cout << "\nwith roi:";
        };
        cv::ocl::minMax(gmat1, &minVal_, &maxVal_);
    };
#endif
}

TEST_P(MinMax, MASK)
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
            double minVal, maxVal;
            cv::Point minLoc, maxLoc;
            t0 = (double)cvGetTickCount();//cpu start
            if (mat1.depth() != CV_8S)
            {
                cv::minMaxLoc(mat1_roi, &minVal, &maxVal, &minLoc, &maxLoc, mask_roi);
            }
            else
            {
                minVal = std::numeric_limits<double>::max();
                maxVal = -std::numeric_limits<double>::max();
                for (int i = 0; i < mat1_roi.rows; ++i)
                    for (int j = 0; j < mat1_roi.cols; ++j)
                    {
                        signed char val = mat1_roi.at<signed char>(i, j);
                        unsigned char m = mask_roi.at<unsigned char>(i, j);
                        if (val < minVal && m) minVal = val;
                        if (val > maxVal && m) maxVal = val;
                    }
            }

            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            gmat1 = mat1_roi;
            gmask = mask_roi;
            double minVal_, maxVal_;
            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::minMax(gmat1, &minVal_, &maxVal_, gmask);
            t2 = (double)cvGetTickCount() - t2;//kernel
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
        gmask = mask_roi;
        double minVal_, maxVal_;
        if(j == 0)
        {
            cout << "no roi:";
        }
        else
        {
            cout << "\nwith roi:";
        };
        cv::ocl::minMax(gmat1, &minVal_, &maxVal_, gmask);
    };
#endif
}


struct MinMaxLoc : ArithmTestBase {};

TEST_P(MinMaxLoc, MAT)
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
            double minVal, maxVal;
            cv::Point minLoc, maxLoc;
            int depth = mat1.depth();
            t0 = (double)cvGetTickCount();//cpu start
            if (depth != CV_8S)
            {
                cv::minMaxLoc(mat1_roi, &minVal, &maxVal, &minLoc, &maxLoc);
            }
            else
            {
                minVal = std::numeric_limits<double>::max();
                maxVal = -std::numeric_limits<double>::max();
                for (int i = 0; i < mat1_roi.rows; ++i)
                    for (int j = 0; j < mat1_roi.cols; ++j)
                    {
                        signed char val = mat1_roi.at<signed char>(i, j);
                        if (val < minVal)
                        {
                            minVal = val;
                            minLoc.x = j;
                            minLoc.y = i;
                        }
                        if (val > maxVal)
                        {
                            maxVal = val;
                            maxLoc.x = j;
                            maxLoc.y = i;
                        }
                    }
            }


            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            gmat1 = mat1_roi;
            double minVal_, maxVal_;
            cv::Point minLoc_, maxLoc_;
            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::minMaxLoc(gmat1, &minVal_, &maxVal_, &minLoc_, &maxLoc_, cv::ocl::oclMat());
            t2 = (double)cvGetTickCount() - t2;//kernel
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
        double minVal_, maxVal_;
        cv::Point minLoc_, maxLoc_;
        if(j == 0)
        {
            cout << "no roi:";
        }
        else
        {
            cout << "\nwith roi:";
        };
        cv::ocl::minMaxLoc(gmat1, &minVal_, &maxVal_, &minLoc_, &maxLoc_, cv::ocl::oclMat());
    };
#endif

}


TEST_P(MinMaxLoc, MASK)
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
            double minVal, maxVal;
            cv::Point minLoc, maxLoc;
            int depth = mat1.depth();
            t0 = (double)cvGetTickCount();//cpu start
            if (depth != CV_8S)
            {
                cv::minMaxLoc(mat1_roi, &minVal, &maxVal, &minLoc, &maxLoc, mask_roi);
            }
            else
            {
                minVal = std::numeric_limits<double>::max();
                maxVal = -std::numeric_limits<double>::max();
                for (int i = 0; i < mat1_roi.rows; ++i)
                    for (int j = 0; j < mat1_roi.cols; ++j)
                    {
                        signed char val = mat1_roi.at<signed char>(i, j);
                        unsigned char m = mask_roi.at<unsigned char>(i , j);
                        if (val < minVal && m)
                        {
                            minVal = val;
                            minLoc.x = j;
                            minLoc.y = i;
                        }
                        if (val > maxVal && m)
                        {
                            maxVal = val;
                            maxLoc.x = j;
                            maxLoc.y = i;
                        }
                    }
            }


            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            gmat1 = mat1_roi;
            gmask = mask_roi;
            double minVal_, maxVal_;
            cv::Point minLoc_, maxLoc_;
            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::minMaxLoc(gmat1, &minVal_, &maxVal_, &minLoc_, &maxLoc_, gmask);
            t2 = (double)cvGetTickCount() - t2;//kernel
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
        gmask = mask_roi;
        double minVal_, maxVal_;
        cv::Point minLoc_, maxLoc_;
        if(j == 0)
        {
            cout << "no roi:";
        }
        else
        {
            cout << "\nwith roi:";
        };
        cv::ocl::minMaxLoc(gmat1, &minVal_, &maxVal_, &minLoc_, &maxLoc_, gmask);
    };
#endif
}


struct Sum : ArithmTestBase {};

TEST_P(Sum, MAT)
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
            cv::sum(mat1_roi);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            gmat1 = mat1_roi;
            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::sum(gmat1);
            t2 = (double)cvGetTickCount() - t2;//kernel
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
        if(j == 0)
        {
            cout << "no roi:";
        }
        else
        {
            cout << "\nwith roi:";
        };
        Scalar gpures = cv::ocl::sum(gmat1);
    };
#endif
}

//TEST_P(Sum, MASK)
//{
//    for(int j=0; j<LOOP_TIMES; j++)
//    {
//
//    }
//}

struct CountNonZero : ArithmTestBase {};

TEST_P(CountNonZero, MAT)
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
            cv::countNonZero(mat1_roi);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            gmat1 = mat1_roi;
            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::countNonZero(gmat1);
            t2 = (double)cvGetTickCount() - t2;//kernel
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
        if(j == 0)
        {
            cout << "no roi:";
        }
        else
        {
            cout << "\nwith roi:";
        };
        cv::ocl::countNonZero(gmat1);
    };
#endif

}



////////////////////////////////phase/////////////////////////////////////////////////
struct Phase : ArithmTestBase {};

TEST_P(Phase, Mat)
{
    if(mat1.depth() != CV_32F && mat1.depth() != CV_64F)
    {
        cout << "\tUnsupported type\t\n";
    }

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
            cv::phase(mat1_roi, mat2_roi, dst_roi, 0);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            gdst_whole = dst;
            gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));

            gmat1 = mat1_roi;
            gmat2 = mat2_roi;
            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::phase(gmat1, gmat2, gdst, 0);
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
        gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));
        gmat1 = mat1_roi;
        gmat2 = mat2_roi;
        if(j == 0)
        {
            cout << "no roi:";
        }
        else
        {
            cout << "\nwith roi:";
        };
        cv::ocl::phase(gmat1, gmat2, gdst, 0);
    };
#endif

}


////////////////////////////////bitwise_and/////////////////////////////////////////////////
struct Bitwise_and : ArithmTestBase {};

TEST_P(Bitwise_and, Mat)
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
            cv::bitwise_and(mat1_roi, mat2_roi, dst_roi);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            gdst_whole = dst;
            gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));

            gmat1 = mat1_roi;
            gmat2 = mat2_roi;
            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::bitwise_and(gmat1, gmat2, gdst);
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
        gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));
        gmat1 = mat1_roi;
        gmat2 = mat2_roi;

        if(j == 0)
        {
            cout << "no roi:";
        }
        else
        {
            cout << "\nwith roi:";
        };
        cv::ocl::bitwise_and(gmat1, gmat2, gdst);
    };
#endif

}

TEST_P(Bitwise_and, Mat_Mask)
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
            cv::bitwise_and(mat1_roi, mat2_roi, dst_roi, mask_roi);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            gdst_whole = dst;
            gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));

            gmat1 = mat1_roi;
            gmat2 = mat2_roi;
            gmask = mask_roi;
            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::bitwise_and(gmat1, gmat2, gdst, gmask);
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
        gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));
        gmat1 = mat1_roi;
        gmat2 = mat2_roi;
        gmask = mask_roi;

        if(j == 0)
        {
            cout << "no roi:";
        }
        else
        {
            cout << "\nwith roi:";
        };
        cv::ocl::bitwise_and(gmat1, gmat2, gdst, gmask);
    };
#endif
}

TEST_P(Bitwise_and, Scalar)
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
            cv::bitwise_and(mat1_roi, val, dst_roi);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            gdst_whole = dst;
            gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));

            gmat1 = mat1_roi;
            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::bitwise_and(gmat1, val, gdst);
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
        cv::ocl::bitwise_and(gmat1, val, gdst);
    };
#endif
}

TEST_P(Bitwise_and, Scalar_Mask)
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
            cv::bitwise_and(mat1_roi, val, dst_roi, mask_roi);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            gdst_whole = dst;
            gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));

            gmat1 = mat1_roi;
            gmat2 = mat2_roi;
            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::bitwise_and(gmat1, val, gdst, gmask);
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
        gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));
        gmat1 = mat1_roi;
        gmask = mask_roi;

        if(j == 0)
        {
            cout << "no roi:";
        }
        else
        {
            cout << "\nwith roi:";
        };
        cv::ocl::bitwise_and(gmat1, val, gdst, gmask);
    };
#endif
}



////////////////////////////////bitwise_or/////////////////////////////////////////////////

struct Bitwise_or : ArithmTestBase {};

TEST_P(Bitwise_or, Mat)
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
            cv::bitwise_or(mat1_roi, mat2_roi, dst_roi);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            gdst_whole = dst;
            gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));

            gmat1 = mat1_roi;
            gmat2 = mat2_roi;
            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::bitwise_or(gmat1, gmat2, gdst);
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
        gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));
        gmat1 = mat1_roi;
        gmat2 = mat2_roi;

        if(j == 0)
        {
            cout << "no roi:";
        }
        else
        {
            cout << "\nwith roi:";
        };
        cv::ocl::bitwise_or(gmat1, gmat2, gdst);
    };
#endif
}

TEST_P(Bitwise_or, Mat_Mask)
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
            cv::bitwise_or(mat1_roi, mat2_roi, dst_roi, mask_roi);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            gdst_whole = dst;
            gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));

            gmat1 = mat1_roi;
            gmat2 = mat2_roi;
            gmask = mask_roi;
            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::bitwise_or(gmat1, gmat2, gdst, gmask);
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
        gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));
        gmat1 = mat1_roi;
        gmat2 = mat2_roi;
        gmask = mask_roi;

        if(j == 0)
        {
            cout << "no roi:";
        }
        else
        {
            cout << "\nwith roi:";
        };
        cv::ocl::bitwise_or(gmat1, gmat2, gdst, gmask);
    };
#endif
}
TEST_P(Bitwise_or, Scalar)
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
            cv::bitwise_or(mat1_roi, val, dst_roi);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            gdst_whole = dst;
            gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));

            gmat1 = mat1_roi;
            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::bitwise_or(gmat1, val, gdst);
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
        cv::ocl::bitwise_or(gmat1, val, gdst);
    };
#endif
}

TEST_P(Bitwise_or, Scalar_Mask)
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
            cv::bitwise_or(mat1_roi, val, dst_roi, mask_roi);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            gdst_whole = dst;
            gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));

            gmat1 = mat1_roi;
            gmask = mask_roi;
            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::bitwise_or(gmat1, val, gdst, gmask);
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
        gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));
        gmat1 = mat1_roi;
        gmask = mask_roi;

        if(j == 0)
        {
            cout << "no roi:";
        }
        else
        {
            cout << "\nwith roi:";
        };
        cv::ocl::bitwise_or(gmat1, val, gdst, gmask);
    };
#endif
}


////////////////////////////////bitwise_xor/////////////////////////////////////////////////

struct Bitwise_xor : ArithmTestBase {};

TEST_P(Bitwise_xor, Mat)
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
            cv::bitwise_xor(mat1_roi, mat2_roi, dst_roi);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            gdst_whole = dst;
            gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));

            gmat1 = mat1_roi;
            gmat2 = mat2_roi;
            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::bitwise_xor(gmat1, gmat2, gdst);
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
        gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));
        gmat1 = mat1_roi;
        gmat2 = mat2_roi;

        if(j == 0)
        {
            cout << "no roi:";
        }
        else
        {
            cout << "\nwith roi:";
        };
        cv::ocl::bitwise_xor(gmat1, gmat2, gdst);
    };
#endif
}

TEST_P(Bitwise_xor, Mat_Mask)
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
            cv::bitwise_xor(mat1_roi, mat2_roi, dst_roi, mask_roi);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            gdst_whole = dst;
            gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));

            gmat1 = mat1_roi;
            gmat2 = mat2_roi;
            gmask = mask_roi;
            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::bitwise_xor(gmat1, gmat2, gdst, gmask);
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
        gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));
        gmat1 = mat1_roi;
        gmat2 = mat2_roi;
        gmask = mask_roi;

        if(j == 0)
        {
            cout << "no roi:";
        }
        else
        {
            cout << "\nwith roi:";
        };
        cv::ocl::bitwise_xor(gmat1, gmat2, gdst, gmask);
    };
#endif
}

TEST_P(Bitwise_xor, Scalar)
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
            cv::bitwise_xor(mat1_roi, val, dst_roi);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            gdst_whole = dst;
            gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));

            gmat1 = mat1_roi;
            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::bitwise_xor(gmat1, val, gdst);
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
        cv::ocl::bitwise_xor(gmat1, val, gdst);
    };
#endif
}

TEST_P(Bitwise_xor, Scalar_Mask)
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
            cv::bitwise_xor(mat1_roi, val, dst_roi, mask_roi);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            gdst_whole = dst;
            gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));

            gmat1 = mat1_roi;
            gmask = mask_roi;
            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::bitwise_xor(gmat1, val, gdst, gmask);
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
        gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));
        gmat1 = mat1_roi;
        gmask = mask_roi;

        if(j == 0)
        {
            cout << "no roi:";
        }
        else
        {
            cout << "\nwith roi:";
        };
        cv::ocl::bitwise_xor(gmat1, val, gdst, gmask);
    };
#endif
}


////////////////////////////////bitwise_not/////////////////////////////////////////////////

struct Bitwise_not : ArithmTestBase {};

TEST_P(Bitwise_not, Mat)
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
            cv::bitwise_not(mat1_roi, dst_roi);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            gdst_whole = dst;
            gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));

            gmat1 = mat1_roi;
            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::bitwise_not(gmat1, gdst);
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
        cv::ocl::bitwise_not(gmat1, gdst);
    };
#endif
}

////////////////////////////////compare/////////////////////////////////////////////////
PARAM_TEST_CASE ( CompareTestBase, MatType, bool)
{
    int type;
    cv::Scalar val;

    //src mat
    cv::Mat mat1;
    cv::Mat mat2;
    cv::Mat mask;
    cv::Mat dst;
    cv::Mat dst1; //bak, for two outputs

    // set up roi
    int roicols;
    int roirows;
    int src1x;
    int src1y;
    int src2x;
    int src2y;
    int dstx;
    int dsty;
    int maskx;
    int masky;


    //src mat with roi
    cv::Mat mat1_roi;
    cv::Mat mat2_roi;
    cv::Mat mask_roi;
    cv::Mat dst_roi;
    cv::Mat dst1_roi; //bak
    //std::vector<cv::ocl::Info> oclinfo;
    //ocl dst mat for testing
    cv::ocl::oclMat gdst_whole;
    cv::ocl::oclMat gdst1_whole; //bak

    //ocl mat with roi
    cv::ocl::oclMat gmat1;
    cv::ocl::oclMat gmat2;
    cv::ocl::oclMat gdst;
    cv::ocl::oclMat gdst1;   //bak
    cv::ocl::oclMat gmask;

    virtual void SetUp()
    {
        //type = GET_PARAM(0);
        type = CV_8UC1;

        cv::RNG &rng = TS::ptr()->get_rng();

        cv::Size size(MWIDTH, MHEIGHT);

        mat1 = randomMat(rng, size, type, 5, 16, false);
        //mat2 = randomMat(rng, cv::Size(512,3), type, 5, 16, false);
        mat2 = randomMat(rng, size, type, 5, 16, false);
        dst  = randomMat(rng, size, type, 5, 16, false);
        dst1  = randomMat(rng, size, type, 5, 16, false);
        mask = randomMat(rng, size, CV_8UC1, 0, 2,  false);

        cv::threshold(mask, mask, 0.5, 255., CV_8UC1);

        val = cv::Scalar(rng.uniform(-10.0, 10.0), rng.uniform(-10.0, 10.0), rng.uniform(-10.0, 10.0), rng.uniform(-10.0, 10.0));
        //int devnums = getDevice(oclinfo);
        //CV_Assert(devnums>0);
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
            roicols =  mat1.cols - 1;
            roirows = mat1.rows - 1;
            src1x   = 1;
            src2x   = 1;
            src1y   = 1;
            src2y   = 1;
            dstx    = 1;
            dsty    = 1;
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
            maskx	 = 0;
            masky	= 0;
        };

        mat1_roi = mat1(Rect(src1x, src1y, roicols, roirows));
        //mat2_roi = mat2(Rect(src2x,src2y,256,1));
        mat2_roi = mat2(Rect(src2x, src2y, roicols, roirows));
        mask_roi = mask(Rect(maskx, masky, roicols, roirows));
        dst_roi  = dst(Rect(dstx, dsty, roicols, roirows));
        dst1_roi = dst1(Rect(dstx, dsty, roicols, roirows));

        //gdst_whole = dst;
        //gdst = gdst_whole(Rect(dstx,dsty,roicols,roirows));

        //gdst1_whole = dst1;
        //gdst1 = gdst1_whole(Rect(dstx,dsty,roicols,roirows));

        //gmat1 = mat1_roi;
        //gmat2 = mat2_roi;
        //gmask = mask_roi;
    }

};
struct Compare : CompareTestBase {};

TEST_P(Compare, Mat)
{
    if(mat1.type() == CV_8SC1)
    {
        cout << "\tUnsupported type\t\n";
    }

    int cmp_codes[] = {CMP_EQ, CMP_GT, CMP_GE, CMP_LT, CMP_LE, CMP_NE};
    const char *cmp_str[] = {"CMP_EQ", "CMP_GT", "CMP_GE", "CMP_LT", "CMP_LE", "CMP_NE"};
    int cmp_num = sizeof(cmp_codes) / sizeof(int);
    for (int i = 0; i < cmp_num; ++i)
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
                cv::compare(mat1_roi, mat2_roi, dst_roi, cmp_codes[i]);
                t0 = (double)cvGetTickCount() - t0;//cpu end

                t1 = (double)cvGetTickCount();//gpu start1
                gdst_whole = dst;
                gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));

                gmat1 = mat1_roi;
                gmat2 = mat2_roi;
                t2 = (double)cvGetTickCount(); //kernel
                cv::ocl::compare(gmat1, gmat2, gdst, cmp_codes[i]);
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
            cout << cmp_str[i] << endl;
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
            gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));
            gmat1 = mat1_roi;
            gmat2 = mat2_roi;
            if(j == 0)
            {
                cout << "no roi:";
            }
            else
            {
                cout << "\nwith roi:";
            };
            cv::ocl::compare(gmat1, gmat2, gdst, cmp_codes[i]);
        };
#endif
    }

}

struct Pow : ArithmTestBase {};

TEST_P(Pow, Mat)
{
    if(mat1.depth() != CV_32F && mat1.depth() != CV_64F)
    {
        cout << "\tUnsupported type\t\n";
    }

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
            double p = 4.5;
            t0 = (double)cvGetTickCount();//cpu start
            cv::pow(mat1_roi, p, dst_roi);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            gdst_whole = dst;
            gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));

            gmat1 = mat1_roi;
            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::pow(gmat1, p, gdst);
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
        double p = 4.5;
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
        cv::ocl::pow(gmat1, p, gdst);
    };
#endif
}


struct MagnitudeSqr : ArithmTestBase {};

TEST_P(MagnitudeSqr, Mat)
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
            for(int i = 0; i < mat1.rows; ++i)
                for(int j = 0; j < mat1.cols; ++j)
                {
                    float val1 = mat1.at<float>(i, j);
                    float val2 = mat2.at<float>(i, j);

                    ((float *)(dst.data))[i * dst.step / 4 + j] = val1 * val1 + val2 * val2;

                }
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            cv::ocl::oclMat clmat1(mat1), clmat2(mat2), cldst;
            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::magnitudeSqr(clmat1, clmat2, cldst);
            t2 = (double)cvGetTickCount() - t2;//kernel
            cv::Mat cpu_dst;
            cldst.download(cpu_dst);//download
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
        cv::ocl::oclMat clmat1(mat1), clmat2(mat2), cldst;
        if(j == 0)
        {
            cout << "no roi:";
        }
        else
        {
            cout << "\nwith roi:";
        };
        cv::ocl::magnitudeSqr(clmat1, clmat2, cldst);
    };
#endif

}


struct AddWeighted : ArithmTestBase {};

TEST_P(AddWeighted, Mat)
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
            double alpha = 2.0, beta = 1.0, gama = 3.0;

            t0 = (double)cvGetTickCount();//cpu start
            cv::addWeighted(mat1_roi, alpha, mat2_roi, beta, gama, dst_roi);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1

            gdst_whole = dst;
            gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));

            gmat1 = mat1_roi;
            gmat2 = mat2_roi;

            t2 = (double)cvGetTickCount(); //kernel
            cv::ocl::addWeighted(gmat1, alpha, gmat2, beta, gama, gdst);
            t2 = (double)cvGetTickCount() - t2;//kernel
            cv::Mat cpu_dst;
            gdst_whole.download(cpu_dst);
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
        double alpha = 2.0, beta = 1.0, gama = 3.0;
        gdst_whole = dst;
        gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));
        gmat1 = mat1_roi;
        gmat2 = mat2_roi;
        if(j == 0)
        {
            cout << "no roi:";
        }
        else
        {
            cout << "\nwith roi:";
        };
        cv::ocl::addWeighted(gmat1, alpha, gmat2, beta, gama, gdst);
        // double alpha=2.0,beta=1.0,gama=3.0;
        // cv::ocl::oclMat clmat1(mat1),clmat2(mat2),cldst;
        // if(j==0){cout<<"no roi:";}else{cout<<"\nwith roi:";};
        // cv::ocl::addWeighted(clmat1,alpha,clmat2,beta,gama, cldst);
    };
#endif

}
/*
struct AddWeighted : ArithmTestBase {};

TEST_P(AddWeighted, Mat)
{
#ifndef PRINT_KERNEL_RUN_TIME
    double totalcputick=0;
    double totalgputick=0;
    double totalgputick_kernel=0;
    double t0=0;
    double t1=0;
    double t2=0;
    for(int j = 0; j < LOOP_TIMES+1; j ++)
    {
        double alpha=2.0,beta=1.0,gama=3.0;

        t0 = (double)cvGetTickCount();//cpu start
        cv::addWeighted(mat1,alpha,mat2,beta,gama,dst);
        t0 = (double)cvGetTickCount() - t0;//cpu end

        t1 = (double)cvGetTickCount();//gpu start1
        cv::ocl::oclMat clmat1(mat1),clmat2(mat2),cldst;

        t2=(double)cvGetTickCount();//kernel
        cv::ocl::addWeighted(clmat1,alpha,clmat2,beta,gama, cldst);
        t2 = (double)cvGetTickCount() - t2;//kernel
        cv::Mat cpu_dst;
        cldst.download(cpu_dst);
        t1 = (double)cvGetTickCount() - t1;//gpu end1
        if(j == 0)
            continue;
        totalgputick=t1+totalgputick;
        totalcputick=t0+totalcputick;
        totalgputick_kernel=t2+totalgputick_kernel;

    }
    cout << "average cpu runtime is  " << totalcputick/((double)cvGetTickFrequency()* LOOP_TIMES *1000.) << "ms" << endl;
    cout << "average gpu runtime is  " << totalgputick/((double)cvGetTickFrequency()* LOOP_TIMES *1000.) << "ms" << endl;
    cout << "average gpu runtime without data transfer is  " << totalgputick_kernel/((double)cvGetTickFrequency()* LOOP_TIMES *1000.) << "ms" << endl;

#else
    //for(int j = LOOPROISTART; j < LOOPROIEND; j ++)
    //	{
    double alpha=2.0,beta=1.0,gama=3.0;
    cv::ocl::oclMat clmat1(mat1),clmat2(mat2),cldst;
    //if(j==0){cout<<"no roi:";}else{cout<<"\nwith roi:";};
    cv::ocl::addWeighted(clmat1,alpha,clmat2,beta,gama, cldst);
    //	};
#endif

}

*/
//********test****************

INSTANTIATE_TEST_CASE_P(Arithm, Lut, Combine(
                            Values(CV_8UC1, CV_8UC4),
                            Values(false))); // Values(false) is the reserved parameter

INSTANTIATE_TEST_CASE_P(Arithm, Exp, Combine(
                            Values(CV_32FC1, CV_64FC1),
                            Values(false))); // Values(false) is the reserved parameter

INSTANTIATE_TEST_CASE_P(Arithm, Log, Combine(
                            Values(CV_32FC1, CV_64FC1),
                            Values(false))); // Values(false) is the reserved parameter

INSTANTIATE_TEST_CASE_P(Arithm, Add, Combine(
                            Values(CV_8UC1, CV_8UC4, CV_32FC1,  CV_32FC4),
                            Values(false)));

INSTANTIATE_TEST_CASE_P(Arithm, Mul, Combine(
                            Values(CV_8UC1, CV_8UC4, CV_32FC1, CV_32FC4),
                            Values(false))); // Values(false) is the reserved parameter

INSTANTIATE_TEST_CASE_P(Arithm, Div, Combine(
                            Values(CV_8UC1, CV_8UC4, CV_32FC1, CV_32FC4),
                            Values(false))); // Values(false) is the reserved parameter


INSTANTIATE_TEST_CASE_P(Arithm, Absdiff, Combine(
                            Values(CV_8UC1, CV_8UC4, CV_32FC1, CV_32FC4),
                            Values(false))); // Values(false) is the reserved parameter

INSTANTIATE_TEST_CASE_P(Arithm, CartToPolar, Combine(
                            Values(CV_32FC1, CV_32FC4),
                            Values(false))); // Values(false) is the reserved parameter

INSTANTIATE_TEST_CASE_P(Arithm, PolarToCart, Combine(
                            Values(CV_32FC1, CV_32FC4),
                            Values(false))); // Values(false) is the reserved parameter

INSTANTIATE_TEST_CASE_P(Arithm, Magnitude, Combine(
                            Values(CV_32FC1, CV_32FC4),
                            Values(false))); // Values(false) is the reserved parameter

INSTANTIATE_TEST_CASE_P(Arithm, Transpose, Combine(
                            Values(CV_8UC1, CV_8UC4, CV_32FC1),
                            Values(false))); // Values(false) is the reserved parameter

INSTANTIATE_TEST_CASE_P(Arithm, Flip, Combine(
                            Values(CV_8UC1, CV_8UC4, CV_32SC1, CV_32FC1, CV_32FC4),
                            Values(false))); // Values(false) is the reserved parameter

INSTANTIATE_TEST_CASE_P(Arithm, MinMax, Combine(
                            Values(CV_8UC1, CV_32FC1),
                            Values(false)));

INSTANTIATE_TEST_CASE_P(Arithm, MinMaxLoc, Combine(
                            Values(CV_8UC1, CV_32FC1),
                            Values(false)));

INSTANTIATE_TEST_CASE_P(Arithm, Sum, Combine(
                            Values(CV_8U, CV_32S, CV_32F),
                            Values(false)));

INSTANTIATE_TEST_CASE_P(Arithm, CountNonZero, Combine(
                            Values(CV_8U, CV_32S, CV_32F),
                            Values(false)));


INSTANTIATE_TEST_CASE_P(Arithm, Phase, Combine(Values(CV_32FC1, CV_32FC4), Values(false)));
//Values(false) is the reserved parameter


INSTANTIATE_TEST_CASE_P(Arithm, Bitwise_and, Combine(
                            Values(CV_8UC1, CV_32SC1, CV_32SC4, CV_32FC1, CV_32FC4), Values(false)));
//Values(false) is the reserved parameter

INSTANTIATE_TEST_CASE_P(Arithm, Bitwise_or, Combine(
                            Values(CV_8UC1, CV_8UC4, CV_32FC1, CV_32FC4), Values(false)));
//Values(false) is the reserved parameter

INSTANTIATE_TEST_CASE_P(Arithm, Bitwise_xor, Combine(
                            Values(CV_8UC1, CV_32SC1, CV_32FC1, CV_32FC4), Values(false)));
//Values(false) is the reserved parameter

INSTANTIATE_TEST_CASE_P(Arithm, Bitwise_not, Combine(
                            Values(CV_8UC1, CV_32SC1, CV_32FC1, CV_32FC4), Values(false)));
//Values(false) is the reserved parameter

INSTANTIATE_TEST_CASE_P(Arithm, Compare, Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1, CV_32SC1, CV_32FC1, CV_64FC1), Values(false)));
//Values(false) is the reserved parameter

INSTANTIATE_TEST_CASE_P(Arithm, Pow, Combine(Values(CV_32FC1, CV_32FC4), Values(false)));
//Values(false) is the reserved parameter

INSTANTIATE_TEST_CASE_P(Arithm, MagnitudeSqr, Combine(
                            Values(CV_32FC1, CV_32FC1),
                            Values(false))); // Values(false) is the reserved parameter

INSTANTIATE_TEST_CASE_P(Arithm, AddWeighted, Combine(
                            Values(CV_8UC1, CV_32SC1, CV_32FC1),
                            Values(false))); // Values(false) is the reserved parameter




#endif // HAVE_OPENCL
