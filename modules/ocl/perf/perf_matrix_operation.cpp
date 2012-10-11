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
////////////////////////////////converto/////////////////////////////////////////////////
PARAM_TEST_CASE(ConvertToTestBase, MatType, MatType)
{
    int type;
    int dst_type;

    //src mat
    cv::Mat mat;
    cv::Mat dst;

    // set up roi
    int roicols;
    int roirows;
    int srcx;
    int srcy;
    int dstx;
    int dsty;

    //src mat with roi
    cv::Mat mat_roi;
    cv::Mat dst_roi;
    //std::vector<cv::ocl::Info> oclinfo;
    //ocl dst mat for testing
    cv::ocl::oclMat gdst_whole;

    //ocl mat with roi
    cv::ocl::oclMat gmat;
    cv::ocl::oclMat gdst;

    virtual void SetUp()
    {
        type     = GET_PARAM(0);
        dst_type = GET_PARAM(1);

        cv::RNG &rng = TS::ptr()->get_rng();
        cv::Size size(MWIDTH, MHEIGHT);

        mat = randomMat(rng, size, type, 5, 16, false);
        dst  = randomMat(rng, size, type, 5, 16, false);
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
            srcy   = 1;
            dstx    = 1;
            dsty    = 1;
        }
        else
        {
            roicols = mat.cols;
            roirows = mat.rows;
            srcx   = 0;
            srcy   = 0;
            dstx   = 0;
            dsty   = 0;
        };

        mat_roi = mat(Rect(srcx, srcy, roicols, roirows));
        dst_roi  = dst(Rect(dstx, dsty, roicols, roirows));

        //gdst_whole = dst;
        //gdst = gdst_whole(Rect(dstx,dsty,roicols,roirows));

        //gmat = mat_roi;
    }
};


struct ConvertTo : ConvertToTestBase {};

TEST_P(ConvertTo, Accuracy)
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
            mat_roi.convertTo(dst_roi, dst_type);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            gdst_whole = dst;
            gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));

            gmat = mat_roi;
            t2 = (double)cvGetTickCount(); //kernel
            gmat.convertTo(gdst, dst_type);
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

        gmat = mat_roi;
        if(j == 0)
        {
            cout << "no roi:";
        }
        else
        {
            cout << "\nwith roi:";
        };
        gmat.convertTo(gdst, dst_type);
    };
#endif

}


///////////////////////////////////////////copyto/////////////////////////////////////////////////////////////

PARAM_TEST_CASE(CopyToTestBase, MatType, bool)
{
    int type;

    cv::Mat mat;
    cv::Mat mask;
    cv::Mat dst;

    // set up roi
    int roicols;
    int roirows;
    int srcx;
    int srcy;
    int dstx;
    int dsty;
    int maskx;
    int masky;

    //src mat with roi
    cv::Mat mat_roi;
    cv::Mat mask_roi;
    cv::Mat dst_roi;
    //std::vector<cv::ocl::Info> oclinfo;
    //ocl dst mat for testing
    cv::ocl::oclMat gdst_whole;

    //ocl mat with roi
    cv::ocl::oclMat gmat;
    cv::ocl::oclMat gdst;
    cv::ocl::oclMat gmask;

    virtual void SetUp()
    {
        type = GET_PARAM(0);

        cv::RNG &rng = TS::ptr()->get_rng();
        cv::Size size(MWIDTH, MHEIGHT);

        mat = randomMat(rng, size, type, 5, 16, false);
        dst  = randomMat(rng, size, type, 5, 16, false);
        mask = randomMat(rng, size, CV_8UC1, 0, 2,  false);

        cv::threshold(mask, mask, 0.5, 255., CV_8UC1);
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
            srcy   = 1;
            dstx    = 1;
            dsty    = 1;
            maskx   = 1;
            masky   = 1;
        }
        else
        {
            roicols = mat.cols;
            roirows = mat.rows;
            srcx   = 0;
            srcy   = 0;
            dstx   = 0;
            dsty   = 0;
            maskx   = 0;
            masky   = 0;
        };

        mat_roi = mat(Rect(srcx, srcy, roicols, roirows));
        mask_roi = mask(Rect(maskx, masky, roicols, roirows));
        dst_roi  = dst(Rect(dstx, dsty, roicols, roirows));

        //gdst_whole = dst;
        //gdst = gdst_whole(Rect(dstx,dsty,roicols,roirows));

        //gmat = mat_roi;
        //gmask = mask_roi;
    }
};

struct CopyTo : CopyToTestBase {};

TEST_P(CopyTo, Without_mask)
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
            mat_roi.copyTo(dst_roi);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            gdst_whole = dst;
            gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));

            gmat = mat_roi;
            t2 = (double)cvGetTickCount(); //kernel
            gmat.copyTo(gdst);
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

        gmat = mat_roi;

        if(j == 0)
        {
            cout << "no roi:";
        }
        else
        {
            cout << "\nwith roi:";
        };
        gmat.copyTo(gdst);
    };
#endif
}

TEST_P(CopyTo, With_mask)
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
            mat_roi.copyTo(dst_roi, mask_roi);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            gdst_whole = dst;
            gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));

            gmat = mat_roi;
            gmask = mask_roi;
            t2 = (double)cvGetTickCount(); //kernel
            gmat.copyTo(gdst, gmask);
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

        gmat = mat_roi;
        gmask = mask_roi;

        if(j == 0)
        {
            cout << "no roi:";
        }
        else
        {
            cout << "\nwith roi:";
        };
        gmat.copyTo(gdst, gmask);
    };
#endif
}

///////////////////////////////////////////copyto/////////////////////////////////////////////////////////////

PARAM_TEST_CASE(SetToTestBase, MatType, bool)
{
    int type;
    cv::Scalar val;

    cv::Mat mat;
    cv::Mat mask;

    // set up roi
    int roicols;
    int roirows;
    int srcx;
    int srcy;
    int maskx;
    int masky;

    //src mat with roi
    cv::Mat mat_roi;
    cv::Mat mask_roi;
    //std::vector<cv::ocl::Info> oclinfo;
    //ocl dst mat for testing
    cv::ocl::oclMat gmat_whole;

    //ocl mat with roi
    cv::ocl::oclMat gmat;
    cv::ocl::oclMat gmask;

    virtual void SetUp()
    {
        type = GET_PARAM(0);

        cv::RNG &rng = TS::ptr()->get_rng();
        cv::Size size(MWIDTH, MHEIGHT);

        mat = randomMat(rng, size, type, 5, 16, false);
        mask = randomMat(rng, size, CV_8UC1, 0, 2,  false);

        cv::threshold(mask, mask, 0.5, 255., CV_8UC1);
        val = cv::Scalar(rng.uniform(-10.0, 10.0), rng.uniform(-10.0, 10.0), rng.uniform(-10.0, 10.0), rng.uniform(-10.0, 10.0));
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
            srcy   = 1;
            maskx   = 1;
            masky   = 1;
        }
        else
        {
            roicols = mat.cols;
            roirows = mat.rows;
            srcx   = 0;
            srcy   = 0;
            maskx   = 0;
            masky   = 0;
        };

        mat_roi = mat(Rect(srcx, srcy, roicols, roirows));
        mask_roi = mask(Rect(maskx, masky, roicols, roirows));

        //gmat_whole = mat;
        //gmat = gmat_whole(Rect(srcx,srcy,roicols,roirows));

        //gmask = mask_roi;
    }
};

struct SetTo : SetToTestBase {};

TEST_P(SetTo, Without_mask)
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
            mat_roi.setTo(val);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            gmat_whole = mat;
            gmat = gmat_whole(Rect(srcx, srcy, roicols, roirows));
            t2 = (double)cvGetTickCount(); //kernel
            gmat.setTo(val);
            t2 = (double)cvGetTickCount() - t2;//kernel
            cv::Mat cpu_dst;
            gmat_whole.download(cpu_dst);//download
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
        gmat_whole = mat;
        gmat = gmat_whole(Rect(srcx, srcy, roicols, roirows));

        if(j == 0)
        {
            cout << "no roi:";
        }
        else
        {
            cout << "\nwith roi:";
        };
        gmat.setTo(val);
    };
#endif
}

TEST_P(SetTo, With_mask)
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
            mat_roi.setTo(val, mask_roi);
            t0 = (double)cvGetTickCount() - t0;//cpu end

            t1 = (double)cvGetTickCount();//gpu start1
            gmat_whole = mat;
            gmat = gmat_whole(Rect(srcx, srcy, roicols, roirows));

            gmask = mask_roi;
            t2 = (double)cvGetTickCount(); //kernel
            gmat.setTo(val, gmask);
            t2 = (double)cvGetTickCount() - t2;//kernel
            cv::Mat cpu_dst;
            gmat_whole.download(cpu_dst);//download
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
        gmat_whole = mat;
        gmat = gmat_whole(Rect(srcx, srcy, roicols, roirows));

        gmask = mask_roi;

        if(j == 0)
        {
            cout << "no roi:";
        }
        else
        {
            cout << "\nwith roi:";
        };
        gmat.setTo(val, gmask);
    };
#endif
}
PARAM_TEST_CASE(DataTransfer, MatType, bool)
{
    int type;
    cv::Mat mat;
    cv::ocl::oclMat gmat_whole;

    virtual void SetUp()
    {
        type = GET_PARAM(0);
        cv::RNG &rng = TS::ptr()->get_rng();
        cv::Size size(MWIDTH, MHEIGHT);
        mat = randomMat(rng, size, type, 5, 16, false);
    }
};
TEST_P(DataTransfer, perf)
{
    double totaluploadtick = 0;
    double totaldownloadtick = 0;
    double totaltick = 0;
    double t0 = 0;
    double t1 = 0;
    cv::Mat cpu_dst;
    for(int j = 0; j < LOOP_TIMES + 1; j ++)
    {
        t0 = (double)cvGetTickCount();
        gmat_whole.upload(mat);//upload
        t0 = (double)cvGetTickCount() - t0;

        t1 = (double)cvGetTickCount();
        gmat_whole.download(cpu_dst);//download
        t1 = (double)cvGetTickCount() - t1;

        if(j == 0)
            continue;
        totaluploadtick = t0 + totaluploadtick;
        totaldownloadtick = t1 + totaldownloadtick;
    }
    EXPECT_MAT_SIMILAR(mat, cpu_dst, 0.0);
    totaltick = totaluploadtick + totaldownloadtick;
    cout << "average upload time is  " << totaluploadtick / ((double)cvGetTickFrequency()* LOOP_TIMES * 1000.) << "ms" << endl;
    cout << "average download time is  " << totaldownloadtick / ((double)cvGetTickFrequency()* LOOP_TIMES * 1000.) << "ms" << endl;
    cout << "average data transfer time is  " << totaltick / ((double)cvGetTickFrequency()* LOOP_TIMES * 1000.) << "ms" << endl;
}
//**********test************

INSTANTIATE_TEST_CASE_P(MatrixOperation, ConvertTo, Combine(
                            Values(CV_8UC1, CV_8UC4, CV_32FC1, CV_32FC4),
                            Values(CV_8UC1, CV_8UC4, CV_32FC1, CV_32FC4)));

INSTANTIATE_TEST_CASE_P(MatrixOperation, CopyTo, Combine(
                            Values(CV_8UC1, CV_8UC4, CV_32FC1, CV_32FC4),
                            Values(false))); // Values(false) is the reserved parameter

INSTANTIATE_TEST_CASE_P(MatrixOperation, SetTo, Combine(
                            Values(CV_8UC1, CV_8UC4, CV_32FC1, CV_32FC4),
                            Values(false))); // Values(false) is the reserved parameter
INSTANTIATE_TEST_CASE_P(MatrixOperation, DataTransfer, Combine(
                            Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_32FC1, CV_32FC3, CV_32FC4),
                            Values(false))); // Values(false) is the reserved parameter
#endif
