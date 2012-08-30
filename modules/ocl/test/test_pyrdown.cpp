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
//    Dachuan Zhao, dachuan@multicorewareinc.com
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

//#define PRINT_CPU_TIME 1000
//#define PRINT_TIME


#include "precomp.hpp"
#include <iomanip>

#ifdef HAVE_OPENCL

using namespace cv;
using namespace cv::ocl;
using namespace cvtest;
using namespace testing;
using namespace std;

PARAM_TEST_CASE(PyrDown, MatType, bool)
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
        mat2 = randomMat(rng, size, type, 5, 16, false);
        dst  = randomMat(rng, size, type, 5, 16, false);
        dst1  = randomMat(rng, size, type, 5, 16, false);
        mask = randomMat(rng, size, CV_8UC1, 0, 2,  false);

        cv::threshold(mask, mask, 0.5, 255., CV_8UC1);

        val = cv::Scalar(rng.uniform(-10.0, 10.0), rng.uniform(-10.0, 10.0), rng.uniform(-10.0, 10.0), rng.uniform(-10.0, 10.0));

        //int devnums = getDevice(oclinfo);
        //CV_Assert(devnums > 0);
        ////if you want to use undefault device, set it here
        ////setDevice(oclinfo[0]);
    }

	void Cleanup()
	{
		mat1.release();
		mat2.release();
		mask.release();
		dst.release();
		dst1.release();
		mat1_roi.release();
		mat2_roi.release();
		mask_roi.release();
		dst_roi.release();
		dst1_roi.release();

		gdst_whole.release();
		gdst1_whole.release();
		gmat1.release();
		gmat2.release();
		gdst.release();
		gdst1.release();
		gmask.release();
	}

    void random_roi()
    {
        cv::RNG &rng = TS::ptr()->get_rng();

#ifdef RANDOMROI
        //randomize ROI
        roicols = rng.uniform(1, mat1.cols);
        roirows = rng.uniform(1, mat1.rows);
        src1x   = rng.uniform(0, mat1.cols - roicols);
        src1y   = rng.uniform(0, mat1.rows - roirows);
        dstx    = rng.uniform(0, dst.cols  - roicols);
        dsty    = rng.uniform(0, dst.rows  - roirows);
#else
        roicols = mat1.cols;
        roirows = mat1.rows;
        src1x = 0;
        src1y = 0;
        dstx = 0;
        dsty = 0;
#endif
        maskx   = rng.uniform(0, mask.cols - roicols);
        masky   = rng.uniform(0, mask.rows - roirows);
        src2x   = rng.uniform(0, mat2.cols - roicols);
        src2y   = rng.uniform(0, mat2.rows - roirows);
        mat1_roi = mat1(Rect(src1x, src1y, roicols, roirows));
        mat2_roi = mat2(Rect(src2x, src2y, roicols, roirows));
        mask_roi = mask(Rect(maskx, masky, roicols, roirows));
        dst_roi  = dst(Rect(dstx, dsty, roicols, roirows));
        dst1_roi = dst1(Rect(dstx, dsty, roicols, roirows));

        gdst_whole = dst;
        gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));

        gdst1_whole = dst1;
        gdst1 = gdst1_whole(Rect(dstx, dsty, roicols, roirows));

        gmat1 = mat1_roi;
        gmat2 = mat2_roi;
        gmask = mask_roi; //end
    }

};

#define VARNAME(A) string(#A);


void PrePrint()
{
		//for(int i = 0; i < MHEIGHT; i++)
		//{
		//	printf("(%d) ", i);
		//	for(int k = 0; k < MWIDTH; k++)
		//	{
		//		printf("%d ", mat1_roi.data[i * MHEIGHT + k]);
		//	}
		//	printf("\n");
		//}
}

void PostPrint()
{
		//dst_roi.convertTo(dst_roi,CV_32S);
		//cpu_dst.convertTo(cpu_dst,CV_32S);
		//dst_roi -= cpu_dst;
		//cpu_dst -= dst_roi;
		//for(int i = 0; i < MHEIGHT / 2; i++)
		//{
		//	printf("(%d) ", i);
		//	for(int k = 0; k < MWIDTH / 2; k++)
		//	{
		//		if(gmat1.depth() == 0)
		//		{
		//			if(gmat1.channels() == 1)
		//			{
		//				printf("%d ", dst_roi.data[i * MHEIGHT / 2 + k]);
		//			}
		//			else
		//			{
		//				printf("%d ", ((unsigned*)dst_roi.data)[i * MHEIGHT / 2 + k]);
		//			}
		//		}
		//		else if(gmat1.depth() == 5)
		//		{
		//			printf("%.6f ", ((float*)dst_roi.data)[i * MHEIGHT / 2 + k]);
		//		}
		//	}
		//	printf("\n");
		//}
		//for(int i = 0; i < MHEIGHT / 2; i++)
		//{
		//	printf("(%d) ", i);
		//	for(int k = 0; k < MWIDTH / 2; k++)
		//	{
		//		if(gmat1.depth() == 0)
		//		{
		//			if(gmat1.channels() == 1)
		//			{
		//				printf("%d ", cpu_dst.data[i * MHEIGHT / 2 + k]);
		//			}
		//			else
		//			{
		//				printf("%d ", ((unsigned*)cpu_dst.data)[i * MHEIGHT / 2 + k]);
		//			}
		//		}
		//		else if(gmat1.depth() == 5)
		//		{
		//			printf("%.6f ", ((float*)cpu_dst.data)[i * MHEIGHT / 2 + k]);
		//		}
		//	}
		//	printf("\n");
		//}
}

////////////////////////////////PyrDown/////////////////////////////////////////////////
//struct PyrDown : ArithmTestBase {};

TEST_P(PyrDown, Mat)
{
    for(int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

		cv::pyrDown(mat1_roi, dst_roi);
		cv::ocl::pyrDown(gmat1, gdst);

        cv::Mat cpu_dst;
        gdst.download(cpu_dst);
        char s[1024];
        sprintf(s, "roicols=%d,roirows=%d,src1x=%d,src1y=%d,dstx=%d,dsty=%d,maskx=%d,masky=%d,src2x=%d,src2y=%d", roicols, roirows, src1x, src1y, dstx, dsty, maskx, masky, src2x, src2y);

		EXPECT_MAT_NEAR(dst_roi, cpu_dst, dst_roi.depth() == CV_32F ? 1e-5f : 1.0f, s);

		Cleanup();
    }
}




//********test****************
INSTANTIATE_TEST_CASE_P(GPU_ImgProc, PyrDown, Combine(
                            Values(CV_8UC1, CV_8UC4, CV_32FC1, CV_32FC4),
                            Values(false))); // Values(false) is the reserved parameter


#endif // HAVE_OPENCL
