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
//    Zero Lin, Zero.Lin@amd.com
//    Zhang Ying, zhangying913@gmail.com
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
//using namespace cv::ocl;

PARAM_TEST_CASE(FilterTestBase, MatType, bool)
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
	std::vector<cv::ocl::Info> oclinfo;
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

		cv::RNG& rng = TS::ptr()->get_rng();
		cv::Size size(MWIDTH, MHEIGHT);

		mat1 = randomMat(rng, size, type, 5, 16, false);
		mat2 = randomMat(rng, size, type, 5, 16, false);
		dst  = randomMat(rng, size, type, 5, 16, false);
		dst1  = randomMat(rng, size, type, 5, 16, false);
		mask = randomMat(rng, size, CV_8UC1, 0, 2,  false);

		cv::threshold(mask, mask, 0.5, 255., CV_8UC1);

		val = cv::Scalar(rng.uniform(-10.0, 10.0), rng.uniform(-10.0, 10.0), rng.uniform(-10.0, 10.0), rng.uniform(-10.0, 10.0));
	}

	void random_roi()
	{
		cv::RNG& rng = TS::ptr()->get_rng();

		//randomize ROI
		roicols = rng.uniform(1, mat1.cols);
		roirows = rng.uniform(1, mat1.rows);
		src1x   = rng.uniform(0, mat1.cols - roicols);
		src1y   = rng.uniform(0, mat1.rows - roirows);
		src2x   = rng.uniform(0, mat2.cols - roicols);
		src2y   = rng.uniform(0, mat2.rows - roirows);
		dstx    = rng.uniform(0, dst.cols  - roicols);
		dsty    = rng.uniform(0, dst.rows  - roirows);
		maskx   = rng.uniform(0, mask.cols - roicols);
		masky   = rng.uniform(0, mask.rows - roirows);

		mat1_roi = mat1(Rect(src1x,src1y,roicols,roirows));
		mat2_roi = mat2(Rect(src2x,src2y,roicols,roirows));
		mask_roi = mask(Rect(maskx,masky,roicols,roirows));
		dst_roi  = dst(Rect(dstx,dsty,roicols,roirows));
		dst1_roi = dst1(Rect(dstx,dsty,roicols,roirows));

		gdst_whole = dst;
		gdst = gdst_whole(Rect(dstx,dsty,roicols,roirows));

		gdst1_whole = dst1;
		gdst1 = gdst1_whole(Rect(dstx,dsty,roicols,roirows));

		gmat1 = mat1_roi;
		gmat2 = mat2_roi;
		gmask = mask_roi;
	}

};

/////////////////////////////////////////////////////////////////////////////////////////////////
// blur

PARAM_TEST_CASE(Blur, MatType, cv::Size, int)
{
	int type;
	cv::Size ksize;
	int bordertype;

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
	std::vector<cv::ocl::Info> oclinfo;
	//ocl dst mat for testing
	cv::ocl::oclMat gdst_whole;

	//ocl mat with roi
	cv::ocl::oclMat gmat1;
	cv::ocl::oclMat gdst;

	virtual void SetUp()
	{
		type = GET_PARAM(0);
		ksize = GET_PARAM(1);
		bordertype = GET_PARAM(2);

		cv::RNG& rng = TS::ptr()->get_rng();
		cv::Size size(MWIDTH, MHEIGHT);

		mat1 = randomMat(rng, size, type, 5, 16, false);
		dst  = randomMat(rng, size, type, 5, 16, false);
		int devnums = getDevice(oclinfo);
		CV_Assert(devnums > 0);
		//if you want to use undefault device, set it here
		//setDevice(oclinfo[0]);
		cv::ocl::setBinpath(CLBINPATH);
	}


	void Has_roi(int b)
	{
		if(b)
		{
			roicols =  mat1.cols-1; 
			roirows = mat1.rows-1;
			src1x   = 1;
			src1y   = 1;
			dstx    = 1;
			dsty    =1;
		}else
		{
			roicols = mat1.cols;
			roirows = mat1.rows;
			src1x = 0;
			src1y = 0;
			dstx = 0;
			dsty = 0;
		};

		mat1_roi = mat1(Rect(src1x,src1y,roicols,roirows));
		dst_roi  = dst(Rect(dstx,dsty,roicols,roirows));

	}

};

TEST_P(Blur, Mat)
{
#ifndef PRINT_KERNEL_RUN_TIME   
	double totalcputick=0;
	double totalgputick=0;
	double totalgputick_kernel=0;
	double t0=0;
	double t1=0;
	double t2=0;	
	for(int k=0;k<2;k++){
		totalcputick=0;
		totalgputick=0;
		totalgputick_kernel=0;
		for(int j = 0; j < LOOP_TIMES+1; j ++)
		{
			Has_roi(k);       

			t0 = (double)cvGetTickCount();//cpu start
			cv::blur(mat1_roi, dst_roi, ksize, Point(-1,-1), bordertype);
			t0 = (double)cvGetTickCount() - t0;//cpu end

			t1 = (double)cvGetTickCount();//gpu start1		
			gdst_whole = dst;
			gdst = gdst_whole(Rect(dstx,dsty,roicols,roirows));

			gmat1 = mat1_roi;
			t2=(double)cvGetTickCount();//kernel
			cv::ocl::blur(gmat1, gdst, ksize, Point(-1,-1), bordertype);
			t2 = (double)cvGetTickCount() - t2;//kernel
			cv::Mat cpu_dst;
			gdst_whole.download (cpu_dst);//download
			t1 = (double)cvGetTickCount() - t1;//gpu end1	

			if(j == 0)
				continue;

			totalgputick=t1+totalgputick;
			totalcputick=t0+totalcputick;	
			totalgputick_kernel=t2+totalgputick_kernel;	

		}
		if(k==0){cout<<"no roi\n";}else{cout<<"with roi\n";};
		cout << "average cpu runtime is  " << totalcputick/((double)cvGetTickFrequency()* LOOP_TIMES *1000.) << "ms" << endl;
		cout << "average gpu runtime is  " << totalgputick/((double)cvGetTickFrequency()* LOOP_TIMES *1000.) << "ms" << endl;
		cout << "average gpu runtime without data transfer is  " << totalgputick_kernel/((double)cvGetTickFrequency()* LOOP_TIMES *1000.) << "ms" << endl;
	}
#else
	for(int j = 0; j < 2; j ++)
	{
		Has_roi(j);
		gdst_whole = dst;
		gdst = gdst_whole(Rect(dstx,dsty,roicols,roirows));
		gmat1 = mat1_roi;
		if(j==0){cout<<"no roi:";}else{cout<<"\nwith roi:";};
		cv::ocl::blur(gmat1, gdst, ksize, Point(-1,-1), bordertype);
	};
#endif

}

/////////////////////////////////////////////////////////////////////////////////////////////////
//Laplacian 

PARAM_TEST_CASE(LaplacianTestBase, MatType, int)
{
	int type;
	int ksize;

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
	std::vector<cv::ocl::Info> oclinfo;
	//ocl dst mat for testing
	cv::ocl::oclMat gdst_whole;

	//ocl mat with roi
	cv::ocl::oclMat gmat;
	cv::ocl::oclMat gdst;

	virtual void SetUp()
	{
		type = GET_PARAM(0);
		ksize = GET_PARAM(1);

		cv::RNG& rng = TS::ptr()->get_rng();
		cv::Size size = cv::Size(2560, 2560);

		mat  = randomMat(rng, size, type, 5, 16, false);
		dst  = randomMat(rng, size, type, 5, 16, false);
		int devnums = getDevice(oclinfo);
		CV_Assert(devnums > 0);
		//if you want to use undefault device, set it here
		//setDevice(oclinfo[0]);
		cv::ocl::setBinpath(CLBINPATH);
	}

	void Has_roi(int b)
	{
		if(b)
		{
			roicols =  mat.cols-1; 
			roirows = mat.rows-1;
			srcx   = 1;
			srcy   = 1;
			dstx    = 1;
			dsty    =1;
		}else
		{
			roicols = mat.cols;
			roirows = mat.rows;
			srcx = 0;
			srcy = 0;
			dstx = 0;
			dsty = 0;
		};

		mat_roi = mat(Rect(srcx,srcy,roicols,roirows));
		dst_roi  = dst(Rect(dstx,dsty,roicols,roirows));

	}

};

struct Laplacian : LaplacianTestBase {};

TEST_P(Laplacian, Accuracy) 
{    

#ifndef PRINT_KERNEL_RUN_TIME   
	double totalcputick=0;
	double totalgputick=0;
	double totalgputick_kernel=0;
	double t0=0;
	double t1=0;
	double t2=0;	
	for(int k=0;k<2;k++){
		totalcputick=0;
		totalgputick=0;
		totalgputick_kernel=0;
		for(int j = 0; j < LOOP_TIMES+1; j ++)
		{
			Has_roi(k);       

			t0 = (double)cvGetTickCount();//cpu start
			cv::Laplacian(mat_roi, dst_roi, -1, ksize, 1);
			t0 = (double)cvGetTickCount() - t0;//cpu end

			t1 = (double)cvGetTickCount();//gpu start1		
			gdst_whole = dst;
			gdst = gdst_whole(Rect(dstx,dsty,roicols,roirows));

			gmat = mat_roi;
			t2=(double)cvGetTickCount();//kernel
			cv::ocl::Laplacian(gmat, gdst, -1, ksize, 1);
			t2 = (double)cvGetTickCount() - t2;//kernel
			cv::Mat cpu_dst;
			gdst_whole.download (cpu_dst);//download
			t1 = (double)cvGetTickCount() - t1;//gpu end1	

			if(j == 0)
				continue;

			totalgputick=t1+totalgputick;
			totalcputick=t0+totalcputick;	
			totalgputick_kernel=t2+totalgputick_kernel;	

		}
		if(k==0){cout<<"no roi\n";}else{cout<<"with roi\n";};
		cout << "average cpu runtime is  " << totalcputick/((double)cvGetTickFrequency()* LOOP_TIMES *1000.) << "ms" << endl;
		cout << "average gpu runtime is  " << totalgputick/((double)cvGetTickFrequency()* LOOP_TIMES *1000.) << "ms" << endl;
		cout << "average gpu runtime without data transfer is  " << totalgputick_kernel/((double)cvGetTickFrequency()* LOOP_TIMES *1000.) << "ms" << endl;
	}
#else
	for(int j = 0; j < 2; j ++)
	{
		Has_roi(j);
		gdst_whole = dst;
		gdst = gdst_whole(Rect(dstx,dsty,roicols,roirows));
		gmat = mat_roi;


		if(j==0){cout<<"no roi:";}else{cout<<"\nwith roi:";};
		cv::ocl::Laplacian(gmat, gdst, -1, ksize, 1);
	};
#endif
}


/////////////////////////////////////////////////////////////////////////////////////////////////
// erode & dilate 

PARAM_TEST_CASE(ErodeDilateBase, MatType, bool)
{
	int type;
	//int iterations;

	//erode or dilate kernel
	cv::Mat kernel;

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
	std::vector<cv::ocl::Info> oclinfo;
	//ocl dst mat for testing
	cv::ocl::oclMat gdst_whole;

	//ocl mat with roi
	cv::ocl::oclMat gmat1;
	cv::ocl::oclMat gdst;

	virtual void SetUp()
	{
		type = GET_PARAM(0);
		//  iterations = GET_PARAM(1);

		cv::RNG& rng = TS::ptr()->get_rng();
		cv::Size size = cv::Size(2560, 2560);

		mat1 = randomMat(rng, size, type, 5, 16, false);
		dst  = randomMat(rng, size, type, 5, 16, false);
		//		rng.fill(kernel, cv::RNG::UNIFORM, cv::Scalar::all(0), cv::Scalar::all(3));
		kernel = randomMat(rng, Size(3,3), CV_8UC1, 0, 3, false);
		int devnums = getDevice(oclinfo);
		CV_Assert(devnums > 0);
		//if you want to use undefault device, set it here
		//setDevice(oclinfo[0]);
		cv::ocl::setBinpath(CLBINPATH);
	}

	void Has_roi(int b)
	{
		if(b)
		{
			roicols =  mat1.cols-1; 
			roirows = mat1.rows-1;
			src1x   = 1;
			src1y   = 1;
			dstx    = 1;
			dsty    =1;
		}else
		{
			roicols = mat1.cols;
			roirows = mat1.rows;
			src1x = 0;
			src1y = 0;
			dstx = 0;
			dsty = 0;
		};

		mat1_roi = mat1(Rect(src1x,src1y,roicols,roirows));
		dst_roi  = dst(Rect(dstx,dsty,roicols,roirows));

	}

};

// erode 

struct Erode : ErodeDilateBase{};

TEST_P(Erode, Mat)
{
#ifndef PRINT_KERNEL_RUN_TIME   
	double totalcputick=0;
	double totalgputick=0;
	double totalgputick_kernel=0;
	double t0=0;
	double t1=0;
	double t2=0;	
	for(int k=0;k<2;k++){
		totalcputick=0;
		totalgputick=0;
		totalgputick_kernel=0;
		for(int j = 0; j < LOOP_TIMES+1; j ++)
		{
			Has_roi(k);       

			t0 = (double)cvGetTickCount();//cpu start
			cv::erode(mat1_roi, dst_roi, kernel);
			t0 = (double)cvGetTickCount() - t0;//cpu end

			t1 = (double)cvGetTickCount();//gpu start1		
			gdst_whole = dst;
			gdst = gdst_whole(Rect(dstx,dsty,roicols,roirows));

			gmat1 = mat1_roi;

			t2=(double)cvGetTickCount();//kernel
			cv::ocl::erode(gmat1, gdst, kernel);
			t2 = (double)cvGetTickCount() - t2;//kernel
			cv::Mat cpu_dst;
			gdst_whole.download (cpu_dst);//download
			t1 = (double)cvGetTickCount() - t1;//gpu end1	

			if(j == 0)
				continue;

			totalgputick=t1+totalgputick;
			totalcputick=t0+totalcputick;	
			totalgputick_kernel=t2+totalgputick_kernel;	

		}
		if(k==0){cout<<"no roi\n";}else{cout<<"with roi\n";};
		cout << "average cpu runtime is  " << totalcputick/((double)cvGetTickFrequency()* LOOP_TIMES *1000.) << "ms" << endl;
		cout << "average gpu runtime is  " << totalgputick/((double)cvGetTickFrequency()* LOOP_TIMES *1000.) << "ms" << endl;
		cout << "average gpu runtime without data transfer is  " << totalgputick_kernel/((double)cvGetTickFrequency()* LOOP_TIMES *1000.) << "ms" << endl;
	}
#else
	for(int j = 0; j < 2; j ++)
	{
		Has_roi(j);
		gdst_whole = dst;
		gdst = gdst_whole(Rect(dstx,dsty,roicols,roirows));
		gmat1 = mat1_roi;

		if(j==0){cout<<"no roi:";}else{cout<<"\nwith roi:";};
		cv::ocl::erode(gmat1, gdst, kernel);
	};
#endif

}

// dilate

struct Dilate : ErodeDilateBase{};

TEST_P(Dilate, Mat)
{

#ifndef PRINT_KERNEL_RUN_TIME   
	double totalcputick=0;
	double totalgputick=0;
	double totalgputick_kernel=0;
	double t0=0;
	double t1=0;
	double t2=0;	
	for(int k=0;k<2;k++){
		totalcputick=0;
		totalgputick=0;
		totalgputick_kernel=0;
		for(int j = 0; j < LOOP_TIMES+1; j ++)
		{
			Has_roi(k);       
			t0 = (double)cvGetTickCount();//cpu start
			cv::dilate(mat1_roi, dst_roi, kernel);
			t0 = (double)cvGetTickCount() - t0;//cpu end

			t1 = (double)cvGetTickCount();//gpu start1		
			gdst_whole = dst;
			gdst = gdst_whole(Rect(dstx,dsty,roicols,roirows));

			gmat1 = mat1_roi;
			t2=(double)cvGetTickCount();//kernel
			cv::ocl::dilate(gmat1, gdst, kernel);
			t2 = (double)cvGetTickCount() - t2;//kernel
			cv::Mat cpu_dst;
			gdst_whole.download (cpu_dst);//download
			t1 = (double)cvGetTickCount() - t1;//gpu end1		

			if(j == 0)
				continue;

			totalgputick=t1+totalgputick;
			totalcputick=t0+totalcputick;	
			totalgputick_kernel=t2+totalgputick_kernel;	

		}
		if(k==0){cout<<"no roi\n";}else{cout<<"with roi\n";};
		cout << "average cpu runtime is  " << totalcputick/((double)cvGetTickFrequency()* LOOP_TIMES *1000.) << "ms" << endl;
		cout << "average gpu runtime is  " << totalgputick/((double)cvGetTickFrequency()* LOOP_TIMES *1000.) << "ms" << endl;
		cout << "average gpu runtime without data transfer is  " << totalgputick_kernel/((double)cvGetTickFrequency()* LOOP_TIMES *1000.) << "ms" << endl;
	}
#else
	for(int j = 0; j < 2; j ++)
	{
		Has_roi(j);
		gdst_whole = dst;
		gdst = gdst_whole(Rect(dstx,dsty,roicols,roirows));
		gmat1 = mat1_roi;
		if(j==0){cout<<"no roi:";}else{cout<<"\nwith roi:";};
		cv::ocl::dilate(gmat1, gdst, kernel);
	};
#endif

}

/////////////////////////////////////////////////////////////////////////////////////////////////
// Sobel 

PARAM_TEST_CASE(Sobel, MatType, int, int, int, int)
{
	int type;
	int dx, dy, ksize, bordertype;

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
	std::vector<cv::ocl::Info> oclinfo;
	//ocl dst mat for testing
	cv::ocl::oclMat gdst_whole;

	//ocl mat with roi
	cv::ocl::oclMat gmat1;
	cv::ocl::oclMat gdst;

	virtual void SetUp()
	{
		type = GET_PARAM(0);
		dx = GET_PARAM(1);
		dy = GET_PARAM(2);
		ksize = GET_PARAM(3);
		bordertype = GET_PARAM(4);
		dx = 2; dy=0;

		cv::RNG& rng = TS::ptr()->get_rng();
		cv::Size size = cv::Size(2560, 2560);

		mat1 = randomMat(rng, size, type, 5, 16, false);
		dst  = randomMat(rng, size, type, 5, 16, false);
		int devnums = getDevice(oclinfo);
		CV_Assert(devnums > 0);
		//if you want to use undefault device, set it here
		//setDevice(oclinfo[0]);
		cv::ocl::setBinpath(CLBINPATH);
	}

	void Has_roi(int b)
	{
		if(b)
		{
			roicols =  mat1.cols-1; 
			roirows = mat1.rows-1;
			src1x   = 1;
			src1y   = 1;
			dstx    = 1;
			dsty    =1;
		}else
		{
			roicols = mat1.cols;
			roirows = mat1.rows;
			src1x = 0;
			src1y = 0;
			dstx = 0;
			dsty = 0;
		};

		mat1_roi = mat1(Rect(src1x,src1y,roicols,roirows));
		dst_roi  = dst(Rect(dstx,dsty,roicols,roirows));

	}

};

TEST_P(Sobel, Mat)
{
#ifndef PRINT_KERNEL_RUN_TIME   
	double totalcputick=0;
	double totalgputick=0;
	double totalgputick_kernel=0;
	double t0=0;
	double t1=0;
	double t2=0;	
	for(int k=0;k<2;k++){
		totalcputick=0;
		totalgputick=0;
		totalgputick_kernel=0;
		for(int j = 0; j < LOOP_TIMES+1; j ++)
		{
			Has_roi(k);       

			t0 = (double)cvGetTickCount();//cpu start
			cv::Sobel(mat1_roi, dst_roi, -1, dx, dy, ksize, /*scale*/0.00001,/*delta*/0, bordertype);
			t0 = (double)cvGetTickCount() - t0;//cpu end

			t1 = (double)cvGetTickCount();//gpu start1		
			gdst_whole = dst;
			gdst = gdst_whole(Rect(dstx,dsty,roicols,roirows));

			gmat1 = mat1_roi;
			t2=(double)cvGetTickCount();//kernel
			cv::ocl::Sobel(gmat1, gdst,-1, dx,dy,ksize,/*scale*/0.00001,/*delta*/0, bordertype);
			t2 = (double)cvGetTickCount() - t2;//kernel
			cv::Mat cpu_dst;
			gdst_whole.download (cpu_dst);//download
			t1 = (double)cvGetTickCount() - t1;//gpu end1	

			if(j == 0)
				continue;

			totalgputick=t1+totalgputick;
			totalcputick=t0+totalcputick;	
			totalgputick_kernel=t2+totalgputick_kernel;	

		}
		if(k==0){cout<<"no roi\n";}else{cout<<"with roi\n";};
		cout << "average cpu runtime is  " << totalcputick/((double)cvGetTickFrequency()* LOOP_TIMES *1000.) << "ms" << endl;
		cout << "average gpu runtime is  " << totalgputick/((double)cvGetTickFrequency()* LOOP_TIMES *1000.) << "ms" << endl;
		cout << "average gpu runtime without data transfer is  " << totalgputick_kernel/((double)cvGetTickFrequency()* LOOP_TIMES *1000.) << "ms" << endl;
	}
#else
	for(int j = 0; j < 2; j ++)
	{
		Has_roi(j);
		gdst_whole = dst;
		gdst = gdst_whole(Rect(dstx,dsty,roicols,roirows));
		gmat1 = mat1_roi;
		if(j==0){cout<<"no roi:";}else{cout<<"\nwith roi:";};
		cv::ocl::Sobel(gmat1, gdst,-1, dx,dy,ksize,/*scale*/0.00001,/*delta*/0, bordertype);
	};
#endif

}

/////////////////////////////////////////////////////////////////////////////////////////////////
// Scharr 

PARAM_TEST_CASE(Scharr, MatType, int, int, int)
{
	int type;
	int dx, dy, bordertype;

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
	std::vector<cv::ocl::Info> oclinfo;
	//ocl dst mat for testing
	cv::ocl::oclMat gdst_whole;

	//ocl mat with roi
	cv::ocl::oclMat gmat1;
	cv::ocl::oclMat gdst;

	virtual void SetUp()
	{
		type = GET_PARAM(0);
		dx = GET_PARAM(1);
		dy = GET_PARAM(2);
		bordertype = GET_PARAM(3);
		dx = 1; dy=0;

		cv::RNG& rng = TS::ptr()->get_rng();
		cv::Size size = cv::Size(2560, 2560);

		mat1 = randomMat(rng, size, type, 5, 16, false);
		dst  = randomMat(rng, size, type, 5, 16, false);
		int devnums = getDevice(oclinfo);
		CV_Assert(devnums > 0);
		//if you want to use undefault device, set it here
		//setDevice(oclinfo[0]);
		cv::ocl::setBinpath(CLBINPATH);
	}

	void Has_roi(int b)
	{
		if(b)
		{
			roicols =  mat1.cols-1; 
			roirows = mat1.rows-1;
			src1x   = 1;
			src1y   = 1;
			dstx    = 1;
			dsty    =1;
		}else
		{
			roicols = mat1.cols;
			roirows = mat1.rows;
			src1x = 0;
			src1y = 0;
			dstx = 0;
			dsty = 0;
		};

		mat1_roi = mat1(Rect(src1x,src1y,roicols,roirows));
		dst_roi  = dst(Rect(dstx,dsty,roicols,roirows));

	}
};

TEST_P(Scharr, Mat)
{
#ifndef PRINT_KERNEL_RUN_TIME   
	double totalcputick=0;
	double totalgputick=0;
	double totalgputick_kernel=0;
	double t0=0;
	double t1=0;
	double t2=0;	
	for(int k=0;k<2;k++){
		totalcputick=0;
		totalgputick=0;
		totalgputick_kernel=0;
		for(int j = 0; j < LOOP_TIMES+1; j ++)
		{
			Has_roi(k);       

			t0 = (double)cvGetTickCount();//cpu start
			cv::Scharr(mat1_roi, dst_roi, -1, dx, dy, /*scale*/1,/*delta*/0, bordertype);
			t0 = (double)cvGetTickCount() - t0;//cpu end

			t1 = (double)cvGetTickCount();//gpu start1		
			gdst_whole = dst;
			gdst = gdst_whole(Rect(dstx,dsty,roicols,roirows));

			gmat1 = mat1_roi;
			t2=(double)cvGetTickCount();//kernel
			cv::ocl::Scharr(gmat1, gdst,-1, dx,dy,/*scale*/1,/*delta*/0, bordertype);
			t2 = (double)cvGetTickCount() - t2;//kernel
			cv::Mat cpu_dst;
			gdst_whole.download (cpu_dst);//download
			t1 = (double)cvGetTickCount() - t1;//gpu end1		

			if(j == 0)
				continue;

			totalgputick=t1+totalgputick;
			totalcputick=t0+totalcputick;	
			totalgputick_kernel=t2+totalgputick_kernel;	

		}
		if(k==0){cout<<"no roi\n";}else{cout<<"with roi\n";};
		cout << "average cpu runtime is  " << totalcputick/((double)cvGetTickFrequency()* LOOP_TIMES *1000.) << "ms" << endl;
		cout << "average gpu runtime is  " << totalgputick/((double)cvGetTickFrequency()* LOOP_TIMES *1000.) << "ms" << endl;
		cout << "average gpu runtime without data transfer is  " << totalgputick_kernel/((double)cvGetTickFrequency()* LOOP_TIMES *1000.) << "ms" << endl;
	}
#else
	for(int j = 0; j < 2; j ++)
	{
		Has_roi(j);
		gdst_whole = dst;
		gdst = gdst_whole(Rect(dstx,dsty,roicols,roirows));
		gmat1 = mat1_roi;

		if(j==0){cout<<"no roi:";}else{cout<<"\nwith roi:";};
		cv::ocl::Scharr(gmat1, gdst,-1, dx,dy,/*scale*/1,/*delta*/0, bordertype);
	};
#endif

}

/////////////////////////////////////////////////////////////////////////////////////////////////
// GaussianBlur

PARAM_TEST_CASE(GaussianBlur, MatType, cv::Size, int)
{
	int type;
	cv::Size ksize;
	int bordertype;

	double sigma1, sigma2;

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
	std::vector<cv::ocl::Info> oclinfo;
	//ocl dst mat for testing
	cv::ocl::oclMat gdst_whole;

	//ocl mat with roi
	cv::ocl::oclMat gmat1;
	cv::ocl::oclMat gdst;

	virtual void SetUp()
	{
		type = GET_PARAM(0);
		ksize = GET_PARAM(1);
		bordertype = GET_PARAM(2);

		cv::RNG& rng = TS::ptr()->get_rng();
		cv::Size size = cv::Size(2560, 2560);

		sigma1 = rng.uniform(0.1, 1.0); 
		sigma2 = rng.uniform(0.1, 1.0);

		mat1 = randomMat(rng, size, type, 5, 16, false);
		dst  = randomMat(rng, size, type, 5, 16, false);
		int devnums = getDevice(oclinfo);
		CV_Assert(devnums > 0);
		//if you want to use undefault device, set it here
		//setDevice(oclinfo[0]);
		cv::ocl::setBinpath(CLBINPATH);
	}

	void Has_roi(int b)
	{
		if(b)
		{
			roicols =  mat1.cols-1; 
			roirows = mat1.rows-1;
			src1x   = 1;
			src1y   = 1;
			dstx    = 1;
			dsty    =1;
		}else
		{
			roicols = mat1.cols;
			roirows = mat1.rows;
			src1x = 0;
			src1y = 0;
			dstx = 0;
			dsty = 0;
		};

		mat1_roi = mat1(Rect(src1x,src1y,roicols,roirows));
		dst_roi  = dst(Rect(dstx,dsty,roicols,roirows));

	}

};

TEST_P(GaussianBlur, Mat)
{
#ifndef PRINT_KERNEL_RUN_TIME   
	double totalcputick=0;
	double totalgputick=0;
	double totalgputick_kernel=0;
	double t0=0;
	double t1=0;
	double t2=0;	
	for(int k=0;k<2;k++){
		totalcputick=0;
		totalgputick=0;
		totalgputick_kernel=0;
		for(int j = 0; j < LOOP_TIMES+1; j ++)
		{
			Has_roi(k);       

			t0 = (double)cvGetTickCount();//cpu start
			cv::GaussianBlur(mat1_roi, dst_roi, ksize, sigma1, sigma2, bordertype);
			t0 = (double)cvGetTickCount() - t0;//cpu end

			t1 = (double)cvGetTickCount();//gpu start1		
			gdst_whole = dst;
			gdst = gdst_whole(Rect(dstx,dsty,roicols,roirows));

			gmat1 = mat1_roi;
			t2=(double)cvGetTickCount();//kernel
			cv::ocl::GaussianBlur(gmat1, gdst, ksize, sigma1, sigma2, bordertype);
			t2 = (double)cvGetTickCount() - t2;//kernel
			cv::Mat cpu_dst;
			gdst_whole.download (cpu_dst);//download
			t1 = (double)cvGetTickCount() - t1;//gpu end1	

			if(j == 0)
				continue;

			totalgputick=t1+totalgputick;
			totalcputick=t0+totalcputick;	
			totalgputick_kernel=t2+totalgputick_kernel;	


		}
		if(k==0){cout<<"no roi\n";}else{cout<<"with roi\n";};
		cout << "average cpu runtime is  " << totalcputick/((double)cvGetTickFrequency()* LOOP_TIMES *1000.) << "ms" << endl;
		cout << "average gpu runtime is  " << totalgputick/((double)cvGetTickFrequency()* LOOP_TIMES *1000.) << "ms" << endl;
		cout << "average gpu runtime without data transfer is  " << totalgputick_kernel/((double)cvGetTickFrequency()* LOOP_TIMES *1000.) << "ms" << endl;
	}
#else
	for(int j = 0; j < 2; j ++)
	{
		Has_roi(j);
		gdst_whole = dst;
		gdst = gdst_whole(Rect(dstx,dsty,roicols,roirows));
		gmat1 = mat1_roi;
		if(j==0){cout<<"no roi:";}else{cout<<"\nwith roi:";};
		cv::ocl::GaussianBlur(gmat1, gdst, ksize, sigma1, sigma2, bordertype);
	};
#endif

}

//************test**********

INSTANTIATE_TEST_CASE_P(Filter, Blur, Combine(Values(CV_8UC1, CV_8UC4, CV_32FC1, CV_32FC4),
						Values(cv::Size(3, 3)/*, cv::Size(5, 5), cv::Size(7, 7)*/),
						Values((MatType)cv::BORDER_CONSTANT, (MatType)cv::BORDER_REPLICATE, (MatType)cv::BORDER_REFLECT, (MatType)cv::BORDER_REFLECT_101)));


INSTANTIATE_TEST_CASE_P(Filters, Laplacian, Combine(
						Values(CV_8UC1, CV_8UC4, CV_32FC1, CV_32FC4),
						Values(1/*, 3*/)));

//INSTANTIATE_TEST_CASE_P(Filter, ErodeDilate, Combine(Values(CV_8UC1, CV_8UC4, CV_32FC1, CV_32FC4), Values(1, 2, 3)));

INSTANTIATE_TEST_CASE_P(Filter, Erode, Combine(Values(CV_8UC1, CV_8UC4, CV_32FC1, CV_32FC4), Values(false)));

//INSTANTIATE_TEST_CASE_P(Filter, ErodeDilate, Combine(Values(CV_8UC1, CV_8UC4, CV_32FC1, CV_32FC4), Values(1, 2, 3)));

INSTANTIATE_TEST_CASE_P(Filter, Dilate, Combine(Values(CV_8UC1, CV_8UC4, CV_32FC1, CV_32FC4), Values(false)));


INSTANTIATE_TEST_CASE_P(Filter, Sobel, Combine(Values(CV_8UC1, CV_8UC4, CV_32FC1, CV_32FC4),
						Values(1, 2), Values(0, 1), Values(3, 5, 7), Values((MatType)cv::BORDER_CONSTANT,
						(MatType)cv::BORDER_REPLICATE, (MatType)cv::BORDER_REFLECT, (MatType)cv::BORDER_REFLECT_101)));


INSTANTIATE_TEST_CASE_P(Filter, Scharr, Combine(
						Values(CV_8UC1, CV_8UC4, CV_32FC1, CV_32FC4), Values(0, 1), Values(0, 1),
						Values((MatType)cv::BORDER_CONSTANT, (MatType)cv::BORDER_REPLICATE, (MatType)cv::BORDER_REFLECT, (MatType)cv::BORDER_REFLECT_101)));

INSTANTIATE_TEST_CASE_P(Filter, GaussianBlur, Combine(
						Values(CV_8UC1, CV_8UC4, CV_32FC1, CV_32FC4),
						Values(cv::Size(3, 3), cv::Size(5, 5), cv::Size(7, 7)),
						Values((MatType)cv::BORDER_CONSTANT, (MatType)cv::BORDER_REPLICATE, (MatType)cv::BORDER_REFLECT, (MatType)cv::BORDER_REFLECT_101)));


#endif // HAVE_OPENCL
