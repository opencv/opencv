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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
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
#include "opencv2/photo.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <stdlib.h>

#include "npr.hpp"

using namespace std;
using namespace cv;

void cv::edgePreservingFilter(InputArray _src, OutputArray _dst, int flags, float sigma_s, float sigma_r)
{
	Mat I = _src.getMat();
	_dst.create(I.size(), CV_8UC3);
	Mat dst = _dst.getMat();

	int h = I.size().height;
	int w = I.size().width;

	Mat res = Mat(h,w,CV_32FC3);
	dst.convertTo(res,CV_32FC3,1.0/255.0);

	Domain_Filter obj;

	Mat img = Mat(I.size(),CV_32FC3);
	I.convertTo(img,CV_32FC3,1.0/255.0);

	obj.filter(img, res, sigma_s, sigma_r, flags);

	convertScaleAbs(res, dst, 255,0);
}

void cv::detailEnhance(InputArray _src, OutputArray _dst, float sigma_s, float sigma_r)
{
	Mat I = _src.getMat();
	_dst.create(I.size(), CV_8UC3);
	Mat dst = _dst.getMat();

	int h = I.size().height;
	int w = I.size().width;
	int channel = I.channels();
	float factor = 3.0;

	Mat img = Mat(I.size(),CV_32FC3);
	I.convertTo(img,CV_32FC3,1.0/255.0);
	
	Mat res = Mat(h,w,CV_32FC3);
	dst.convertTo(res,CV_32FC3,1.0/255.0);

	Mat result = Mat(img.size(),CV_32FC3);
	Mat lab = Mat(img.size(),CV_32FC3);
	Mat l_channel = Mat(img.size(),CV_32FC1);
	Mat a_channel = Mat(img.size(),CV_32FC1);
	Mat b_channel = Mat(img.size(),CV_32FC1);

	cvtColor(img,lab,COLOR_BGR2Lab);

	for(int i = 0; i < h; i++)
		for(int j = 0; j < w; j++)
		{
			l_channel.at<float>(i,j) = lab.at<float>(i,j*channel+0);
			a_channel.at<float>(i,j) = lab.at<float>(i,j*channel+1);
			b_channel.at<float>(i,j) = lab.at<float>(i,j*channel+2);
		}

	Mat L = Mat(img.size(),CV_32FC1);

	l_channel.convertTo(L,CV_32FC1,1.0/255.0);

	Domain_Filter obj;

	obj.filter(L, res, sigma_s, sigma_r, 1);

	Mat detail = Mat(h,w,CV_32FC1);

	for(int i = 0; i < h; i++)
		for(int j = 0; j < w; j++)
			detail.at<float>(i,j) = L.at<float>(i,j) - res.at<float>(i,j);

	for(int i = 0; i < h; i++)
		for(int j = 0; j < w; j++)
			L.at<float>(i,j) = res.at<float>(i,j) + factor*detail.at<float>(i,j);

	L.convertTo(l_channel,CV_32FC1,255);

	for(int i = 0; i < h; i++)
		for(int j = 0; j < w; j++)
		{
			lab.at<float>(i,j*channel+0) = l_channel.at<float>(i,j);
			lab.at<float>(i,j*channel+1) = a_channel.at<float>(i,j);
			lab.at<float>(i,j*channel+2) = b_channel.at<float>(i,j);
		}

	cvtColor(lab,result,COLOR_Lab2BGR);
	result.convertTo(dst,CV_8UC3,255);
}

void cv::pencilSketch(InputArray _src, OutputArray _dst, OutputArray _dst1, float sigma_s, float sigma_r, float shade_factor)
{
	Mat I = _src.getMat();
	_dst.create(I.size(), CV_8UC1);
	Mat dst = _dst.getMat();

	_dst1.create(I.size(), CV_8UC3);
	Mat dst1 = _dst1.getMat();
	
	Mat img = Mat(I.size(),CV_32FC3);
	I.convertTo(img,CV_32FC3,1.0/255.0);

	Domain_Filter obj;

	Mat sketch = Mat(I.size(),CV_32FC1);
	Mat color_sketch = Mat(I.size(),CV_32FC3);

	obj.pencil_sketch(img, sketch, color_sketch, sigma_s, sigma_r, shade_factor);

	sketch.convertTo(dst,CV_8UC1,255);
	color_sketch.convertTo(dst1,CV_8UC3,255);

}

void cv::stylization(InputArray _src, OutputArray _dst, float sigma_s, float sigma_r)
{
	Mat I = _src.getMat();
	_dst.create(I.size(), CV_8UC3);
	Mat dst = _dst.getMat();

	Mat img = Mat(I.size(),CV_32FC3);
	I.convertTo(img,CV_32FC3,1.0/255.0);

	int h = img.size().height;
	int w = img.size().width;
	int channel = img.channels();

	Mat res = Mat(h,w,CV_32FC3);

	Domain_Filter obj;
	obj.filter(img, res, sigma_s, sigma_r, NORMCONV_FILTER);

	vector <Mat> planes;
	split(res, planes);

	Mat magXR = Mat(h, w, CV_32FC1);
	Mat magYR = Mat(h, w, CV_32FC1);

	Mat magXG = Mat(h, w, CV_32FC1);
	Mat magYG = Mat(h, w, CV_32FC1);

	Mat magXB = Mat(h, w, CV_32FC1);
	Mat magYB = Mat(h, w, CV_32FC1);

	Sobel(planes[0], magXR, CV_32FC1, 1, 0, 3);
	Sobel(planes[0], magYR, CV_32FC1, 0, 1, 3);

	Sobel(planes[1], magXG, CV_32FC1, 1, 0, 3);
	Sobel(planes[1], magYG, CV_32FC1, 0, 1, 3);

	Sobel(planes[2], magXB, CV_32FC1, 1, 0, 3);
	Sobel(planes[2], magYB, CV_32FC1, 0, 1, 3);

	Mat magx = Mat(h,w,CV_32FC1);
	Mat magy = Mat(h,w,CV_32FC1);

	Mat mag1 = Mat(h,w,CV_32FC1);
	Mat mag2 = Mat(h,w,CV_32FC1);
	Mat mag3 = Mat(h,w,CV_32FC1);

	magnitude(magXR,magYR,mag1);
	magnitude(magXG,magYG,mag2);
	magnitude(magXB,magYB,mag3);

	Mat magnitude = Mat(h,w,CV_32FC1);

	for(int i =0;i < h;i++)
		for(int j=0;j<w;j++)
		{
			magnitude.at<float>(i,j) = mag1.at<float>(i,j) + mag2.at<float>(i,j) + mag3.at<float>(i,j);
		}

	for(int i =0;i < h;i++)
		for(int j=0;j<w;j++)
		{
			magnitude.at<float>(i,j) = 1.0 -  magnitude.at<float>(i,j);
		}

	Mat stylized = Mat(h,w,CV_32FC3);

	for(int i =0;i < h;i++)
		for(int j=0;j<w;j++)
			for(int c=0;c<channel;c++)
			{
				stylized.at<float>(i,j*channel + c) = res.at<float>(i,j*channel + c) * magnitude.at<float>(i,j);
			}

	stylized.convertTo(dst,CV_8UC3,255);
}

void cv::edgeEnhance(InputArray _src, OutputArray _dst, float sigma_s, float sigma_r)
{
	Mat I = _src.getMat();
	_dst.create(I.size(), CV_8UC1);
	Mat dst = _dst.getMat();

	Mat img = Mat(I.size(),CV_32FC3);
	I.convertTo(img,CV_32FC3,1.0/255.0);

	Mat orig = img.clone();

	int h = img.size().height;
	int w = img.size().width;

	Mat res = Mat(h,w,CV_32FC3);
	Mat magnitude = Mat(h,w,CV_32FC1);

	Mat mag8 = Mat(h,w,CV_32FC1);

	Domain_Filter obj;

	obj.filter(img, res, sigma_s, sigma_r, NORMCONV_FILTER);

	obj.find_magnitude(res,magnitude);

	magnitude.convertTo(dst,CV_8UC1,255);
}
