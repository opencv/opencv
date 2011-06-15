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
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

/*
 This is a modification of the variational stereo correspondence algorithm, described in:
 S. Kosov, T. Thormaehlen, H.-P. Seidel "Accurate Real-Time Disparity Estimation with Variational Methods"
 Proceedings of the 5th International Symposium on Visual Computing, Vegas, USA

 This code is written by Sergey G. Kosov for "Visir PX" application as part of Project X (www.project-10.de)
 */ 

#include "precomp.hpp"
#include <limits.h>

namespace cv 
{
StereoVar::StereoVar() : levels(3), pyrScale(0.5), nIt(3), minDisp(0), maxDisp(16), poly_n(5), poly_sigma(1.2), fi(1000.0f), lambda(0.0f), penalization(PENALIZATION_TICHONOV), cycle(CYCLE_V), flags(USE_SMART_ID) 
{
}

StereoVar::StereoVar(int _levels, double _pyrScale, int _nIt, int _minDisp, int _maxDisp, int _poly_n, double _poly_sigma, float _fi, float _lambda, int _penalization, int _cycle, int _flags) : levels(_levels), pyrScale(_pyrScale), nIt(_nIt), minDisp(_minDisp), maxDisp(_maxDisp), poly_n(_poly_n), poly_sigma(_poly_sigma), fi(_fi), lambda(_lambda), penalization(_penalization), cycle(_cycle), flags(_flags)
{ // No Parameters check, since they are all public
}

StereoVar::~StereoVar()
{
}

static Mat diffX(Mat &img)
{
	// TODO try pointers or assm
	register int x, y;
	Mat dst(img.size(), img.type());
	dst.setTo(0);
	for (x = 0; x < img.cols - 1; x++)
		for (y = 0; y < img.rows; y++)
			dst.at<float>(y, x) = img.at<float>(y, x + 1) - img.at<float>(y ,x);
	return dst;
}

static Mat Gradient(Mat &img)
{
	Mat sobel, sobelX, sobelY;
	img.copyTo(sobelX);
	img.copyTo(sobelY);
	Sobel(img, sobelX, sobelX.type(), 1, 0, 1);
	Sobel(img, sobelY, sobelY.type(), 0, 1, 1);
	sobelX = abs(sobelX);
	sobelY = abs(sobelY);
	add(sobelX, sobelY, sobel);
	sobelX.release();
	sobelY.release();
	return sobel;
}

static float g_c(Mat z, int x, int y, float l)
{
	return 0.5f*l / sqrtf(l*l + z.at<float>(y,x)*z.at<float>(y,x));
}

static float g_p(Mat z, int x, int y, float l)
{
	return 0.5f*l*l / (l*l + z.at<float>(y,x)*z.at<float>(y,x)) ;
}

void StereoVar::VariationalSolver(Mat &I1, Mat &I2, Mat &I2x, Mat &u, int level)
{
	register int n, x, y;
	float gl = 1, gr = 1, gu = 1, gd = 1, gc = 4;
	Mat U; 
	Mat Sobel;
	u.copyTo(U);

	int		N = nIt;
	float	l = lambda;
	float	Fi = fi;

	double scale = pow(pyrScale, (double) level);
	if (flags & USE_SMART_ID) {										
		N = (int) (N / scale);
		Fi /= (float) scale;
		l *= (float) scale;
	}
	for (n = 0; n < N; n++) {
		if (penalization != PENALIZATION_TICHONOV) {if(!Sobel.empty()) Sobel.release(); Sobel = Gradient(U);}
		for (x = 1; x < u.cols - 1; x++) {
			for (y = 1 ; y < u.rows - 1; y++) {
				switch (penalization) {
					case PENALIZATION_CHARBONNIER:
						gc = g_c(Sobel, x, y, l);
						gl = gc + g_c(Sobel, x - 1, y, l);
						gr = gc + g_c(Sobel, x + 1, y, l);
						gu = gc + g_c(Sobel, x, y + 1, l);
						gd = gc + g_c(Sobel, x, y - 1, l);
						gc = gl + gr + gu + gd;
						break;
					case PENALIZATION_PERONA_MALIK:
						gc = g_p(Sobel, x, y, l);
						gl = gc + g_p(Sobel, x - 1, y, l);
						gr = gc + g_p(Sobel, x + 1, y, l);
						gu = gc + g_p(Sobel, x, y + 1, l);
						gd = gc + g_p(Sobel, x, y - 1, l);
						gc = gl + gr + gu + gd;
						break;
				}

				float fi = Fi;
				if (maxDisp > minDisp) {
					if (U.at<float>(y,x) > maxDisp * scale) {fi*=1000; U.at<float>(y,x) = static_cast<float>(maxDisp * scale);} 
					if (U.at<float>(y,x) < minDisp * scale) {fi*=1000; U.at<float>(y,x) = static_cast<float>(minDisp * scale);} 
				}

				int A = (int) (U.at<float>(y,x));
				int neg = 0; if (U.at<float>(y,x) <= 0) neg = -1;

				if (x + A >= u.cols)
					u.at<float>(y, x) = U.at<float>(y, u.cols - A - 1);
				else if (x + A + neg < 0)
					u.at<float>(y, x) = U.at<float>(y, - A + 2);
				else { 
					u.at<float>(y, x) = A + (I2x.at<float>(y, x + A + neg) * (I1.at<float>(y, x) - I2.at<float>(y, x + A))
										  + fi * (gr * U.at<float>(y, x + 1) + gl * U.at<float>(y, x - 1) + gu * U.at<float>(y + 1, x) + gd * U.at<float>(y - 1, x) - gc * A)) 
										  / (I2x.at<float>(y, x + A + neg) * I2x.at<float>(y, x + A + neg) + gc * fi) ; 
				}
			}//y
			u.at<float>(0, x) = u.at<float>(1, x);
			u.at<float>(u.rows - 1, x) = u.at<float>(u.rows - 2, x);
		}//x
		for (y = 0; y < u.rows; y++) {
			u.at<float>(y, 0) = u.at<float>(y, 1);
			u.at<float>(y, u.cols - 1) = u.at<float>(y, u.cols - 2);
		}
		u.copyTo(U);
	}//n
}

void StereoVar::VCycle_MyFAS(Mat &I1, Mat &I2, Mat &I2x, Mat &_u, int level)
{
	CvSize imgSize = _u.size();
	CvSize frmSize = cvSize((int) (imgSize.width * pyrScale + 0.5), (int) (imgSize.height * pyrScale + 0.5));
	Mat I1_h, I2_h, I2x_h, u_h, U, U_h;

	//PRE relaxation
	VariationalSolver(I1, I2, I2x, _u, level);

	if (level >= levels - 1) return;
	level ++;

	//scaling DOWN
	resize(I1, I1_h, frmSize, 0, 0, INTER_AREA);
	resize(I2, I2_h, frmSize, 0, 0, INTER_AREA);
	resize(_u, u_h, frmSize, 0, 0, INTER_AREA);
	u_h.convertTo(u_h, u_h.type(), pyrScale);
	I2x_h = diffX(I2_h);

	//Next level
	U_h = u_h.clone();
	VCycle_MyFAS(I1_h, I2_h, I2x_h, U_h, level);

	subtract(U_h, u_h, U_h);
	U_h.convertTo(U_h, U_h.type(), 1.0 / pyrScale);

	//scaling UP
	resize(U_h, U, imgSize);

	//correcting the solution
	add(_u, U, _u);

	//POST relaxation
	VariationalSolver(I1, I2, I2x, _u, level - 1);

	if (flags & USE_MEDIAN_FILTERING) medianBlur(_u, _u, 3);

	I1_h.release();
	I2_h.release();
	I2x_h.release();
	u_h.release();
	U.release();
	U_h.release();
}

void StereoVar::FMG(Mat &I1, Mat &I2, Mat &I2x, Mat &u, int level)
{
	double	scale = pow(pyrScale, (double) level);
	CvSize	frmSize = cvSize((int) (u.cols * scale + 0.5), (int) (u.rows * scale + 0.5));
	Mat I1_h, I2_h, I2x_h, u_h;

	//scaling DOWN
	resize(I1, I1_h, frmSize, 0, 0, INTER_AREA);
	resize(I2, I2_h, frmSize, 0, 0, INTER_AREA);
	resize(u, u_h, frmSize, 0, 0, INTER_AREA);
	u_h.convertTo(u_h, u_h.type(), scale);
	I2x_h = diffX(I2_h);

	switch (cycle) {
		case CYCLE_O:
			VariationalSolver(I1_h, I2_h, I2x_h, u_h, level);
			break;
		case CYCLE_V:
			VCycle_MyFAS(I1_h, I2_h, I2x_h, u_h, level);
			break;
	}

	u_h.convertTo(u_h, u_h.type(), 1.0 / scale);

	//scaling UP
	resize(u_h, u, u.size(), 0, 0, INTER_CUBIC);

	I1_h.release();
	I2_h.release();
	I2x_h.release();
	u_h.release();

	level--;
	if (flags & USE_MEDIAN_FILTERING) medianBlur(u, u, 3);
	if (level >= 0) FMG(I1, I2, I2x, u, level);
}

void StereoVar::operator ()( const Mat& left, const Mat& right, Mat& disp )
{
	CV_Assert(left.size() == right.size() && left.type() == right.type());
	CvSize imgSize = left.size();
	int MaxD = MAX(std::abs(minDisp), std::abs(maxDisp)); 
	int SignD = 1; if (MIN(minDisp, maxDisp) < 0) SignD = -1;
	if (minDisp >= maxDisp) {MaxD = 256; SignD = 1;}
		
	Mat u;
	if ((flags & USE_INITIAL_DISPARITY) && (!disp.empty())) {
		CV_Assert(disp.size() == left.size() && disp.type() == CV_8UC1);
		disp.convertTo(u, CV_32FC1, static_cast<double>(SignD * MaxD) / 256);
	} else {
		u.create(imgSize, CV_32FC1);
		u.setTo(0);
	}

	// Preprocessing
	Mat leftgray, rightgray;
	if (left.type() != CV_8UC1) {
		cvtColor(left, leftgray, CV_BGR2GRAY);
		cvtColor(right, rightgray, CV_BGR2GRAY);
	} else {
		left.copyTo(leftgray);
		right.copyTo(rightgray);
	}
	if (flags & USE_EQUALIZE_HIST) {
		equalizeHist(leftgray, leftgray);
		equalizeHist(rightgray, rightgray);
	}
	if (poly_sigma > 0.0001) {
		GaussianBlur(leftgray, leftgray, cvSize(poly_n, poly_n), poly_sigma);
		GaussianBlur(rightgray, rightgray, cvSize(poly_n, poly_n), poly_sigma);
	}
		
	Mat I1, I2;
	leftgray.convertTo(I1, CV_32FC1);
	rightgray.convertTo(I2, CV_32FC1);
	leftgray.release();
	rightgray.release();

	Mat I2x = diffX(I2);
		
	FMG(I1, I2, I2x, u, levels - 1);		
		
	I1.release();
	I2.release();
	I2x.release();
	
	
	disp.create( left.size(), CV_8UC1 );
	u = abs(u);
	u.convertTo(disp, disp.type(), 256 / MaxD, 0);	

	u.release();
}
} // namespace