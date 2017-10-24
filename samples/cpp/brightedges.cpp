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
// Copyright (C) 2017, IBM Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//      Marc Fiammante marc.fiammante@fr.ibm.com
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
//   * The name of OpenCV Foundation or contributors may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the OpenCV Foundation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
#include "brightedges.h"
#include "opencv2/opencv.hpp"
#include <iostream>
 
namespace cv
{
	bool isPixelMinimum(Mat &edge,int row, int col,int contrast) {
		unsigned char *input = (unsigned char*)(edge.data);
		int pixel = input[edge.cols * row + col]+contrast;
		int m2 = input[edge.cols * (row - 2) + (col - 2)];
		int m1 = input[edge.cols * (row-1) + (col-1)];	
		int p1 = input[edge.cols * (row + 1) + (col + 1)];
		int p2 = input[edge.cols * (row + 2) + (col + 2)];
		if ((pixel< (m1+m2)/2) && (pixel < (p1+p2)/2)) return true; // Local minimum diagonal
		m2 = input[edge.cols * (row - 2) + (col)];
		m1 = input[edge.cols * (row - 1) + (col)];
		p1 = input[edge.cols * (row + 1) + (col)];
		p2 = input[edge.cols * (row + 2) + (col)];
		if ((pixel< (m1 + m2) / 2) && (pixel < (p1 + p2) / 2)) return true; // Local minimum diagonal
		m2 = input[edge.cols * (row - 2) + (col+2)];
		m1 = input[edge.cols * (row - 1) + (col+1)];
		p1 = input[edge.cols * (row + 1) + (col-1)];
		p2 = input[edge.cols * (row + 2) + (col-2)];
		if ((pixel< (m1 + m2) / 2) && (pixel < (p1 + p2) / 2)) return true; // Local minimum diagonal
		m2 = input[edge.cols * (row ) + (col + 2)];
		m1 = input[edge.cols * (row ) + (col + 1)];
		p1 = input[edge.cols * (row ) + (col - 1)];
		p2 = input[edge.cols * (row ) + (col - 2)];
		if ((pixel< (m1 + m2) / 2) && (pixel < (p1 + p2) / 2)) return true; // Local minimum diagonal
		return false;
	}
	int contrastEdges(Mat &cedge,Mat &edge,int contrast) {
		unsigned char *input = (unsigned char*)(edge.data);
	
		for (int row = 2; row < cedge.rows - 2; row++) {
			for (int col = 2; col < cedge.cols - 2; col++) {			
				if (isPixelMinimum(cedge, row, col,contrast)) {
					input[edge.cols * row + col] = 0;
				}
				else {
					input[edge.cols * row + col] = 255;
				}
				
				
				// std::cout << "i=" << i << ", j=" << j << " a=" << a <<"\n";
			}
		}
		return 0;
	}
	CV_EXPORTS_W  void BrightEdges(Mat &image, Mat &edge, int contrast, int shortrange, int longrange)
	{
		Mat gray, gblur, bblur, diff,cedge;
		GaussianBlur(image, gblur, Size(shortrange, shortrange), 0);
		blur(image, bblur, Size(longrange, longrange));
		absdiff(gblur, bblur, diff);
		cvtColor(diff, gray, CV_BGR2GRAY);
		equalizeHist(gray, cedge);
		edge = cedge.clone();
		if (contrast>0) contrastEdges(cedge,edge,contrast);
	}
}
