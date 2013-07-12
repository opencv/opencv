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

#include "opencv2/photo.hpp"
#include "opencv2/imgproc.hpp"

namespace cv
{

static void downsample(Mat& src, Mat& dst)
{
	dst = Mat(src.rows / 2, src.cols / 2, CV_8UC1);

	int offset = src.cols * 2;
	uchar *src_ptr = src.ptr();
	uchar *dst_ptr = dst.ptr();
	for(int y = 0; y < dst.rows; y ++) {
		uchar *ptr = src_ptr;
		for(int x = 0; x < dst.cols; x++) {
			dst_ptr[0] = ptr[0];
			dst_ptr++;
			ptr += 2;
		}
		src_ptr += offset;
	}
}

static void buildPyr(Mat& img, std::vector<Mat>& pyr, int maxlevel) 
{
	pyr.resize(maxlevel + 1);
	pyr[0] = img.clone();
	for(int level = 0; level < maxlevel; level++) {
		downsample(pyr[level], pyr[level + 1]);
	}
}

static int getMedian(Mat& img)
{
	int channels = 0;
	Mat hist; 
	int hist_size = 256;
	float range[] = {0, 256} ;
	const float* ranges[] = {range};
	calcHist(&img, 1, &channels, Mat(), hist, 1, &hist_size, ranges);
	float *ptr = hist.ptr<float>();
	int median = 0, sum = 0;
	int thresh = img.total() / 2;
	while(sum < thresh && median < 256) {
		sum += (int)ptr[median];
		median++;
	}
	return median;
}

static void computeBitmaps(Mat& img, Mat& tb, Mat& eb, int exclude_range)
{
	int median = getMedian(img);
	compare(img, median, tb, CMP_GT);
	compare(abs(img - median), exclude_range, eb, CMP_GT);
}

void shiftMat(InputArray _src, Point shift, OutputArray _dst) 
{
	Mat src = _src.getMat();
	if(src.depth() != CV_8U) {
		CV_Error(Error::StsBadArg, "Image must have CV_8U depth.");
	}
	_dst.create(src.size(), src.type());
	Mat dst = _dst.getMat();

	dst = Mat::zeros(src.size(), src.type());
	int width = src.cols - abs(shift.x);
	int height = src.rows - abs(shift.y);
	Rect dst_rect(max(shift.x, 0), max(shift.y, 0), width, height);
	Rect src_rect(max(-shift.x, 0), max(-shift.y, 0), width, height);
	src(src_rect).copyTo(dst(dst_rect));
}

Point getExpShift(InputArray _img0, InputArray _img1, int max_bits, int exclude_range)
{
	Mat img0 = _img0.getMat();
	Mat img1 = _img1.getMat();
	if(img0.type() != CV_8UC1 || img1.type() != CV_8UC1) {
		CV_Error(Error::StsBadArg, "Images must have CV_8UC1 type.");
	}
	if(img0.size() != img0.size()) {
		CV_Error(Error::StsBadArg, "Image dimensions must be equal.");
	}
	int maxlevel = (int)(log((double)max(img0.rows, img0.cols)) / log(2.0)) - 1;
	maxlevel = min(maxlevel, max_bits - 1);

	std::vector<Mat> pyr0;
	std::vector<Mat> pyr1;
	buildPyr(img0, pyr0, maxlevel);
	buildPyr(img1, pyr1, maxlevel);	
	
	Point shift(0, 0);
	for(int level = maxlevel; level >= 0; level--) {
		
		shift *= 2;
		Mat tb1, tb2, eb1, eb2;
		computeBitmaps(pyr0[level], tb1, eb1, exclude_range);
		computeBitmaps(pyr1[level], tb2, eb2, exclude_range);

		int min_err = pyr0[level].total();
		Point new_shift(shift);
		for(int i = -1; i <= 1; i++) {
			for(int j = -1; j <= 1; j++) {
				Point test_shift = shift + Point(i, j);
				Mat shifted_tb2, shifted_eb2, diff;
				shiftMat(tb2, test_shift, shifted_tb2);
				shiftMat(eb2, test_shift, shifted_eb2);
				bitwise_xor(tb1, shifted_tb2, diff);
				bitwise_and(diff, eb1, diff);
				bitwise_and(diff, shifted_eb2, diff);
				int err = countNonZero(diff);
				if(err < min_err) {
					new_shift = test_shift;
					min_err = err;
				}		
			}
		}
		shift = new_shift;
	}
	return shift;
}
};