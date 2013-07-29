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

#include <iostream>

namespace cv
{

static void triangleWeights(float weights[])
{
	for(int i = 0; i < 128; i++) {
		weights[i] = i + 1.0f;
	}
	for(int i = 128; i < 256; i++) {
		weights[i] = 256.0f - i;
	}
}

static Mat linearResponse()
{
	Mat response(256, 1, CV_32F);
    for(int i = 1; i < 256; i++) {
        response.at<float>(i) = logf((float)i);
    }
    response.at<float>(0) = response.at<float>(1);
	return response;
}

static void modifyCheckResponse(Mat &response)
{
	if(response.empty()) {
		response = linearResponse();
	}
	CV_Assert(response.rows == 256 && (response.cols == 1 || response.cols == 3));
	response.convertTo(response, CV_32F);
	if(response.cols == 1) {
		Mat result(256, 3, CV_32F);
		for(int i = 0; i < 3; i++) {
			response.copyTo(result.col(i));
		}
		response = result;
	}
}

static void checkImages(const std::vector<Mat>& images, bool hdr, const std::vector<float>& _exp_times = std::vector<float>())
{
	CV_Assert(!images.empty());
	CV_Assert(!hdr || images.size() == _exp_times.size());
	int width = images[0].cols;
	int height = images[0].rows;
	int channels = images[0].channels();
	for(size_t i = 0; i < images.size(); i++) {

		CV_Assert(images[i].cols == width && images[i].rows == height);
		CV_Assert(images[i].channels() == channels && images[i].depth() == CV_8U);
	}
}

void alignImages(InputArrayOfArrays _src, std::vector<Mat>& dst)
{
	std::vector<Mat> src;
    _src.getMatVector(src);
	checkImages(src, false);
	dst.resize(src.size());

	size_t pivot = src.size() / 2;
	dst[pivot] = src[pivot];
	Mat gray_base;
	cvtColor(src[pivot], gray_base, COLOR_RGB2GRAY);

	for(size_t i = 0; i < src.size(); i++) {
		if(i == pivot) {
			continue;
		}
		Mat gray;
		cvtColor(src[i], gray, COLOR_RGB2GRAY);
		Point shift = getExpShift(gray_base, gray);
		shiftMat(src[i], shift, dst[i]);
	}
}

void makeHDR(InputArrayOfArrays _images, const std::vector<float>& _exp_times, OutputArray _dst, Mat response)
{
	std::vector<Mat> images;
    _images.getMatVector(images);
	checkImages(images, true, _exp_times);
	modifyCheckResponse(response);
	_dst.create(images[0].size(), CV_MAKETYPE(CV_32F, images[0].channels()));
	Mat result = _dst.getMat();

	std::vector<float> exp_times(_exp_times.size());
	for(size_t i = 0; i < exp_times.size(); i++) {
        exp_times[i] = logf(_exp_times[i]);
	}

	float weights[256];
	triangleWeights(weights);
	
	int channels = images[0].channels();
	float *res_ptr = result.ptr<float>();
	for(size_t pos = 0; pos < result.total(); pos++, res_ptr += channels) {

		std::vector<float> sum(channels, 0);
		float weight_sum = 0;
		for(size_t im = 0; im < images.size(); im++) {

			uchar *img_ptr = images[im].ptr() + channels * pos;
			float w = 0;
			for(int channel = 0; channel < channels; channel++) {
				w += weights[img_ptr[channel]];
			}
			w /= channels;
			weight_sum += w;
			for(int channel = 0; channel < channels; channel++) {
				sum[channel] += w * (response.at<float>(img_ptr[channel], channel) - exp_times[im]);
			}
		}
		for(int channel = 0; channel < channels; channel++) {
			res_ptr[channel] = exp(sum[channel] / weight_sum);
		}
	}
}

void exposureFusion(InputArrayOfArrays _images, OutputArray _dst, float wc, float ws, float we)
{
	std::vector<Mat> images;
    _images.getMatVector(images);
	checkImages(images, false);

	std::vector<Mat> weights(images.size());
	Mat weight_sum = Mat::zeros(images[0].size(), CV_32FC1);
	for(size_t im = 0; im < images.size(); im++) {
		Mat img, gray, contrast, saturation, wellexp;
		std::vector<Mat> channels(3);

		images[im].convertTo(img, CV_32FC3, 1.0/255.0);
		cvtColor(img, gray, COLOR_RGB2GRAY);
		split(img, channels);

		Laplacian(gray, contrast, CV_32F);
		contrast = abs(contrast);

		Mat mean = (channels[0] + channels[1] + channels[2]) / 3.0f;
		saturation = Mat::zeros(channels[0].size(), CV_32FC1);
		for(int i = 0; i < 3;  i++) {
			Mat deviation = channels[i] - mean;
			pow(deviation, 2.0, deviation);
			saturation += deviation;
		}
        sqrt(saturation, saturation);

		wellexp = Mat::ones(gray.size(), CV_32FC1);
		for(int i = 0; i < 3; i++) {
			Mat exp = channels[i] - 0.5f;
			pow(exp, 2, exp);
			exp = -exp / 0.08;
			wellexp = wellexp.mul(exp);
		}

		pow(contrast, wc, contrast);
		pow(saturation, ws, saturation);
		pow(wellexp, we, wellexp);

		weights[im] = contrast;
		weights[im] = weights[im].mul(saturation);
		weights[im] = weights[im].mul(wellexp);
		weight_sum += weights[im];
	}
    int maxlevel = static_cast<int>(logf(static_cast<float>(max(images[0].rows, images[0].cols))) / logf(2.0)) - 1;
	std::vector<Mat> res_pyr(maxlevel + 1);

	for(size_t im = 0; im < images.size(); im++) {
		weights[im] /= weight_sum;
		Mat img;
		images[im].convertTo(img, CV_32FC3, 1/255.0);
		std::vector<Mat> img_pyr, weight_pyr;
		buildPyramid(img, img_pyr, maxlevel);
		buildPyramid(weights[im], weight_pyr, maxlevel);
		for(int lvl = 0; lvl < maxlevel; lvl++) {
			Mat up;
			pyrUp(img_pyr[lvl + 1], up, img_pyr[lvl].size());
			img_pyr[lvl] -= up;
		}
		for(int lvl = 0; lvl <= maxlevel; lvl++) {
			std::vector<Mat> channels(3);
			split(img_pyr[lvl], channels);
			for(int i = 0; i < 3; i++) {
				channels[i] = channels[i].mul(weight_pyr[lvl]);
			}
			merge(channels, img_pyr[lvl]);
			if(res_pyr[lvl].empty()) {
				res_pyr[lvl] = img_pyr[lvl];
			} else {
				res_pyr[lvl] += img_pyr[lvl];
			}
		}
	}
	for(int lvl = maxlevel; lvl > 0; lvl--) {
		Mat up;
		pyrUp(res_pyr[lvl], up, res_pyr[lvl - 1].size());
	    res_pyr[lvl - 1] += up;
	}
	_dst.create(images[0].size(), CV_32FC3);
	Mat result = _dst.getMat();
	res_pyr[0].copyTo(result);
}

void estimateResponse(InputArrayOfArrays _images, const std::vector<float>& exp_times, OutputArray _dst, int samples, float lambda)
{
	std::vector<Mat> images;
    _images.getMatVector(images);
	checkImages(images, true, exp_times);
	_dst.create(256, images[0].channels(), CV_32F);
	Mat response = _dst.getMat();

	float w[256];
	triangleWeights(w);

	for(int channel = 0; channel < images[0].channels(); channel++) {
		Mat A = Mat::zeros(samples * images.size() + 257, 256 + samples, CV_32F);
		Mat B = Mat::zeros(A.rows, 1, CV_32F);

		int eq = 0;
		for(int i = 0; i < samples; i++) {

			int pos = 3 * (rand() % images[0].total()) + channel;
			for(size_t j = 0; j < images.size(); j++) {

				int val = (images[j].ptr() + pos)[0];
				A.at<float>(eq, val) = w[val];
				A.at<float>(eq, 256 + i) = -w[val];
				B.at<float>(eq, 0) = w[val] * log(exp_times[j]);		
				eq++;
			}
		}
		A.at<float>(eq, 128) = 1;
		eq++;

		for(int i = 0; i < 254; i++) {
			A.at<float>(eq, i) = lambda * w[i + 1];
			A.at<float>(eq, i + 1) = -2 * lambda * w[i + 1];
			A.at<float>(eq, i + 2) = lambda * w[i + 1];
			eq++;
		}
		Mat solution;
		solve(A, B, solution, DECOMP_SVD);
		solution.rowRange(0, 256).copyTo(response.col(channel));
	}
}

};

