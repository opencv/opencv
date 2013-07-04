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

static void triangleWeights(float weights[])
{
	for(int i = 0; i < 128; i++) {
		weights[i] = i + 1.0f;
	}
	for(int i = 128; i < 256; i++) {
		weights[i] = 256.0f - i;
	}
}

static void generateResponce(float responce[])
{
    for(int i = 0; i < 256; i++) {
        responce[i] = log((float)i);
    }
    responce[0] = responce[1];
}

void makeHDR(InputArrayOfArrays _images, const std::vector<float>& _exp_times, OutputArray _dst)
{
	std::vector<Mat> images;
    _images.getMatVector(images);
	if(images.empty()) {
		CV_Error(Error::StsBadArg, "Need at least one image");
	}
	if(images.size() != _exp_times.size()) {
		CV_Error(Error::StsBadArg, "Number of images and number of exposure times must be equal.");
	}
	int width = images[0].cols;
	int height = images[0].rows;
	for(size_t i = 0; i < images.size(); i++) {

		if(images[i].cols != width || images[i].rows != height) {
			CV_Error(Error::StsBadArg, "Image dimensions must be equal.");
		}
		if(images[i].type() != CV_8UC3) {
			CV_Error(Error::StsBadArg, "Images must have CV_8UC3 type.");
		}
	}
	_dst.create(images[0].size(), CV_32FC3);
	Mat result = _dst.getMat();
	std::vector<float> exp_times(_exp_times.size());
	for(size_t i = 0; i < exp_times.size(); i++) {
		exp_times[i] = log(_exp_times[i]);
	}

	float weights[256], responce[256];
	triangleWeights(weights);
	generateResponce(responce);

	float max = 0;
	float *res_ptr = result.ptr<float>();
	for(size_t pos = 0; pos < result.total(); pos++, res_ptr += 3) {

		float sum[3] = {0, 0, 0};
		float weight_sum = 0;
		for(size_t im = 0; im < images.size(); im++) {

			uchar *img_ptr = images[im].ptr() + 3 * pos;
			float w = (weights[img_ptr[0]] + weights[img_ptr[1]] +
				       weights[img_ptr[2]]) / 3;
			weight_sum += w;
			for(int channel = 0; channel < 3; channel++) {
				sum[channel] += w * (responce[img_ptr[channel]] - exp_times[im]);
			}
		}
		for(int channel = 0; channel < 3; channel++) {
			res_ptr[channel] = exp(sum[channel] / weight_sum);
			if(res_ptr[channel] > max) {
				max = res_ptr[channel];
			}
		}
	}
	result = result / max;
}

};