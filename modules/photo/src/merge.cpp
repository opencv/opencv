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
#include "opencv2/imgproc.hpp"
#include "hdr_common.hpp"
#include <iostream>

namespace cv
{

class MergeDebevecImpl : public MergeDebevec
{
public:
	MergeDebevecImpl() :
		name("MergeDebevec"),
		weights(tringleWeights())
	{
	}
	
	void process(InputArrayOfArrays src, OutputArray dst, const std::vector<float>& times, InputArray input_response)
	{
		std::vector<Mat> images;
		src.getMatVector(images);
		dst.create(images[0].size(), CV_MAKETYPE(CV_32F, images[0].channels()));
		Mat result = dst.getMat();

		CV_Assert(images.size() == times.size());
		CV_Assert(images[0].depth() == CV_8U);
		checkImageDimensions(images);

		Mat response = input_response.getMat();
		CV_Assert(response.rows == 256 && response.cols >= images[0].channels());
		Mat log_response;
		log(response, log_response);
		
		std::vector<float> exp_times(times.size());
		for(size_t i = 0; i < exp_times.size(); i++) {
			exp_times[i] = logf(times[i]);
		}
	
		int channels = images[0].channels();
		float *res_ptr = result.ptr<float>();
		for(size_t pos = 0; pos < result.total(); pos++, res_ptr += channels) {

			std::vector<float> sum(channels, 0);
			float weight_sum = 0;
			for(size_t im = 0; im < images.size(); im++) {

				uchar *img_ptr = images[im].ptr() + channels * pos;
				float w = 0;
				for(int channel = 0; channel < channels; channel++) {
					w += weights.at<float>(img_ptr[channel]);
				}
				w /= channels; 
				weight_sum += w;
				for(int channel = 0; channel < channels; channel++) {
					sum[channel] += w * (log_response.at<float>(img_ptr[channel], channel) - exp_times[im]);
				}
			}
			for(int channel = 0; channel < channels; channel++) {
				res_ptr[channel] = exp(sum[channel] / weight_sum);
			}
		}
	}

	void process(InputArrayOfArrays src, OutputArray dst, const std::vector<float>& times)
	{
		Mat response(256, 3, CV_32F);
		for(int i = 0; i < 256; i++) {
			for(int j = 0; j < 3; j++) {
				response.at<float>(i, j) = max(i, 1);
			}
		}
		process(src, dst, times, response);
	}

protected:
	String name;
	Mat weights;
};

Ptr<MergeDebevec> createMergeDebevec()
{
	return new MergeDebevecImpl;
}

class MergeMertensImpl : public MergeMertens
{
public:
	MergeMertensImpl(float wcon, float wsat, float wexp) :
		wcon(wcon),
		wsat(wsat),
		wexp(wexp),
		name("MergeMertens")
	{
	}
	
	void process(InputArrayOfArrays src, OutputArrayOfArrays dst, const std::vector<float>& times, InputArray response)
	{
		process(src, dst);
	}

	void process(InputArrayOfArrays src, OutputArray dst)
	{
		std::vector<Mat> images;
		src.getMatVector(images);
		checkImageDimensions(images);

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

			pow(contrast, wcon, contrast);
			pow(saturation, wsat, saturation);
			pow(wellexp, wexp, wellexp);

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
		dst.create(images[0].size(), CV_32FC3);
		res_pyr[0].copyTo(dst.getMat());
	}

	float getContrastWeight() const { return wcon; }
	void setContrastWeight(float val) { wcon = val; }

	float getSaturationWeight() const { return wsat; }
	void setSaturationWeight(float val) { wsat = val; }

	float getExposureWeight() const { return wexp; }
	void setExposureWeight(float val) { wexp = val; }

	void write(FileStorage& fs) const
    {
        fs << "name" << name
		   << "contrast_weight" << wcon
		   << "saturation_weight" << wsat
		   << "exposure_weight" << wexp;
    }

    void read(const FileNode& fn)
    {
        FileNode n = fn["name"];
        CV_Assert(n.isString() && String(n) == name);
        wcon = fn["contrast_weight"];
		wsat = fn["saturation_weight"];
		wexp = fn["exposure_weight"];
    }

protected:
	String name;
	float wcon, wsat, wexp;
};

Ptr<MergeMertens> createMergeMertens(float wcon, float wsat, float wexp)
{
	return new MergeMertensImpl(wcon, wsat, wexp);
}

}
