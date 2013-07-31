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

#include "precomp.hpp"
#include "opencv2/photo.hpp"
#include "opencv2/imgproc.hpp"

namespace cv
{

class TonemapLinearImpl : public TonemapLinear
{
public:
	TonemapLinearImpl(float gamma) : gamma(gamma), name("TonemapLinear")
	{
	}

	void process(InputArray _src, OutputArray _dst) 
	{
		Mat src = _src.getMat();
		CV_Assert(!src.empty());
		_dst.create(src.size(), CV_32FC3);
		Mat dst = _dst.getMat();
		
		double min, max;
		minMaxLoc(src, &min, &max);
		if(max - min > DBL_EPSILON) {
			dst = (src - min) / (max - min);
		} else {
			src.copyTo(dst);
		}

		pow(dst, 1.0f / gamma, dst);
	}

	float getGamma() const { return gamma; }
	void setGamma(float val) { gamma = val; }

	void write(FileStorage& fs) const
    {
        fs << "name" << name
           << "gamma" << gamma;
    }

    void read(const FileNode& fn)
    {
        FileNode n = fn["name"];
        CV_Assert(n.isString() && String(n) == name);
        gamma = fn["gamma"];
    }

protected:
	String name;
	float gamma;
};

Ptr<TonemapLinear> createTonemapLinear(float gamma)
{
	return new TonemapLinearImpl(gamma);
}

class TonemapDragoImpl : public TonemapDrago
{
public:
	TonemapDragoImpl(float gamma, float bias) : 
	    gamma(gamma), 
        bias(bias),
		name("TonemapLinear")
	{
	}

	void process(InputArray _src, OutputArray _dst) 
	{
		Mat src = _src.getMat();
		CV_Assert(!src.empty());
		_dst.create(src.size(), CV_32FC3);
		Mat img = _dst.getMat();
		
		Ptr<TonemapLinear> linear = createTonemapLinear(1.0f);
		linear->process(src, img);

		Mat gray_img;
		cvtColor(img, gray_img, COLOR_RGB2GRAY);
		Mat log_img;
		log(gray_img, log_img);
		float mean = expf(static_cast<float>(sum(log_img)[0]) / log_img.total());
		gray_img /= mean;
		log_img.release();

		double max;
		minMaxLoc(gray_img, NULL, &max);

		Mat map;
		log(gray_img + 1.0f, map);
		Mat div;
		pow(gray_img / (float)max, logf(bias) / logf(0.5f), div);
		log(2.0f + 8.0f * div, div);
		map = map.mul(1.0f / div);
		map = map.mul(1.0f / gray_img);
		div.release();
		gray_img.release();

		std::vector<Mat> channels(3);
		split(img, channels);
		for(int i = 0; i < 3; i++) {
			channels[i] = channels[i].mul(map);
		}
		map.release();
		merge(channels, img);
		
		linear->setGamma(gamma);
		linear->process(img, img);
	}

	float getGamma() const { return gamma; }
	void setGamma(float val) { gamma = val; }

	float getBias() const { return bias; }
	void setBias(float val) { bias = val; }

	void write(FileStorage& fs) const
    {
        fs << "name" << name
           << "gamma" << gamma
		   << "bias" << bias;
    }

    void read(const FileNode& fn)
    {
        FileNode n = fn["name"];
        CV_Assert(n.isString() && String(n) == name);
        gamma = fn["gamma"];
		bias = fn["bias"];
    }

protected:
	String name;
	float gamma, bias;
};

Ptr<TonemapDrago> createTonemapDrago(float gamma, float bias)
{
	return new TonemapDragoImpl(gamma, bias);
}
 
class TonemapDurandImpl : public TonemapDurand
{
public:
	TonemapDurandImpl(float gamma, float contrast, float sigma_color, float sigma_space) : 
	    gamma(gamma), 
        contrast(contrast),
		sigma_color(sigma_color),
		sigma_space(sigma_space),
		name("TonemapDurand")
	{
	}

	void process(InputArray _src, OutputArray _dst) 
	{
		Mat src = _src.getMat();
		CV_Assert(!src.empty());
		_dst.create(src.size(), CV_32FC3);
		Mat dst = _dst.getMat();
		
		Mat gray_img;
		cvtColor(src, gray_img, COLOR_RGB2GRAY);
		Mat log_img;
		log(gray_img, log_img);
		Mat map_img;
		bilateralFilter(log_img, map_img, -1, sigma_color, sigma_space);
        
		double min, max;
		minMaxLoc(map_img, &min, &max);
		float scale = contrast / (float)(max - min);

		exp(map_img * (scale - 1.0f) + log_img, map_img);
		log_img.release();
		map_img = map_img.mul(1.0f / gray_img);
		gray_img.release();

		std::vector<Mat> channels(3);
		split(src, channels);
		for(int i = 0; i < 3; i++) {
			channels[i] = channels[i].mul(map_img);
		}
		merge(channels, dst);
		pow(dst, 1.0f / gamma, dst);
	}

	float getGamma() const { return gamma; }
	void setGamma(float val) { gamma = val; }

	float getContrast() const { return contrast; }
	void setContrast(float val) { contrast = val; }

	float getSigmaColor() const { return sigma_color; }
	void setSigmaColor(float val) { sigma_color = val; }

	float getSigmaSpace() const { return sigma_space; }
	void setSigmaSpace(float val) { sigma_space = val; }

	void write(FileStorage& fs) const
    {
        fs << "name" << name
           << "gamma" << gamma
		   << "contrast" << contrast 
		   << "sigma_color" << sigma_color 
		   << "sigma_space" << sigma_space;
    }

    void read(const FileNode& fn)
    {
        FileNode n = fn["name"];
        CV_Assert(n.isString() && String(n) == name);
        gamma = fn["gamma"];
		contrast = fn["contrast"];
		sigma_color = fn["sigma_color"];
		sigma_space = fn["sigma_space"];
    }

protected:
	String name;
	float gamma, contrast, sigma_color, sigma_space;
};

Ptr<TonemapDurand> createTonemapDurand(float gamma, float contrast, float sigma_color, float sigma_space)
{
	return new TonemapDurandImpl(gamma, contrast, sigma_color, sigma_space);
}

class TonemapReinhardDevlinImpl : public TonemapReinhardDevlin
{
public:
	TonemapReinhardDevlinImpl(float gamma, float intensity, float light_adapt, float color_adapt) : 
	    gamma(gamma), 
        intensity(intensity),
		light_adapt(light_adapt),
		color_adapt(color_adapt),
		name("TonemapReinhardDevlin")
	{
	}

	void process(InputArray _src, OutputArray _dst)
	{
		Mat src = _src.getMat();
		CV_Assert(!src.empty());
		_dst.create(src.size(), CV_32FC3);
		Mat img = _dst.getMat();
		
		Ptr<TonemapLinear> linear = createTonemapLinear(1.0f);
		linear->process(src, img);
		
		Mat gray_img;
		cvtColor(img, gray_img, COLOR_RGB2GRAY);
		Mat log_img;
		log(gray_img, log_img);

		float log_mean = (float)sum(log_img)[0] / log_img.total();
		double log_min, log_max;
		minMaxLoc(log_img, &log_min, &log_max);
		log_img.release();

		double key = (float)((log_max - log_mean) / (log_max - log_min));
		float map_key = 0.3f + 0.7f * pow((float)key, 1.4f);
		intensity = exp(-intensity);
		Scalar chan_mean = mean(img);
		float gray_mean = (float)mean(gray_img)[0];

		std::vector<Mat> channels(3);
		split(img, channels);

		for(int i = 0; i < 3; i++) {
			float global = color_adapt * (float)chan_mean[i] + (1.0f - color_adapt) * gray_mean;
			Mat adapt = color_adapt * channels[i] + (1.0f - color_adapt) * gray_img;
			adapt = light_adapt * adapt + (1.0f - light_adapt) * global;
			pow(intensity * adapt, map_key, adapt);
			channels[i] = channels[i].mul(1.0f / (adapt + channels[i]));		
		}
		gray_img.release();
		merge(channels, img);
		
		linear->setGamma(gamma);
		linear->process(img, img);
	}

	float getGamma() const { return gamma; }
	void setGamma(float val) { gamma = val; }

	float getIntensity() const { return intensity; }
	void setIntensity(float val) { intensity = val; }

	float getLightAdaptation() const { return light_adapt; }
	void setLightAdaptation(float val) { light_adapt = val; }

	float getColorAdaptation() const { return color_adapt; }
	void setColorAdaptation(float val) { color_adapt = val; }

	void write(FileStorage& fs) const
    {
        fs << "name" << name
           << "gamma" << gamma
		   << "intensity" << intensity 
		   << "light_adapt" << light_adapt 
		   << "color_adapt" << color_adapt;
    }

    void read(const FileNode& fn)
    {
        FileNode n = fn["name"];
        CV_Assert(n.isString() && String(n) == name);
        gamma = fn["gamma"];
		intensity = fn["intensity"];
		light_adapt = fn["light_adapt"];
		color_adapt = fn["color_adapt"];
    }

protected:
	String name;
	float gamma, intensity, light_adapt, color_adapt;
};

Ptr<TonemapReinhardDevlin> createTonemapReinhardDevlin(float gamma, float contrast, float sigma_color, float sigma_space)
{
	return new TonemapReinhardDevlinImpl(gamma, contrast, sigma_color, sigma_space);
}

}