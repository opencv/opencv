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

Tonemap::Tonemap(float gamma) : gamma(gamma)
{
}

Tonemap::~Tonemap()
{
}

void Tonemap::process(InputArray src, OutputArray dst)
{
	Mat srcMat = src.getMat();
    CV_Assert(!srcMat.empty());
    dst.create(srcMat.size(), CV_32FC3);
    img = dst.getMat();
	srcMat.copyTo(img);
	linearMap();
	tonemap();
	gammaCorrection();
}

void Tonemap::linearMap()
{
	double min, max;
    minMaxLoc(img, &min, &max);
    if(max - min > DBL_EPSILON) {
        img = (img - min) / (max - min);
    }
}

void Tonemap::gammaCorrection()
{
	pow(img, 1.0f / gamma, img);
}

void TonemapLinear::tonemap()
{
}

TonemapLinear::TonemapLinear(float gamma) : Tonemap(gamma)
{
}

TonemapDrago::TonemapDrago(float gamma, float bias) :
    Tonemap(gamma),
	bias(bias)
{
}

void TonemapDrago::tonemap() 
{
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
	linearMap();
}

TonemapDurand::TonemapDurand(float gamma, float contrast, float sigma_color, float sigma_space) : 
    Tonemap(gamma),			
	contrast(contrast),
	sigma_color(sigma_color),
	sigma_space(sigma_space)
{
}

void TonemapDurand::tonemap()
{
	Mat gray_img;
    cvtColor(img, gray_img, COLOR_RGB2GRAY);
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
    split(img, channels);
    for(int i = 0; i < 3; i++) {
        channels[i] = channels[i].mul(map_img);
    }
    merge(channels, img);
}

TonemapReinhardDevlin::TonemapReinhardDevlin(float gamma, float intensity, float color_adapt, float light_adapt) : 
    Tonemap(gamma),			
	intensity(intensity),
	color_adapt(color_adapt),
	light_adapt(light_adapt)
{
}

void TonemapReinhardDevlin::tonemap()
{
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
	linearMap();
}

Ptr<Tonemap> Tonemap::create(const String& TonemapType)
{
    return Algorithm::create<Tonemap>("Tonemap." + TonemapType);
}

CV_INIT_ALGORITHM(TonemapLinear, "Tonemap.Linear",
				  obj.info()->addParam(obj, "gamma", obj.gamma));

CV_INIT_ALGORITHM(TonemapDrago, "Tonemap.Drago",
                  obj.info()->addParam(obj, "gamma", obj.gamma);
                  obj.info()->addParam(obj, "bias", obj.bias));

CV_INIT_ALGORITHM(TonemapDurand, "Tonemap.Durand",
				  obj.info()->addParam(obj, "gamma", obj.gamma);
                  obj.info()->addParam(obj, "contrast", obj.contrast);
				  obj.info()->addParam(obj, "sigma_color", obj.sigma_color);
                  obj.info()->addParam(obj, "sigma_space", obj.sigma_space));

CV_INIT_ALGORITHM(TonemapReinhardDevlin, "Tonemap.ReinhardDevlin",
				  obj.info()->addParam(obj, "gamma", obj.gamma);
                  obj.info()->addParam(obj, "intensity", obj.intensity);
				  obj.info()->addParam(obj, "color_adapt", obj.color_adapt);
                  obj.info()->addParam(obj, "light_adapt", obj.light_adapt));
}
