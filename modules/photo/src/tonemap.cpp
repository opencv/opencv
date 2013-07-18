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

static float getParam(const std::vector<float>& params, size_t i, float defval) 
{
    if(params.size() > i) {
        return params[i];
    } else {
        return defval;
    }       
}

static void DragoMap(Mat& src_img, Mat &dst_img, const std::vector<float>& params)
{
    float bias_value = getParam(params, 1, 0.85f);
    Mat gray_img;
    cvtColor(src_img, gray_img, COLOR_RGB2GRAY);
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
    pow(gray_img / (float)max, log(bias_value) / log(0.5f), div);
    log(2.0f + 8.0f * div, div);
    map = map.mul(1.0f / div);
    map = map.mul(1.0f / gray_img);
    div.release();
    gray_img.release();

    std::vector<Mat> channels(3);
    split(src_img, channels);
    for(int i = 0; i < 3; i++) {
        channels[i] = channels[i].mul(map);
    }
    map.release();
    merge(channels, dst_img);
}

static void ReinhardDevlinMap(Mat& src_img, Mat &dst_img, const std::vector<float>& params)
{
    float intensity   = getParam(params, 1, 0.0f);
    float color_adapt = getParam(params, 2, 0.0f);
    float light_adapt = getParam(params, 3, 1.0f);

    Mat gray_img;
    cvtColor(src_img, gray_img, COLOR_RGB2GRAY);
    Mat log_img;
    log(gray_img, log_img);

    float log_mean = (float)sum(log_img)[0] / log_img.total();
    double log_min, log_max;
    minMaxLoc(log_img, &log_min, &log_max);
    log_img.release();

    double key = (float)((log_max - log_mean) / (log_max - log_min));
    float map_key = 0.3f + 0.7f * pow((float)key, 1.4f);
    intensity = exp(-intensity);
    Scalar chan_mean = mean(src_img);
    float gray_mean = (float)mean(gray_img)[0];

    std::vector<Mat> channels(3);
    split(src_img, channels);

    for(int i = 0; i < 3; i++) {
        float global = color_adapt * (float)chan_mean[i] + (1.0f - color_adapt) * gray_mean;
        Mat adapt = color_adapt * channels[i] + (1.0f - color_adapt) * gray_img;
        adapt = light_adapt * adapt + (1.0f - light_adapt) * global;
        pow(intensity * adapt, map_key, adapt);
        channels[i] = channels[i].mul(1.0f / (adapt + channels[i]));		
    }
    gray_img.release();
    merge(channels, dst_img);
}

static void DurandMap(Mat& src_img, Mat& dst_img, const std::vector<float>& params)
{
    float contrast   = getParam(params, 1, 4.0f);
    float sigma_color = getParam(params, 2, 2.0f);
    float sigma_space = getParam(params, 3, 2.0f);

    Mat gray_img;
    cvtColor(src_img, gray_img, COLOR_RGB2GRAY);
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
    split(src_img, channels);
    for(int i = 0; i < 3; i++) {
        channels[i] = channels[i].mul(map_img);
    }
    merge(channels, dst_img);
}

void tonemap(InputArray _src, OutputArray _dst, int algorithm,
             const std::vector<float>& params)
{
    typedef void (*tonemap_func)(Mat&, Mat&, const std::vector<float>&);
    tonemap_func functions[TONEMAP_COUNT] = {
        NULL, DragoMap, ReinhardDevlinMap, DurandMap};

    Mat src = _src.getMat();
    if(src.empty()) {
        CV_Error(Error::StsBadArg, "Empty input image");
    }
    if(algorithm < 0 || algorithm >= TONEMAP_COUNT) {
        CV_Error(Error::StsBadArg, "Wrong algorithm index");
    }

    _dst.create(src.size(), CV_32FC3);
    Mat dst = _dst.getMat();
    src.copyTo(dst);

    double min, max;
    minMaxLoc(dst, &min, &max);
    if(max - min < 1e-10f) {
        return;
    }
    dst = (dst - min) / (max - min);
    if(functions[algorithm]) {
        functions[algorithm](dst, dst, params);
    }
    minMaxLoc(dst, &min, &max);
    dst = (dst - min) / (max - min);
    float gamma = getParam(params, 0, 1.0f);		
    pow(dst, 1.0f / gamma, dst);			
}
}
