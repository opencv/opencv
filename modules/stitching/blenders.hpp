/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
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
#ifndef __OPENCV_BLENDERS_HPP__
#define __OPENCV_BLENDERS_HPP__

#include "precomp.hpp"

// Simple blender which puts one image over another
class Blender
{
public:
    enum { NO, FEATHER, MULTI_BAND };
    static cv::Ptr<Blender> createDefault(int type);

    void prepare(const std::vector<cv::Point> &corners, const std::vector<cv::Size> &sizes);
    virtual void prepare(cv::Rect dst_roi);
    virtual void feed(const cv::Mat &img, const cv::Mat &mask, cv::Point tl);
    virtual void blend(cv::Mat &dst, cv::Mat &dst_mask);

protected:
    cv::Mat dst_, dst_mask_;
    cv::Rect dst_roi_;
};


class FeatherBlender : public Blender
{
public:
    FeatherBlender(float sharpness = 0.02f) { setSharpness(sharpness); }
    float sharpness() const { return sharpness_; }
    void setSharpness(float val) { sharpness_ = val; }

    void prepare(cv::Rect dst_roi);
    void feed(const cv::Mat &img, const cv::Mat &mask, cv::Point tl);
    void blend(cv::Mat &dst, cv::Mat &dst_mask);

private:
    float sharpness_;
    cv::Mat weight_map_;
    cv::Mat dst_weight_map_;
};


class MultiBandBlender : public Blender
{
public:
    MultiBandBlender(int num_bands = 5) { setNumBands(num_bands); }
    int numBands() const { return actual_num_bands_; }
    void setNumBands(int val) { actual_num_bands_ = val; }

    void prepare(cv::Rect dst_roi);
    void feed(const cv::Mat &img, const cv::Mat &mask, cv::Point tl);
    void blend(cv::Mat &dst, cv::Mat &dst_mask);

private:
    int actual_num_bands_, num_bands_;
    std::vector<cv::Mat> dst_pyr_laplace_;
    std::vector<cv::Mat> dst_band_weights_;
    cv::Rect dst_roi_final_;
};


//////////////////////////////////////////////////////////////////////////////
// Auxiliary functions

void normalize(const cv::Mat& weight, cv::Mat& src);

void createWeightMap(const cv::Mat& mask, float sharpness, cv::Mat& weight);

void createLaplacePyr(const std::vector<cv::Mat>& pyr_gauss, std::vector<cv::Mat>& pyr_laplace);

// Restores source image in-place (result will be stored in pyr[0])
void restoreImageFromLaplacePyr(std::vector<cv::Mat>& pyr);

#endif // __OPENCV_BLENDERS_HPP__
