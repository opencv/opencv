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

#ifndef __OPENCV_STITCHING_BLENDERS_HPP__
#define __OPENCV_STITCHING_BLENDERS_HPP__

#include "opencv2/core/core.hpp"

namespace cv {
namespace detail {

// Simple blender which puts one image over another
class CV_EXPORTS Blender
{
public:
    virtual ~Blender() {}

    enum { NO, FEATHER, MULTI_BAND };
    static Ptr<Blender> createDefault(int type, bool try_gpu = false);

    void prepare(const std::vector<Point> &corners, const std::vector<Size> &sizes);    
    virtual void prepare(Rect dst_roi);
    virtual void feed(const Mat &img, const Mat &mask, Point tl);
    virtual void blend(Mat &dst, Mat &dst_mask);

protected:
    Mat dst_, dst_mask_;
    Rect dst_roi_;
};


class CV_EXPORTS FeatherBlender : public Blender
{
public:
    FeatherBlender(float sharpness = 0.02f) { setSharpness(sharpness); }
    float sharpness() const { return sharpness_; }
    void setSharpness(float val) { sharpness_ = val; }

    void prepare(Rect dst_roi);
    void feed(const Mat &img, const Mat &mask, Point tl);
    void blend(Mat &dst, Mat &dst_mask);

private:
    float sharpness_;
    Mat weight_map_;
    Mat dst_weight_map_;
};


class CV_EXPORTS MultiBandBlender : public Blender
{
public:
    MultiBandBlender(int try_gpu = false, int num_bands = 5);
    int numBands() const { return actual_num_bands_; }
    void setNumBands(int val) { actual_num_bands_ = val; }

    void prepare(Rect dst_roi);
    void feed(const Mat &img, const Mat &mask, Point tl);
    void blend(Mat &dst, Mat &dst_mask);

private:
    int actual_num_bands_, num_bands_;
    std::vector<Mat> dst_pyr_laplace_;
    std::vector<Mat> dst_band_weights_;
    Rect dst_roi_final_;
    bool can_use_gpu_;
};


//////////////////////////////////////////////////////////////////////////////
// Auxiliary functions

void CV_EXPORTS normalizeUsingWeightMap(const Mat& weight, Mat& src);

void CV_EXPORTS createWeightMap(const Mat& mask, float sharpness, Mat& weight);

void CV_EXPORTS createLaplacePyr(const Mat &img, int num_levels, std::vector<Mat>& pyr);

void CV_EXPORTS createLaplacePyrGpu(const Mat &img, int num_levels, std::vector<Mat>& pyr);

// Restores source image
void CV_EXPORTS restoreImageFromLaplacePyr(std::vector<Mat>& pyr);

} // namespace detail
} // namespace cv

#endif // __OPENCV_STITCHING_BLENDERS_HPP__
