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
#include "blenders.hpp"
#include "util.hpp"

using namespace std;
using namespace cv;

static const float WEIGHT_EPS = 1e-5f;

Ptr<Blender> Blender::createDefault(int type)
{
    if (type == NO)
        return new Blender();
    if (type == FEATHER)
        return new FeatherBlender();
    if (type == MULTI_BAND)
        return new MultiBandBlender();
    CV_Error(CV_StsBadArg, "unsupported blending method");
    return NULL;
}


void Blender::prepare(const vector<Point> &corners, const vector<Size> &sizes)
{
    prepare(resultRoi(corners, sizes));
}


void Blender::prepare(Rect dst_roi)
{
    dst_.create(dst_roi.size(), CV_16SC3);
    dst_.setTo(Scalar::all(0));
    dst_mask_.create(dst_roi.size(), CV_8U);
    dst_mask_.setTo(Scalar::all(0));
    dst_roi_ = dst_roi;
}


void Blender::feed(const Mat &img, const Mat &mask, Point tl) 
{
    CV_Assert(img.type() == CV_16SC3);
    CV_Assert(mask.type() == CV_8U);

    int dx = tl.x - dst_roi_.x;
    int dy = tl.y - dst_roi_.y;

    for (int y = 0; y < img.rows; ++y)
    {
        const Point3_<short> *src_row = img.ptr<Point3_<short> >(y);
        Point3_<short> *dst_row = dst_.ptr<Point3_<short> >(dy + y);

        const uchar *mask_row = mask.ptr<uchar>(y);
        uchar *dst_mask_row = dst_mask_.ptr<uchar>(dy + y);

        for (int x = 0; x < img.cols; ++x)
        {
            if (mask_row[x]) 
                dst_row[dx + x] = src_row[x];
            dst_mask_row[dx + x] |= mask_row[x];
        }
    }
}


void Blender::blend(Mat &dst, Mat &dst_mask)
{
    dst_.setTo(Scalar::all(0), dst_mask_ == 0);
    dst = dst_;
    dst_mask = dst_mask_;
    dst_.release();
    dst_mask_.release();
}


void FeatherBlender::prepare(Rect dst_roi)
{
    Blender::prepare(dst_roi);
    dst_weight_map_.create(dst_roi.size(), CV_32F);
    dst_weight_map_.setTo(0);
}


void FeatherBlender::feed(const Mat &img, const Mat &mask, Point tl)
{
    CV_Assert(img.type() == CV_16SC3);
    CV_Assert(mask.type() == CV_8U);

    createWeightMap(mask, sharpness_, weight_map_);
    int dx = tl.x - dst_roi_.x;
    int dy = tl.y - dst_roi_.y;

    for (int y = 0; y < img.rows; ++y)
    {
        const Point3_<short>* src_row = img.ptr<Point3_<short> >(y);
        Point3_<short>* dst_row = dst_.ptr<Point3_<short> >(dy + y);
        const float* weight_row = weight_map_.ptr<float>(y);
        float* dst_weight_row = dst_weight_map_.ptr<float>(dy + y);

        for (int x = 0; x < img.cols; ++x)               
        {
            dst_row[dx + x].x += static_cast<short>(src_row[x].x * weight_row[x]);
            dst_row[dx + x].y += static_cast<short>(src_row[x].y * weight_row[x]);
            dst_row[dx + x].z += static_cast<short>(src_row[x].z * weight_row[x]);
            dst_weight_row[dx + x] += weight_row[x];
        }
    }
}


void FeatherBlender::blend(Mat &dst, Mat &dst_mask)
{
    normalize(dst_weight_map_, dst_);
    dst_mask_ = dst_weight_map_ > WEIGHT_EPS;
    Blender::blend(dst, dst_mask);
}


void MultiBandBlender::prepare(Rect dst_roi)
{
    dst_roi_final_ = dst_roi;
    dst_roi.width += ((1 << num_bands_) - dst_roi.width % (1 << num_bands_)) % (1 << num_bands_);
    dst_roi.height += ((1 << num_bands_) - dst_roi.height % (1 << num_bands_)) % (1 << num_bands_);
    Blender::prepare(dst_roi);

    dst_pyr_laplace_.resize(num_bands_ + 1);
    dst_pyr_laplace_[0] = dst_;

    dst_band_weights_.resize(num_bands_ + 1);
    dst_band_weights_[0].create(dst_roi.size(), CV_32F);
    dst_band_weights_[0].setTo(0);

    for (int i = 1; i <= num_bands_; ++i)
    {
        dst_pyr_laplace_[i].create((dst_pyr_laplace_[i - 1].rows + 1) / 2, 
                                   (dst_pyr_laplace_[i - 1].cols + 1) / 2, CV_16SC3);
        dst_band_weights_[i].create((dst_band_weights_[i - 1].rows + 1) / 2,
                                    (dst_band_weights_[i - 1].cols + 1) / 2, CV_32F);
        dst_pyr_laplace_[i].setTo(Scalar::all(0));
        dst_band_weights_[i].setTo(0);
    }
}


void MultiBandBlender::feed(const Mat &img, const Mat &mask, Point tl)
{
    CV_Assert(img.type() == CV_16SC3);
    CV_Assert(mask.type() == CV_8U);

    int gap = 3 * (1 << num_bands_);
    Point tl_new(max(dst_roi_.x, tl.x - gap), 
                 max(dst_roi_.y, tl.y - gap));
    Point br_new(min(dst_roi_.br().x, tl.x + img.cols + gap), 
                 min(dst_roi_.br().y, tl.y + img.rows + gap));

    tl_new.x = dst_roi_.x + (((tl_new.x - dst_roi_.x) >> num_bands_) << num_bands_);
    tl_new.y = dst_roi_.y + (((tl_new.y - dst_roi_.y) >> num_bands_) << num_bands_);
    int width = br_new.x - tl_new.x;
    int height = br_new.y - tl_new.y;
    width += ((1 << num_bands_) - width % (1 << num_bands_)) % (1 << num_bands_);
    height += ((1 << num_bands_) - height % (1 << num_bands_)) % (1 << num_bands_);
    br_new.x = tl_new.x + width;
    br_new.y = tl_new.y + height;
    int dy = max(br_new.y - dst_roi_.br().y, 0);
    int dx = max(br_new.x - dst_roi_.br().x, 0);
    tl_new.x -= dx; br_new.x -= dx;
    tl_new.y -= dy; br_new.y -= dy;

    int top = tl.y - tl_new.y;
    int left = tl.x - tl_new.x;
    int bottom = br_new.y - tl.y - img.rows;
    int right = br_new.x - tl.x - img.cols;

    // Create the source image Laplacian pyramid
    vector<Mat> src_pyr_gauss(num_bands_ + 1);
    src_pyr_gauss[0] = img;
    copyMakeBorder(img, src_pyr_gauss[0], top, bottom, left, right, 
                   BORDER_REFLECT);
    for (int i = 0; i < num_bands_; ++i)
        pyrDown(src_pyr_gauss[i], src_pyr_gauss[i + 1]);
    vector<Mat> src_pyr_laplace;
    createLaplacePyr(src_pyr_gauss, src_pyr_laplace);
    src_pyr_gauss.clear();

    // Create the weight map Gaussian pyramid
    Mat weight_map;
    mask.convertTo(weight_map, CV_32F, 1./255.);
    vector<Mat> weight_pyr_gauss(num_bands_ + 1);
    copyMakeBorder(weight_map, weight_pyr_gauss[0], top, bottom, left, right, 
                   BORDER_CONSTANT);
    for (int i = 0; i < num_bands_; ++i)
        pyrDown(weight_pyr_gauss[i], weight_pyr_gauss[i + 1]);

    int y_tl = tl_new.y - dst_roi_.y;
    int y_br = br_new.y - dst_roi_.y;
    int x_tl = tl_new.x - dst_roi_.x;
    int x_br = br_new.x - dst_roi_.x;

    // Add weighted layer of the source image to the final Laplacian pyramid layer
    for (int i = 0; i <= num_bands_; ++i)
    {
        for (int y = y_tl; y < y_br; ++y)
        {
            int y_ = y - y_tl;
            const Point3_<short>* src_row = src_pyr_laplace[i].ptr<Point3_<short> >(y_);
            Point3_<short>* dst_row = dst_pyr_laplace_[i].ptr<Point3_<short> >(y);
            const float* weight_row = weight_pyr_gauss[i].ptr<float>(y_);
            float* dst_weight_row = dst_band_weights_[i].ptr<float>(y);

            for (int x = x_tl; x < x_br; ++x)               
            {
                int x_ = x - x_tl;
                dst_row[x].x += static_cast<short>(src_row[x_].x * weight_row[x_]);
                dst_row[x].y += static_cast<short>(src_row[x_].y * weight_row[x_]);
                dst_row[x].z += static_cast<short>(src_row[x_].z * weight_row[x_]);
                dst_weight_row[x] += weight_row[x_];
            }
        }
        x_tl /= 2; y_tl /= 2; 
        x_br /= 2; y_br /= 2;
    }    
}


void MultiBandBlender::blend(Mat &dst, Mat &dst_mask)
{
    for (int i = 0; i <= num_bands_; ++i)
        normalize(dst_band_weights_[i], dst_pyr_laplace_[i]);

    restoreImageFromLaplacePyr(dst_pyr_laplace_);

    dst_ = dst_pyr_laplace_[0];
    dst_ = dst_(Range(0, dst_roi_final_.height), Range(0, dst_roi_final_.width));
    dst_mask_ = dst_band_weights_[0] > WEIGHT_EPS;
    dst_mask_ = dst_mask_(Range(0, dst_roi_final_.height), Range(0, dst_roi_final_.width));
    dst_pyr_laplace_.clear();
    dst_band_weights_.clear();

    Blender::blend(dst, dst_mask);
}


//////////////////////////////////////////////////////////////////////////////
// Auxiliary functions

Rect resultRoi(const vector<Point> &corners, const vector<Size> &sizes)
{
    Point tl(numeric_limits<int>::max(), numeric_limits<int>::max());
    Point br(numeric_limits<int>::min(), numeric_limits<int>::min());

    CV_Assert(sizes.size() == corners.size());
    for (size_t i = 0; i < corners.size(); ++i)
    {
        tl.x = min(tl.x, corners[i].x);
        tl.y = min(tl.y, corners[i].y);
        br.x = max(br.x, corners[i].x + sizes[i].width);
        br.y = max(br.y, corners[i].y + sizes[i].height);
    }

    return Rect(tl, br);
}


void normalize(const Mat& weight, Mat& src)
{
    CV_Assert(weight.type() == CV_32F);
    CV_Assert(src.type() == CV_16SC3);
    for (int y = 0; y < src.rows; ++y)
    {
        Point3_<short> *row = src.ptr<Point3_<short> >(y);
        const float *weight_row = weight.ptr<float>(y);

        for (int x = 0; x < src.cols; ++x)
        {
            row[x].x = static_cast<short>(row[x].x / (weight_row[x] + WEIGHT_EPS));
            row[x].y = static_cast<short>(row[x].y / (weight_row[x] + WEIGHT_EPS));
            row[x].z = static_cast<short>(row[x].z / (weight_row[x] + WEIGHT_EPS));
        }
    }
}


void createWeightMap(const Mat &mask, float sharpness, Mat &weight)
{
    CV_Assert(mask.type() == CV_8U);
    distanceTransform(mask, weight, CV_DIST_L1, 3);
    threshold(weight * sharpness, weight, 1.f, 1.f, THRESH_TRUNC);
}


void createLaplacePyr(const vector<Mat> &pyr_gauss, vector<Mat> &pyr_laplace)
{
    if (pyr_gauss.size() == 0)
        return;
    pyr_laplace.resize(pyr_gauss.size());
    Mat tmp;
    for (size_t i = 0; i < pyr_laplace.size() - 1; ++i)
    {
        pyrUp(pyr_gauss[i + 1], tmp, pyr_gauss[i].size());
        subtract(pyr_gauss[i], tmp, pyr_laplace[i]);
    }
    pyr_laplace[pyr_laplace.size() - 1] = pyr_gauss[pyr_laplace.size() - 1].clone();
}


void restoreImageFromLaplacePyr(vector<Mat> &pyr)
{
    if (pyr.size() == 0)
        return;
    Mat tmp;
    for (size_t i = pyr.size() - 1; i > 0; --i)
    {
        pyrUp(pyr[i], tmp, pyr[i - 1].size());
        add(tmp, pyr[i - 1], pyr[i - 1]);
    }
}


