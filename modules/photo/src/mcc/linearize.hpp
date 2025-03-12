// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
//
//                       License Agreement
//              For Open Source Computer Vision Library
//
// Copyright(C) 2020, Huawei Technologies Co.,Ltd. All rights reserved.
// Third party copyrights are property of their respective owners.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//             http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Author: Longbu Wang <wanglongbu@huawei.com.com>
//         Jinheng Zhang <zhangjinheng1@huawei.com>
//         Chenqi Shan <shanchenqi@huawei.com>

#ifndef __OPENCV_CCM_LINEARIZE_HPP__
#define __OPENCV_CCM_LINEARIZE_HPP__

#include <opencv2/core.hpp>
#include <map>
#include "color.hpp"
#include "opencv2/ccm.hpp"
namespace cv {
namespace ccm {

/** @brief Polyfit model.
*/
class Polyfit
{
public:
    int deg;
    Mat p;
    Polyfit() {};

    /** @brief Polyfit method.
    https://en.wikipedia.org/wiki/Polynomial_regression
    polynomial: yi = a0 + a1*xi + a2*xi^2 + ... + an*xi^deg (i = 1,2,...,n)
    and deduct: Ax = y
    */
    Polyfit(Mat x, Mat y, int deg);
    virtual ~Polyfit() {};
    Mat operator()(const Mat& inp);

private:
    double fromEW(double x);
};

/** @brief Logpolyfit model.
*/
class LogPolyfit

{
public:
    int deg;
    Polyfit p;

    LogPolyfit() {};

    /** @brief Logpolyfit method.
    */
    LogPolyfit(Mat x, Mat y, int deg);
    virtual ~LogPolyfit() {};
    Mat operator()(const Mat& inp);
};

/** @brief Linearization base.
*/

class Linear
{
public:
    Linear() {};
    virtual ~Linear() {};

    /** @brief Inference.
        @param inp the input array, type of cv::Mat.
    */
    virtual Mat linearize(Mat inp);
    /* *\brief Evaluate linearization model.
    */
    virtual void value(void) {};
};

/** @brief Linearization identity.
           make no change.
*/
class LinearIdentity : public Linear
{};

/** @brief Linearization gamma correction.
*/
class LinearGamma : public Linear
{
public:
    double gamma;

    LinearGamma(double gamma_)
        : gamma(gamma_) {};

    Mat linearize(Mat inp) CV_OVERRIDE;
};

/** @brief Linearization.
           Grayscale polynomial fitting.
*/
template <class T>
class LinearGray : public Linear
{
public:
    int deg;
    T p;
    LinearGray(int deg_, Mat src, Color dst, Mat mask, RGBBase_ cs)
        : deg(deg_)
    {
        dst.getGray();
        Mat lear_gray_mask = mask & dst.grays;

        // the grayscale function is approximate for src is in relative color space.
        src = rgb2gray(maskCopyTo(src, lear_gray_mask));
        Mat dst_ = maskCopyTo(dst.toGray(cs.io), lear_gray_mask);
        calc(src, dst_);
    }

    /** @brief monotonically increase is not guaranteed.
        @param src the input array, type of cv::Mat.
        @param dst the input array, type of cv::Mat.
    */
    void calc(const Mat& src, const Mat& dst)
    {
        p = T(src, dst, deg);
    };

    Mat linearize(Mat inp) CV_OVERRIDE
    {
        return p(inp);
    };
};

/** @brief Linearization.
           Fitting channels respectively.
*/
template <class T>
class LinearColor : public Linear
{
public:
    int deg;
    T pr;
    T pg;
    T pb;

    LinearColor(int deg_, Mat src_, Color dst, Mat mask, RGBBase_ cs)
        : deg(deg_)
    {
        Mat src = maskCopyTo(src_, mask);
        Mat dst_ = maskCopyTo(dst.to(*cs.l).colors, mask);
        calc(src, dst_);
    }

    void calc(const Mat& src, const Mat& dst)
    {
        Mat schannels[3];
        Mat dchannels[3];
        split(src, schannels);
        split(dst, dchannels);
        pr = T(schannels[0], dchannels[0], deg);
        pg = T(schannels[1], dchannels[1], deg);
        pb = T(schannels[2], dchannels[2], deg);
    };

    Mat linearize(Mat inp) CV_OVERRIDE
    {
        Mat channels[3];
        split(inp, channels);
        std::vector<Mat> channel;
        Mat res;
        merge(std::vector<Mat> { pr(channels[0]), pg(channels[1]), pb(channels[2]) }, res);
        return res;
    };
};

/** @brief Get linearization method.
           used in ccm model.
    @param gamma used in LinearGamma.
    @param deg degrees.
    @param src the input array, type of cv::Mat.
    @param dst the input array, type of cv::Mat.
    @param mask the input array, type of cv::Mat.
    @param cs type of RGBBase_.
    @param linear_type type of linear.
*/

std::shared_ptr<Linear> getLinear(double gamma, int deg, Mat src, Color dst, Mat mask, RGBBase_ cs, LINEAR_TYPE linear_type);

}
}  // namespace cv::ccm

#endif
