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

#ifndef CV_SEAMLESS_CLONING_HPP___
#define CV_SEAMLESS_CLONING_HPP___

#include "precomp.hpp"
#include "opencv2/photo.hpp"

#include <vector>

namespace cv
{

    class Cloning
    {
        public:
            void normalClone(const cv::Mat& destination, const cv::Mat &mask, cv::Mat &wmask, cv::Mat &cloned, int flag);
            void illuminationChange(cv::Mat &I, cv::Mat &mask, cv::Mat &wmask, cv::Mat &cloned, float alpha, float beta);
            void localColorChange(cv::Mat &I, cv::Mat &mask, cv::Mat &wmask, cv::Mat &cloned, float red_mul, float green_mul, float blue_mul);
            void textureFlatten(cv::Mat &I, cv::Mat &mask, cv::Mat &wmask, float low_threshold, float high_threhold, int kernel_size, cv::Mat &cloned);

        protected:

            void initVariables(const cv::Mat &destination, const cv::Mat &binaryMask);
            void computeDerivatives(const cv::Mat &destination, const cv::Mat &patch, cv::Mat &binaryMask);
            void scalarProduct(cv::Mat mat, float r, float g, float b);
            void poisson(const cv::Mat &destination);
            void evaluate(const cv::Mat &I, cv::Mat &wmask, const cv::Mat &cloned);
            void dst(const Mat& src, Mat& dest, bool invert = false);
            void solve(const Mat &img, Mat& mod_diff, Mat &result);

            void poissonSolver(const cv::Mat &img, cv::Mat &gxx , cv::Mat &gyy, cv::Mat &result);

            void arrayProduct(const cv::Mat& lhs, const cv::Mat& rhs, cv::Mat& result) const;

            void computeGradientX(const cv::Mat &img, cv::Mat &gx);
            void computeGradientY(const cv::Mat &img, cv::Mat &gy);
            void computeLaplacianX(const cv::Mat &img, cv::Mat &gxx);
            void computeLaplacianY(const cv::Mat &img, cv::Mat &gyy);

        private:
            std::vector <cv::Mat> rgbx_channel, rgby_channel, output;
            cv::Mat destinationGradientX, destinationGradientY;
            cv::Mat patchGradientX, patchGradientY;
            cv::Mat binaryMaskFloat, binaryMaskFloatInverted;

            std::vector<float> filter_X, filter_Y;
    };
}
#endif