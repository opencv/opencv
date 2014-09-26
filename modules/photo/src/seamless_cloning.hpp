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
            void normal_clone(const cv::Mat &I, const cv::Mat &mask, const cv::Mat &wmask, cv::Mat &cloned, int num);
            void illum_change(cv::Mat &I, cv::Mat &mask, cv::Mat &wmask, cv::Mat &cloned, float alpha, float beta);
            void local_color_change(cv::Mat &I, cv::Mat &mask, cv::Mat &wmask, cv::Mat &cloned, float red_mul, float green_mul, float blue_mul);
            void texture_flatten(cv::Mat &I, cv::Mat &mask, cv::Mat &wmask, double low_threshold, double high_threhold, int kernel_size, cv::Mat &cloned);

        protected:

            void init_var(const cv::Mat &I, const cv::Mat &wmask);
            void initialization(const cv::Mat &I, const cv::Mat &mask, const cv::Mat &wmask);
            void scalar_product(cv::Mat mat, float r, float g, float b);
            void array_product(cv::Mat mat1, cv::Mat mat2, cv::Mat mat3);
            void poisson(const cv::Mat &I, const cv::Mat &gx, const cv::Mat &gy, const cv::Mat &sx, const cv::Mat &sy);
            void evaluate(const cv::Mat &I, const cv::Mat &wmask, const cv::Mat &cloned);
            void getGradientx(const cv::Mat &img, cv::Mat &gx);
            void getGradienty(const cv::Mat &img, cv::Mat &gy);
            void lapx(const cv::Mat &img, cv::Mat &gxx);
            void lapy(const cv::Mat &img, cv::Mat &gyy);
            void dst(double *mod_diff, double *sineTransform,int h,int w);
            void idst(double *mod_diff, double *sineTransform,int h,int w);
            void transpose(double *mat, double *mat_t,int h,int w);
            void solve(const cv::Mat &img, double *mod_diff, cv::Mat &result);
            void poisson_solver(const cv::Mat &img, cv::Mat &gxx , cv::Mat &gyy, cv::Mat &result);


        private:
            std::vector <cv::Mat> rgb_channel, rgbx_channel, rgby_channel, output;
            cv::Mat grx, gry, sgx, sgy, srx32, sry32, grx32, gry32, smask, smask1;

    };
}
#endif