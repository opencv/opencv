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
// Copyright (C) 2015, Pavel Vlasanek, all rights reserved.
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

#ifndef __OPENCV_FUZZY_H__
#define __OPENCV_FUZZY_H__

#include "opencv2/core.hpp"

namespace cv
{

namespace ft
{
    enum
    {
        LINEAR = 1,
        SINUS = 2
    };

    enum
    {
        ONE_STEP = 1,
        MULTI_STEP = 2,
        ITERATIVE = 3
    };

    CV_EXPORTS void FT02D_components(cv::InputArray matrix, cv::InputArray kernel, cv::OutputArray components, cv::InputArray mask);
    CV_EXPORTS void FT02D_components(cv::InputArray matrix, cv::InputArray kernel, cv::OutputArray components);
    CV_EXPORTS void FT02D_inverseFT(cv::InputArray components, cv::InputArray kernel, cv::OutputArray output, int width, int height);
    CV_EXPORTS void FT02D_process(const cv::Mat &image, const cv::Mat &kernel, cv::Mat &output, const cv::Mat &mask);

    CV_EXPORTS int FT02D_check(const cv::Mat &image, const cv::Mat &kernel, cv::Mat &output, const cv::Mat &mask, cv::Mat &outputMask, bool firstStop = false);

    CV_EXPORTS void createKernel(cv::InputArray A, cv::InputArray B, cv::OutputArray kernel, const int chn = 1);
    CV_EXPORTS void createKernel(int function, int radius, cv::OutputArray kernel, const int chn = 1);

    CV_EXPORTS void inpaint(const cv::Mat &image, const cv::Mat &mask, cv::Mat &output, int radius = 2, int function = ft::LINEAR, int algorithm = ft::ONE_STEP);
    CV_EXPORTS void filter(const cv::Mat &image, const cv::Mat &kernel, cv::Mat &output);
}

}

#endif // __OPENCV_FUZZY_H__
