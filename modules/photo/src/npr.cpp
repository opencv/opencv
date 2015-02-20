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

#include "precomp.hpp"
#include "opencv2/photo.hpp"
#include <iostream>
#include <stdlib.h>

#include "npr.hpp"

using namespace std;
using namespace cv;

void cv::edgePreservingFilter(InputArray _src, OutputArray _dst, int flags, float sigma_s, float sigma_r)
{
    Mat I = _src.getMat();
    _dst.create(I.size(), CV_8UC3);
    Mat dst = _dst.getMat();

    int h = I.size().height;
    int w = I.size().width;

    Mat res = Mat(h,w,CV_32FC3);
    dst.convertTo(res,CV_32FC3,1.0/255.0);

    Domain_Filter obj;

    Mat img = Mat(I.size(),CV_32FC3);
    I.convertTo(img,CV_32FC3,1.0/255.0);

    obj.filter(img, res, sigma_s, sigma_r, flags);

    convertScaleAbs(res, dst, 255,0);
}

void cv::detailEnhance(InputArray _src, OutputArray _dst, float sigma_s, float sigma_r)
{
    Mat I = _src.getMat();
    _dst.create(I.size(), CV_8UC3);
    Mat dst = _dst.getMat();

    int h = I.size().height;
    int w = I.size().width;
    float factor = 3.0f;

    Mat img = Mat(I.size(),CV_32FC3);
    I.convertTo(img,CV_32FC3,1.0/255.0);

    Mat res = Mat(h,w,CV_32FC1);
    dst.convertTo(res,CV_32FC3,1.0/255.0);

    Mat result = Mat(img.size(),CV_32FC3);
    Mat lab = Mat(img.size(),CV_32FC3);
    vector <Mat> lab_channel;

    cvtColor(img,lab,COLOR_BGR2Lab);
    split(lab,lab_channel);

    Mat L = Mat(img.size(),CV_32FC1);

    lab_channel[0].convertTo(L,CV_32FC1,1.0/255.0);

    Domain_Filter obj;

    obj.filter(L, res, sigma_s, sigma_r, 1);

    Mat detail = Mat(h,w,CV_32FC1);

    detail = L - res;
    multiply(detail,factor,detail);
    L = res + detail;

    L.convertTo(lab_channel[0],CV_32FC1,255);

    merge(lab_channel,lab);

    cvtColor(lab,result,COLOR_Lab2BGR);
    result.convertTo(dst,CV_8UC3,255);
}

void cv::pencilSketch(InputArray _src, OutputArray _dst1, OutputArray _dst2, float sigma_s, float sigma_r, float shade_factor)
{
    Mat I = _src.getMat();
    _dst1.create(I.size(), CV_8UC1);
    Mat dst1 = _dst1.getMat();

    _dst2.create(I.size(), CV_8UC3);
    Mat dst2 = _dst2.getMat();

    Mat img = Mat(I.size(),CV_32FC3);
    I.convertTo(img,CV_32FC3,1.0/255.0);

    Domain_Filter obj;

    Mat sketch = Mat(I.size(),CV_32FC1);
    Mat color_sketch = Mat(I.size(),CV_32FC3);

    obj.pencil_sketch(img, sketch, color_sketch, sigma_s, sigma_r, shade_factor);

    sketch.convertTo(dst1,CV_8UC1,255);
    color_sketch.convertTo(dst2,CV_8UC3,255);

}

void cv::stylization(InputArray _src, OutputArray _dst, float sigma_s, float sigma_r)
{
    Mat I = _src.getMat();
    _dst.create(I.size(), CV_8UC3);
    Mat dst = _dst.getMat();

    Mat img = Mat(I.size(),CV_32FC3);
    I.convertTo(img,CV_32FC3,1.0/255.0);

    int h = img.size().height;
    int w = img.size().width;

    Mat res = Mat(h,w,CV_32FC3);
    Mat magnitude = Mat(h,w,CV_32FC1);

    Domain_Filter obj;
    obj.filter(img, res, sigma_s, sigma_r, NORMCONV_FILTER);

    obj.find_magnitude(res,magnitude);

    Mat stylized = Mat(h,w,CV_32FC3);

    vector <Mat> temp;
    split(res,temp);
    multiply(temp[0],magnitude,temp[0]);
    multiply(temp[1],magnitude,temp[1]);
    multiply(temp[2],magnitude,temp[2]);
    merge(temp,stylized);

    stylized.convertTo(dst,CV_8UC3,255);
}
