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

#include "seamless_cloning.hpp"

using namespace std;
using namespace cv;

void cv::seamlessClone(InputArray _src, InputArray _dst, InputArray _mask, Point p, OutputArray _blend, int flags)
{
    const Mat src  = _src.getMat();
    const Mat dest = _dst.getMat();
    const Mat mask = _mask.getMat();
    _blend.create(dest.size(), CV_8UC3);
    Mat blend = _blend.getMat();

    int minx = INT_MAX, miny = INT_MAX, maxx = INT_MIN, maxy = INT_MIN;
    int h = mask.size().height;
    int w = mask.size().width;

    Mat gray = Mat(mask.size(),CV_8UC1);
    Mat dst_mask = Mat::zeros(dest.size(),CV_8UC1);
    Mat cs_mask = Mat::zeros(src.size(),CV_8UC3);
    Mat cd_mask = Mat::zeros(dest.size(),CV_8UC3);

    if(mask.channels() == 3)
        cvtColor(mask, gray, COLOR_BGR2GRAY );
    else
        gray = mask;

    for(int i=0;i<h;i++)
    {
        for(int j=0;j<w;j++)
        {
            if(gray.at<uchar>(i,j) == 255)
            {
                minx = std::min(minx,i);
                maxx = std::max(maxx,i);
                miny = std::min(miny,j);
                maxy = std::max(maxy,j);
            }
        }
    }

    int lenx = maxx - minx;
    int leny = maxy - miny;

    Mat patch = Mat::zeros(Size(leny, lenx), CV_8UC3);

    int minxd = p.y - lenx/2;
    int maxxd = p.y + lenx/2;
    int minyd = p.x - leny/2;
    int maxyd = p.x + leny/2;

    CV_Assert(minxd >= 0 && minyd >= 0 && maxxd <= dest.rows && maxyd <= dest.cols);

    Rect roi_d(minyd,minxd,leny,lenx);
    Rect roi_s(miny,minx,leny,lenx);

    Mat destinationROI = dst_mask(roi_d);
    Mat sourceROI = cs_mask(roi_s);

    gray(roi_s).copyTo(destinationROI);
    src(roi_s).copyTo(sourceROI,gray(roi_s));
    src(roi_s).copyTo(patch, gray(roi_s));

    destinationROI = cd_mask(roi_d);
    cs_mask(roi_s).copyTo(destinationROI);


    Cloning obj;
    obj.normalClone(dest,cd_mask,dst_mask,blend,flags);

}

void cv::colorChange(InputArray _src, InputArray _mask, OutputArray _dst, float r, float g, float b)
{
    Mat src  = _src.getMat();
    Mat mask  = _mask.getMat();
    _dst.create(src.size(), src.type());
    Mat blend = _dst.getMat();

    float red = r;
    float green = g;
    float blue = b;

    Mat gray = Mat::zeros(mask.size(),CV_8UC1);

    if(mask.channels() == 3)
        cvtColor(mask, gray, COLOR_BGR2GRAY );
    else
        gray = mask;

    Mat cs_mask = Mat::zeros(src.size(),CV_8UC3);

    src.copyTo(cs_mask,gray);

    Cloning obj;
    obj.localColorChange(src,cs_mask,gray,blend,red,green,blue);
}

void cv::illuminationChange(InputArray _src, InputArray _mask, OutputArray _dst, float a, float b)
{

    Mat src  = _src.getMat();
    Mat mask  = _mask.getMat();
    _dst.create(src.size(), src.type());
    Mat blend = _dst.getMat();
    float alpha = a;
    float beta = b;

    Mat gray = Mat::zeros(mask.size(),CV_8UC1);

    if(mask.channels() == 3)
        cvtColor(mask, gray, COLOR_BGR2GRAY );
    else
        gray = mask;

    Mat cs_mask = Mat::zeros(src.size(),CV_8UC3);

    src.copyTo(cs_mask,gray);

    Cloning obj;
    obj.illuminationChange(src,cs_mask,gray,blend,alpha,beta);

}

void cv::textureFlattening(InputArray _src, InputArray _mask, OutputArray _dst,
                           float low_threshold, float high_threshold, int kernel_size)
{

    Mat src  = _src.getMat();
    Mat mask  = _mask.getMat();
    _dst.create(src.size(), src.type());
    Mat blend = _dst.getMat();

    Mat gray = Mat::zeros(mask.size(),CV_8UC1);

    if(mask.channels() == 3)
        cvtColor(mask, gray, COLOR_BGR2GRAY );
    else
        gray = mask;

    Mat cs_mask = Mat::zeros(src.size(),CV_8UC3);

    src.copyTo(cs_mask,gray);

    Cloning obj;
    obj.textureFlatten(src,cs_mask,gray,low_threshold,high_threshold,kernel_size,blend);
}
