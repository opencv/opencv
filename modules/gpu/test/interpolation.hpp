/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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

#ifndef __OPENCV_TEST_INTERPOLATION_HPP__
#define __OPENCV_TEST_INTERPOLATION_HPP__

template <typename T> T readVal(const cv::Mat& src, int y, int x, int c, int border_type, cv::Scalar borderVal = cv::Scalar())
{
    if (border_type == cv::BORDER_CONSTANT)
        return (y >= 0 && y < src.rows && x >= 0 && x < src.cols) ? src.at<T>(y, x * src.channels() + c) : cv::saturate_cast<T>(borderVal.val[c]);

    return src.at<T>(cv::borderInterpolate(y, src.rows, border_type), cv::borderInterpolate(x, src.cols, border_type) * src.channels() + c);
}

template <typename T> struct NearestInterpolator
{
    static T getValue(const cv::Mat& src, float y, float x, int c, int border_type, cv::Scalar borderVal = cv::Scalar())
    {
        return readVal<T>(src, cvFloor(y), cvFloor(x), c, border_type, borderVal);
    }
};

template <typename T> struct LinearInterpolator
{
    static T getValue(const cv::Mat& src, float y, float x, int c, int border_type, cv::Scalar borderVal = cv::Scalar())
    {
        x -= 0.5f;
        y -= 0.5f;

        int x1 = cvFloor(x);
        int y1 = cvFloor(y);
        int x2 = x1 + 1;
        int y2 = y1 + 1;

        float res = 0;

        res += readVal<T>(src, y1, x1, c, border_type, borderVal) * ((x2 - x) * (y2 - y));
        res += readVal<T>(src, y1, x2, c, border_type, borderVal) * ((x - x1) * (y2 - y));
        res += readVal<T>(src, y2, x1, c, border_type, borderVal) * ((x2 - x) * (y - y1));
        res += readVal<T>(src, y2, x2, c, border_type, borderVal) * ((x - x1) * (y - y1));

        return cv::saturate_cast<T>(res);
    }
};

template <typename T> struct CubicInterpolator
{
    static float getValue(float p[4], float x)
    {
        return static_cast<float>(p[1] + 0.5 * x * (p[2] - p[0] + x*(2.0*p[0] - 5.0*p[1] + 4.0*p[2] - p[3] + x*(3.0*(p[1] - p[2]) + p[3] - p[0]))));
    }

    static float getValue(float p[4][4], float x, float y)
    {
        float arr[4];

        arr[0] = getValue(p[0], x);
        arr[1] = getValue(p[1], x);
        arr[2] = getValue(p[2], x);
        arr[3] = getValue(p[3], x);

        return getValue(arr, y);
    }

    static T getValue(const cv::Mat& src, float y, float x, int c, int border_type, cv::Scalar borderVal = cv::Scalar())
    {
        int ix = cvRound(x);
        int iy = cvRound(y);

        float vals[4][4] =
        {
            {(float)readVal<T>(src, iy - 2, ix - 2, c, border_type, borderVal), (float)readVal<T>(src, iy - 2, ix - 1, c, border_type, borderVal), (float)readVal<T>(src, iy - 2, ix, c, border_type, borderVal), (float)readVal<T>(src, iy - 2, ix + 1, c, border_type, borderVal)},
            {(float)readVal<T>(src, iy - 1, ix - 2, c, border_type, borderVal), (float)readVal<T>(src, iy - 1, ix - 1, c, border_type, borderVal), (float)readVal<T>(src, iy - 1, ix, c, border_type, borderVal), (float)readVal<T>(src, iy - 1, ix + 1, c, border_type, borderVal)},
            {(float)readVal<T>(src, iy    , ix - 2, c, border_type, borderVal), (float)readVal<T>(src, iy    , ix - 1, c, border_type, borderVal), (float)readVal<T>(src, iy    , ix, c, border_type, borderVal), (float)readVal<T>(src, iy    , ix + 1, c, border_type, borderVal)},
            {(float)readVal<T>(src, iy + 1, ix - 2, c, border_type, borderVal), (float)readVal<T>(src, iy + 1, ix - 1, c, border_type, borderVal), (float)readVal<T>(src, iy + 1, ix, c, border_type, borderVal), (float)readVal<T>(src, iy + 1, ix + 1, c, border_type, borderVal)},
        };

        return cv::saturate_cast<T>(getValue(vals, static_cast<float>((x - ix + 2.0) / 4.0), static_cast<float>((y - iy + 2.0) / 4.0)));
    }
};

#endif // __OPENCV_TEST_INTERPOLATION_HPP__
