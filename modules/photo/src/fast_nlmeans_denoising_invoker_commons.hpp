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
// Third party copyrights are property of their respective icvers.
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

#ifndef __OPENCV_FAST_NLMEANS_DENOISING_INVOKER_COMMONS_HPP__
#define __OPENCV_FAST_NLMEANS_DENOISING_INVOKER_COMMONS_HPP__

using namespace cv;

template <typename T> static inline int calcDist(const T a, const T b);

template <> inline int calcDist(const uchar a, const uchar b)
{
    return (a-b) * (a-b);
}

template <> inline int calcDist(const Vec2b a, const Vec2b b)
{
    return (a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]);
}

template <> inline int calcDist(const Vec3b a, const Vec3b b)
{
    return (a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]) + (a[2]-b[2])*(a[2]-b[2]);
}

template <typename T> static inline int calcDist(const Mat& m, int i1, int j1, int i2, int j2)
{
    const T a = m.at<T>(i1, j1);
    const T b = m.at<T>(i2, j2);
    return calcDist<T>(a,b);
}

template <typename T> static inline int calcUpDownDist(T a_up, T a_down, T b_up, T b_down)
{
    return calcDist(a_down, b_down) - calcDist(a_up, b_up);
}

template <> inline int calcUpDownDist(uchar a_up, uchar a_down, uchar  b_up, uchar b_down)
{
    int A = a_down - b_down;
    int B = a_up - b_up;
    return (A-B)*(A+B);
}

template <typename T> static inline void incWithWeight(int* estimation, int weight, T p);

template <> inline void incWithWeight(int* estimation, int weight, uchar p)
{
    estimation[0] += weight * p;
}

template <> inline void incWithWeight(int* estimation, int weight, Vec2b p)
{
    estimation[0] += weight * p[0];
    estimation[1] += weight * p[1];
}

template <> inline void incWithWeight(int* estimation, int weight, Vec3b p)
{
    estimation[0] += weight * p[0];
    estimation[1] += weight * p[1];
    estimation[2] += weight * p[2];
}

template <> inline void incWithWeight(int* estimation, int weight, int p)
{
    estimation[0] += weight * p;
}

template <> inline void incWithWeight(int* estimation, int weight, Vec2i p)
{
    estimation[0] += weight * p[0];
    estimation[1] += weight * p[1];
}

template <> inline void incWithWeight(int* estimation, int weight, Vec3i p)
{
    estimation[0] += weight * p[0];
    estimation[1] += weight * p[1];
    estimation[2] += weight * p[2];
}

template <typename T> static inline T saturateCastFromArray(int* estimation);

template <> inline uchar saturateCastFromArray(int* estimation)
{
    return saturate_cast<uchar>(estimation[0]);
}

template <> inline Vec2b saturateCastFromArray(int* estimation)
{
    Vec2b res;
    res[0] = saturate_cast<uchar>(estimation[0]);
    res[1] = saturate_cast<uchar>(estimation[1]);
    return res;
}

template <> inline Vec3b saturateCastFromArray(int* estimation)
{
    Vec3b res;
    res[0] = saturate_cast<uchar>(estimation[0]);
    res[1] = saturate_cast<uchar>(estimation[1]);
    res[2] = saturate_cast<uchar>(estimation[2]);
    return res;
}

template <> inline int saturateCastFromArray(int* estimation)
{
    return estimation[0];
}

template <> inline Vec2i saturateCastFromArray(int* estimation)
{
    estimation[1] = 0;
    return Vec2i(estimation);
}

template <> inline Vec3i saturateCastFromArray(int* estimation)
{
    return Vec3i(estimation);
}

#endif
