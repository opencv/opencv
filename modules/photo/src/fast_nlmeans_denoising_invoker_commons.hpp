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

template <typename T, typename IT> struct calcDist_
{
    static inline IT f(const T a, const T b);
};

template <typename IT> struct calcDist_<uchar, IT>
{
    static inline IT f(uchar a, uchar b)
    {
        return (IT)(a-b) * (IT)(a-b);
    }
};

template <typename IT> struct calcDist_<Vec2b, IT>
{
    static inline IT f(const Vec2b a, const Vec2b b)
    {
        return (IT)(a[0]-b[0])*(IT)(a[0]-b[0]) + (IT)(a[1]-b[1])*(IT)(a[1]-b[1]);
    }
};

template <typename IT> struct calcDist_<Vec3b, IT>
{
    static inline IT f(const Vec3b a, const Vec3b b)
    {
        return
            (IT)(a[0]-b[0])*(IT)(a[0]-b[0]) +
            (IT)(a[1]-b[1])*(IT)(a[1]-b[1]) +
            (IT)(a[2]-b[2])*(IT)(a[2]-b[2]);
    }
};

template <typename T, typename IT> static inline IT calcDist(const T a, const T b)
{
    return calcDist_<T, IT>::f(a, b);
}

template <typename T, typename IT>
static inline IT calcDist(const Mat& m, int i1, int j1, int i2, int j2)
{
    const T a = m.at<T>(i1, j1);
    const T b = m.at<T>(i2, j2);
    return calcDist<T, IT>(a,b);
}

template <typename T, typename IT> struct calcUpDownDist_
{
    static inline IT f(T a_up, T a_down, T b_up, T b_down)
    {
        return calcDist<T, IT>(a_down, b_down) - calcDist<T, IT>(a_up, b_up);
    }
};

template <typename IT> struct calcUpDownDist_<uchar, IT>
{
    static inline IT f(uchar a_up, uchar a_down, uchar b_up, uchar b_down)
    {
        IT A = a_down - b_down;
        IT B = a_up - b_up;
        return (A-B)*(A+B);
    }
};

template <typename T, typename IT>
static inline IT calcUpDownDist(T a_up, T a_down, T b_up, T b_down)
{
    return calcUpDownDist_<T, IT>::f(a_up, a_down, b_up, b_down);
};

template <typename T, typename IT> struct incWithWeight_
{
    static inline void f(IT* estimation, IT weight, T p);
};

template <typename IT> struct incWithWeight_<uchar, IT>
{
    static inline void f(IT* estimation, IT weight, uchar p)
    {
        estimation[0] += weight * p;
    }
};

template <typename IT> struct incWithWeight_<Vec2b, IT>
{
    static inline void f(IT* estimation, IT weight, Vec2b p)
    {
        estimation[0] += weight * p[0];
        estimation[1] += weight * p[1];
    }
};

template <typename IT> struct incWithWeight_<Vec3b, IT>
{
    static inline void f(IT* estimation, IT weight, Vec3b p)
    {
        estimation[0] += weight * p[0];
        estimation[1] += weight * p[1];
        estimation[2] += weight * p[2];
    }
};

template <typename T, typename IT>
static inline void incWithWeight(IT* estimation, IT weight, T p)
{
    return incWithWeight_<T, IT>::f(estimation, weight, p);
}

template <typename T, typename IT> struct saturateCastFromArray_
{
    static inline T f(IT* estimation);
};

template <typename IT> struct saturateCastFromArray_<uchar, IT>
{
    static inline uchar f(IT* estimation)
    {
        return saturate_cast<uchar>(estimation[0]);
    }
};

template <typename IT> struct saturateCastFromArray_<Vec2b, IT>
{
    static inline Vec2b f(IT* estimation)
    {
        Vec2b res;
        res[0] = saturate_cast<uchar>(estimation[0]);
        res[1] = saturate_cast<uchar>(estimation[1]);
        return res;
    }
};

template <typename IT> struct saturateCastFromArray_<Vec3b, IT>
{
    static inline Vec3b f(IT* estimation)
    {
        Vec3b res;
        res[0] = saturate_cast<uchar>(estimation[0]);
        res[1] = saturate_cast<uchar>(estimation[1]);
        res[2] = saturate_cast<uchar>(estimation[2]);
        return res;
    }
};

template <typename T, typename IT> static inline T saturateCastFromArray(IT* estimation)
{
    return saturateCastFromArray_<T, IT>::f(estimation);
}

#endif
