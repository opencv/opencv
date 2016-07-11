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

#ifndef __OPENCV_BM3D_DENOISING_INVOKER_COMMONS_HPP__
#define __OPENCV_BM3D_DENOISING_INVOKER_COMMONS_HPP__

//#define DEBUG_PRINT

using namespace cv;

// std::isnan is a part of C++11 and it is not supported in MSVS2010/2012
#if defined _MSC_VER && _MSC_VER < 1800 /* MSVC 2013 */
#include <float.h>
namespace std {
    template <typename T> bool isnan(T value) { return _isnan(value) != 0; }
}
#endif

// Returns largest power of 2 smaller than the input value
inline int getLargestPowerOf2SmallerThan(unsigned x)
{
    x = x | (x >> 1);
    x = x | (x >> 2);
    x = x | (x >> 4);
    x = x | (x >> 8);
    x = x | (x >> 16);
    return x - (x >> 1);
}

class DistAbs
{
    template <typename T> struct calcDist_
    {
        static inline int f(const T a, const T b)
        {
            return std::abs(a - b);
        }
    };

    template <typename ET> struct calcDist_<Vec<ET, 2> >
    {
        static inline int f(const Vec<ET, 2> a, const Vec<ET, 2> b)
        {
            return std::abs((int)(a[0] - b[0])) + std::abs((int)(a[1] - b[1]));
        }
    };

    template <typename ET> struct calcDist_<Vec<ET, 3> >
    {
        static inline int f(const Vec<ET, 3> a, const Vec<ET, 3> b)
        {
            return
                std::abs((int)(a[0] - b[0])) +
                std::abs((int)(a[1] - b[1])) +
                std::abs((int)(a[2] - b[2]));
        }
    };

    template <typename ET> struct calcDist_<Vec<ET, 4> >
    {
        static inline int f(const Vec<ET, 4> a, const Vec<ET, 4> b)
        {
            return
                std::abs((int)(a[0] - b[0])) +
                std::abs((int)(a[1] - b[1])) +
                std::abs((int)(a[2] - b[2])) +
                std::abs((int)(a[3] - b[3]));
        }
    };

public:
    template <typename T> static inline int calcDist(const T a, const T b)
    {
        return calcDist_<T>::f(a, b);
    }

    template <typename T>
    static inline int calcDist(const Mat& m, int i1, int j1, int i2, int j2)
    {
        const T a = m.at<T>(i1, j1);
        const T b = m.at<T>(i2, j2);
        return calcDist<T>(a, b);
    }
};

class DistSquared
{
    template <typename T> struct calcDist_
    {
        static inline int f(const T a, const T b)
        {
            return (a - b) * (a - b);
        }
    };

    template <typename ET> struct calcDist_<Vec<ET, 2> >
    {
        static inline int f(const Vec<ET, 2> a, const Vec<ET, 2> b)
        {
            return (int)(a[0] - b[0])*(int)(a[0] - b[0]) + (int)(a[1] - b[1])*(int)(a[1] - b[1]);
        }
    };

    template <typename ET> struct calcDist_<Vec<ET, 3> >
    {
        static inline int f(const Vec<ET, 3> a, const Vec<ET, 3> b)
        {
            return
                (int)(a[0] - b[0])*(int)(a[0] - b[0]) +
                (int)(a[1] - b[1])*(int)(a[1] - b[1]) +
                (int)(a[2] - b[2])*(int)(a[2] - b[2]);
        }
    };

    template <typename ET> struct calcDist_<Vec<ET, 4> >
    {
        static inline int f(const Vec<ET, 4> a, const Vec<ET, 4> b)
        {
            return
                (int)(a[0] - b[0])*(int)(a[0] - b[0]) +
                (int)(a[1] - b[1])*(int)(a[1] - b[1]) +
                (int)(a[2] - b[2])*(int)(a[2] - b[2]) +
                (int)(a[3] - b[3])*(int)(a[3] - b[3]);
        }
    };

public:
    template <typename T> static inline int calcDist(const T a, const T b)
    {
        return calcDist_<T>::f(a, b);
    }

    template <typename T>
    static inline int calcDist(const Mat& m, int i1, int j1, int i2, int j2)
    {
        const T a = m.at<T>(i1, j1);
        const T b = m.at<T>(i2, j2);
        return calcDist<T>(a, b);
    }
};

template <typename T, typename IT, typename WT> struct incWithWeight_
{
    static inline void f(IT* estimation, IT* weights_sum, WT weight, T p)
    {
        estimation[0] += (IT)weight * p;
        weights_sum[0] += (IT)weight;
    }
};

template <typename ET, typename IT, typename WT> struct incWithWeight_<Vec<ET, 2>, IT, WT>
{
    static inline void f(IT* estimation, IT* weights_sum, WT weight, Vec<ET, 2> p)
    {
        estimation[0] += (IT)weight * p[0];
        estimation[1] += (IT)weight * p[1];
        weights_sum[0] += (IT)weight;
    }
};

template <typename ET, typename IT, typename WT> struct incWithWeight_<Vec<ET, 3>, IT, WT>
{
    static inline void f(IT* estimation, IT* weights_sum, WT weight, Vec<ET, 3> p)
    {
        estimation[0] += (IT)weight * p[0];
        estimation[1] += (IT)weight * p[1];
        estimation[2] += (IT)weight * p[2];
        weights_sum[0] += (IT)weight;
    }
};

template <typename ET, typename IT, typename WT> struct incWithWeight_<Vec<ET, 4>, IT, WT>
{
    static inline void f(IT* estimation, IT* weights_sum, WT weight, Vec<ET, 4> p)
    {
        estimation[0] += (IT)weight * p[0];
        estimation[1] += (IT)weight * p[1];
        estimation[2] += (IT)weight * p[2];
        estimation[3] += (IT)weight * p[3];
        weights_sum[0] += (IT)weight;
    }
};

template <typename ET, typename IT, typename EW> struct incWithWeight_<Vec<ET, 2>, IT, Vec<EW, 2> >
{
    static inline void f(IT* estimation, IT* weights_sum, Vec<EW, 2> weight, Vec<ET, 2> p)
    {
        estimation[0] += (IT)weight[0] * p[0];
        estimation[1] += (IT)weight[1] * p[1];
        weights_sum[0] += (IT)weight[0];
        weights_sum[1] += (IT)weight[1];
    }
};

template <typename ET, typename IT, typename EW> struct incWithWeight_<Vec<ET, 3>, IT, Vec<EW, 3> >
{
    static inline void f(IT* estimation, IT* weights_sum, Vec<EW, 3> weight, Vec<ET, 3> p)
    {
        estimation[0] += (IT)weight[0] * p[0];
        estimation[1] += (IT)weight[1] * p[1];
        estimation[2] += (IT)weight[2] * p[2];
        weights_sum[0] += (IT)weight[0];
        weights_sum[1] += (IT)weight[1];
        weights_sum[2] += (IT)weight[2];
    }
};

template <typename ET, typename IT, typename EW> struct incWithWeight_<Vec<ET, 4>, IT, Vec<EW, 4> >
{
    static inline void f(IT* estimation, IT* weights_sum, Vec<EW, 4> weight, Vec<ET, 4> p)
    {
        estimation[0] += (IT)weight[0] * p[0];
        estimation[1] += (IT)weight[1] * p[1];
        estimation[2] += (IT)weight[2] * p[2];
        estimation[3] += (IT)weight[3] * p[3];
        weights_sum[0] += (IT)weight[0];
        weights_sum[1] += (IT)weight[1];
        weights_sum[2] += (IT)weight[2];
        weights_sum[3] += (IT)weight[3];
    }
};

template <typename T, typename IT, typename WT>
static inline void incWithWeight(IT* estimation, IT* weights_sum, WT weight, T p)
{
    return incWithWeight_<T, IT, WT>::f(estimation, weights_sum, weight, p);
}

template <typename IT, typename UIT, int nc, int nw> struct divByWeightsSum_
{
    static inline void f(IT* estimation, IT* weights_sum);
};

template <typename IT, typename UIT> struct divByWeightsSum_<IT, UIT, 1, 1>
{
    static inline void f(IT* estimation, IT* weights_sum)
    {
        estimation[0] = (static_cast<UIT>(estimation[0]) + weights_sum[0] / 2) / weights_sum[0];
    }
};

template <typename IT, typename UIT, int n> struct divByWeightsSum_<IT, UIT, n, 1>
{
    static inline void f(IT* estimation, IT* weights_sum)
    {
        for (size_t i = 0; i < n; i++)
            estimation[i] = (static_cast<UIT>(estimation[i]) + weights_sum[0] / 2) / weights_sum[0];
    }
};

template <typename IT, typename UIT, int n> struct divByWeightsSum_<IT, UIT, n, n>
{
    static inline void f(IT* estimation, IT* weights_sum)
    {
        for (size_t i = 0; i < n; i++)
            estimation[i] = (static_cast<UIT>(estimation[i]) + weights_sum[i] / 2) / weights_sum[i];
    }
};

template <typename IT, typename UIT, int nc, int nw>
static inline void divByWeightsSum(IT* estimation, IT* weights_sum)
{
    return divByWeightsSum_<IT, UIT, nc, nw>::f(estimation, weights_sum);
}

template <typename T, typename IT> struct saturateCastFromArray_
{
    static inline T f(IT* estimation)
    {
        return saturate_cast<T>(estimation[0]);
    }
};

template <typename ET, typename IT> struct saturateCastFromArray_<Vec<ET, 2>, IT>
{
    static inline Vec<ET, 2> f(IT* estimation)
    {
        Vec<ET, 2> res;
        res[0] = saturate_cast<ET>(estimation[0]);
        res[1] = saturate_cast<ET>(estimation[1]);
        return res;
    }
};

template <typename ET, typename IT> struct saturateCastFromArray_<Vec<ET, 3>, IT>
{
    static inline Vec<ET, 3> f(IT* estimation)
    {
        Vec<ET, 3> res;
        res[0] = saturate_cast<ET>(estimation[0]);
        res[1] = saturate_cast<ET>(estimation[1]);
        res[2] = saturate_cast<ET>(estimation[2]);
        return res;
    }
};

template <typename ET, typename IT> struct saturateCastFromArray_<Vec<ET, 4>, IT>
{
    static inline Vec<ET, 4> f(IT* estimation)
    {
        Vec<ET, 4> res;
        res[0] = saturate_cast<ET>(estimation[0]);
        res[1] = saturate_cast<ET>(estimation[1]);
        res[2] = saturate_cast<ET>(estimation[2]);
        res[3] = saturate_cast<ET>(estimation[3]);
        return res;
    }
};

#ifdef DEBUG_PRINT
#include <iostream>
#endif

void ComputeThresholdMap1D(
    short *outThrMap1D,
    const float *thrMap1D,
    float *thrMap2D,
    const float &hardThr1D,
    const float *coeff,
    const int &templateWindowSizeSq)
{
    short *thrMapPtr1D = outThrMap1D;
    for (int ii = 0; ii < 4; ++ii)
    {
#ifdef DEBUG_PRINT
        std::cout << "group size: " << (1 << ii) << std::endl;
#endif
        for (int jj = 0; jj < templateWindowSizeSq; ++jj)
        {
#ifdef DEBUG_PRINT
            if (jj % (int)std::sqrt(templateWindowSizeSq) == 0)
                std::cout << std::endl;
            std::cout << "\t";
#endif
            for (int ii1 = 0; ii1 < (1 << ii); ++ii1)
            {
                int indexIn1D = (1 << ii) - 1 + ii1;
                int indexIn2D = jj;
                int thr = static_cast<int>(thrMap1D[indexIn1D] * thrMap2D[indexIn2D] * hardThr1D * coeff[ii]);
                if (thr > std::numeric_limits<short>::max())
                    thr = std::numeric_limits<short>::max();

                if (jj == 0 && ii1 == 0)
                    thr = 0;

                *thrMapPtr1D++ = (short)thr;

#ifdef DEBUG_PRINT
                std::cout << thr << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
#else
           }
        }
#endif
    }
}

#endif
