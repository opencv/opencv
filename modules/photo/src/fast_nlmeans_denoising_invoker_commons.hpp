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

// std::isnan is a part of C++11 and it is not supported in MSVS2010/2012
#if defined _MSC_VER && _MSC_VER < 1800 /* MSVC 2013 */
#include <float.h>
namespace std {
template <typename T> bool isnan(T value) { return _isnan(value) != 0; }
}
#endif

template <typename T> struct pixelInfo_
{
    static const int channels = 1;
    typedef T sampleType;
};

template <typename ET, int n> struct pixelInfo_<Vec<ET, n> >
{
    static const int channels = n;
    typedef ET sampleType;
};

template <typename T> struct pixelInfo: public pixelInfo_<T>
{
    typedef typename pixelInfo_<T>::sampleType sampleType;

    static inline sampleType sampleMax()
    {
        return std::numeric_limits<sampleType>::max();
    }

    static inline sampleType sampleMin()
    {
        return std::numeric_limits<sampleType>::min();
    }

    static inline size_t sampleBytes()
    {
        return sizeof(sampleType);
    }

    static inline size_t sampleBits()
    {
        return 8*sampleBytes();
    }
};

class DistAbs
{
    template <typename T> struct calcDist_
    {
        static inline int f(const T a, const T b)
        {
            return std::abs((int)(a-b));
        }
    };

    template <typename ET> struct calcDist_<Vec<ET, 2> >
    {
        static inline int f(const Vec<ET, 2> a, const Vec<ET, 2> b)
        {
            return std::abs((int)(a[0]-b[0])) + std::abs((int)(a[1]-b[1]));
        }
    };

    template <typename ET> struct calcDist_<Vec<ET, 3> >
    {
        static inline int f(const Vec<ET, 3> a, const Vec<ET, 3> b)
        {
            return
                std::abs((int)(a[0]-b[0])) +
                std::abs((int)(a[1]-b[1])) +
                std::abs((int)(a[2]-b[2]));
        }
    };

    template <typename ET> struct calcDist_<Vec<ET, 4> >
    {
        static inline int f(const Vec<ET, 4> a, const Vec<ET, 4> b)
        {
            return
                std::abs((int)(a[0]-b[0])) +
                std::abs((int)(a[1]-b[1])) +
                std::abs((int)(a[2]-b[2])) +
                std::abs((int)(a[3]-b[3]));
        }
    };

    template <typename T, typename WT> struct calcWeight_
    {
        static inline WT f(double dist, const float *h, WT fixed_point_mult)
        {
            double w = std::exp(-dist*dist / (h[0]*h[0] * pixelInfo<T>::channels));
            if (std::isnan(w)) w = 1.0; // Handle h = 0.0

            static const double WEIGHT_THRESHOLD = 0.001;
            WT weight = (WT)cvRound(fixed_point_mult * w);
            if (weight < WEIGHT_THRESHOLD * fixed_point_mult) weight = 0;

            return weight;
        }
    };

    template <typename T, typename ET, int n> struct calcWeight_<T, Vec<ET, n> >
    {
        static inline Vec<ET, n> f(double dist, const float *h, ET fixed_point_mult)
        {
            Vec<ET, n> res;
            for (int i=0; i<n; i++)
                res[i] = calcWeight<T, ET>(dist, &h[i], fixed_point_mult);
            return res;
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
        return calcDist<T>(a,b);
    }

    template <typename T>
    static inline int calcUpDownDist(T a_up, T a_down, T b_up, T b_down)
    {
        return calcDist<T>(a_down, b_down) - calcDist<T>(a_up, b_up);
    }

    template <typename T, typename WT>
    static inline WT calcWeight(double dist, const float *h,
                                typename pixelInfo<WT>::sampleType fixed_point_mult)
    {
        return calcWeight_<T, WT>::f(dist, h, fixed_point_mult);
    }

    template <typename T>
    static inline int maxDist()
    {
        return (int)pixelInfo<T>::sampleMax() * pixelInfo<T>::channels;
    }
};

class DistSquared
{
    template <typename T> struct calcDist_
    {
        static inline int f(const T a, const T b)
        {
            return (int)(a-b) * (int)(a-b);
        }
    };

    template <typename ET> struct calcDist_<Vec<ET, 2> >
    {
        static inline int f(const Vec<ET, 2> a, const Vec<ET, 2> b)
        {
            return (int)(a[0]-b[0])*(int)(a[0]-b[0]) + (int)(a[1]-b[1])*(int)(a[1]-b[1]);
        }
    };

    template <typename ET> struct calcDist_<Vec<ET, 3> >
    {
        static inline int f(const Vec<ET, 3> a, const Vec<ET, 3> b)
        {
            return
                (int)(a[0]-b[0])*(int)(a[0]-b[0]) +
                (int)(a[1]-b[1])*(int)(a[1]-b[1]) +
                (int)(a[2]-b[2])*(int)(a[2]-b[2]);
        }
    };

    template <typename ET> struct calcDist_<Vec<ET, 4> >
    {
        static inline int f(const Vec<ET, 4> a, const Vec<ET, 4> b)
        {
            return
                (int)(a[0]-b[0])*(int)(a[0]-b[0]) +
                (int)(a[1]-b[1])*(int)(a[1]-b[1]) +
                (int)(a[2]-b[2])*(int)(a[2]-b[2]) +
                (int)(a[3]-b[3])*(int)(a[3]-b[3]);
        }
    };

    template <typename T> struct calcUpDownDist_
    {
        static inline int f(T a_up, T a_down, T b_up, T b_down)
        {
            int A = a_down - b_down;
            int B = a_up - b_up;
            return (A-B)*(A+B);
        }
    };

    template <typename ET, int n> struct calcUpDownDist_<Vec<ET, n> >
    {
    private:
        typedef Vec<ET, n> T;
    public:
        static inline int f(T a_up, T a_down, T b_up, T b_down)
        {
            return calcDist<T>(a_down, b_down) - calcDist<T>(a_up, b_up);
        }
    };

    template <typename T, typename WT> struct calcWeight_
    {
        static inline WT f(double dist, const float *h, WT fixed_point_mult)
        {
            double w = std::exp(-dist / (h[0]*h[0] * pixelInfo<T>::channels));
            if (std::isnan(w)) w = 1.0; // Handle h = 0.0

            static const double WEIGHT_THRESHOLD = 0.001;
            WT weight = (WT)cvRound(fixed_point_mult * w);
            if (weight < WEIGHT_THRESHOLD * fixed_point_mult) weight = 0;

            return weight;
        }
    };

    template <typename T, typename ET, int n> struct calcWeight_<T, Vec<ET, n> >
    {
        static inline Vec<ET, n> f(double dist, const float *h, ET fixed_point_mult)
        {
            Vec<ET, n> res;
            for (int i=0; i<n; i++)
                res[i] = calcWeight<T, ET>(dist, &h[i], fixed_point_mult);
            return res;
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
        return calcDist<T>(a,b);
    }

    template <typename T>
    static inline int calcUpDownDist(T a_up, T a_down, T b_up, T b_down)
    {
        return calcUpDownDist_<T>::f(a_up, a_down, b_up, b_down);
    }

    template <typename T, typename WT>
    static inline WT calcWeight(double dist, const float *h,
                                typename pixelInfo<WT>::sampleType fixed_point_mult)
    {
        return calcWeight_<T, WT>::f(dist, h, fixed_point_mult);
    }

    template <typename T>
    static inline int maxDist()
    {
        return (int)pixelInfo<T>::sampleMax() * (int)pixelInfo<T>::sampleMax() *
            pixelInfo<T>::channels;
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
    incWithWeight_<T, IT, WT>::f(estimation, weights_sum, weight, p);
}

template <typename IT, typename UIT, int nc, int nw> struct divByWeightsSum_
{
    static inline void f(IT* estimation, IT* weights_sum);
};

template <typename IT, typename UIT> struct divByWeightsSum_<IT, UIT, 1, 1>
{
    static inline void f(IT* estimation, IT* weights_sum)
    {
        estimation[0] = (static_cast<UIT>(estimation[0]) + weights_sum[0]/2) / weights_sum[0];
    }
};

template <typename IT, typename UIT, int n> struct divByWeightsSum_<IT, UIT, n, 1>
{
    static inline void f(IT* estimation, IT* weights_sum)
    {
        for (size_t i = 0; i < n; i++)
            estimation[i] = (static_cast<UIT>(estimation[i]) + weights_sum[0]/2) / weights_sum[0];
    }
};

template <typename IT, typename UIT, int n> struct divByWeightsSum_<IT, UIT, n, n>
{
    static inline void f(IT* estimation, IT* weights_sum)
    {
        for (size_t i = 0; i < n; i++)
            estimation[i] = (static_cast<UIT>(estimation[i]) + weights_sum[i]/2) / weights_sum[i];
    }
};

template <typename IT, typename UIT, int nc, int nw>
static inline void divByWeightsSum(IT* estimation, IT* weights_sum)
{
    divByWeightsSum_<IT, UIT, nc, nw>::f(estimation, weights_sum);
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

template <typename T, typename IT> static inline T saturateCastFromArray(IT* estimation)
{
    return saturateCastFromArray_<T, IT>::f(estimation);
}

#endif
