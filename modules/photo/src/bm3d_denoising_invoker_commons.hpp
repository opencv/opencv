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

template <typename T, typename DT, typename CT>
class BlockMatch
{
public:
    // Data accessor
    T* data()
    {
        return data_;
    }

    // Const version of data accessor
    const T* data() const
    {
        return data_;
    }

    // Allocate memory for data
    void init(const int &blockSizeSq)
    {
        data_ = new T[blockSizeSq];
    }

    // Release data memory
    void release()
    {
        delete[] data_;
    }

    // Overloaded operator for convenient assignment
    void operator()(const DT &_dist, const CT &_coord_x, const CT &_coord_y)
    {
        dist = _dist;
        coord_x = _coord_x;
        coord_y = _coord_y;
    }

    // Overloaded array subscript operator
    T& operator[](const std::size_t &idx)
    {
        return data_[idx];
    };

    // Overloaded const array subscript operator
    const T& operator[](const std::size_t &idx) const
    {
        return data_[idx];
    };

    // Overloaded comparison operator for sorting
    bool operator<(const BlockMatch& right) const
    {
        return dist < right.dist;
    }

    // Block matching distance
    DT dist;

    // Relative coordinates to the current search window
    CT coord_x;
    CT coord_y;

private:
    // Pointer to the pixel values of the block
    T *data_;
};

class DistAbs
{
    template <typename T>
    struct calcDist_
    {
        static inline int f(const T &a, const T &b)
        {
            return std::abs(a - b);
        }
    };

    template <typename ET>
    struct calcDist_<Vec<ET, 2> >
    {
        static inline int f(const Vec<ET, 2> a, const Vec<ET, 2> b)
        {
            return std::abs((int)(a[0] - b[0])) + std::abs((int)(a[1] - b[1]));
        }
    };

    template <typename ET>
    struct calcDist_<Vec<ET, 3> >
    {
        static inline int f(const Vec<ET, 3> a, const Vec<ET, 3> b)
        {
            return
                std::abs((int)(a[0] - b[0])) +
                std::abs((int)(a[1] - b[1])) +
                std::abs((int)(a[2] - b[2]));
        }
    };

    template <typename ET>
    struct calcDist_<Vec<ET, 4> >
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
    template <typename T>
    static inline int calcDist(const T &a, const T &b)
    {
        return calcDist_<T>::f(a, b);
    }

    template <typename T>
    static inline T calcBlockMatchingThreshold(const T &blockMatchThrL2, const T &blockSizeSq)
    {
        return (T)(std::sqrt((double)blockMatchThrL2) * blockSizeSq);
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
    template <typename T>
    struct calcDist_
    {
        static inline int f(const T &a, const T &b)
        {
            return (a - b) * (a - b);
        }
    };

    template <typename ET>
    struct calcDist_<Vec<ET, 2> >
    {
        static inline int f(const Vec<ET, 2> a, const Vec<ET, 2> b)
        {
            return (int)(a[0] - b[0])*(int)(a[0] - b[0]) + (int)(a[1] - b[1])*(int)(a[1] - b[1]);
        }
    };

    template <typename ET>
    struct calcDist_<Vec<ET, 3> >
    {
        static inline int f(const Vec<ET, 3> a, const Vec<ET, 3> b)
        {
            return
                (int)(a[0] - b[0])*(int)(a[0] - b[0]) +
                (int)(a[1] - b[1])*(int)(a[1] - b[1]) +
                (int)(a[2] - b[2])*(int)(a[2] - b[2]);
        }
    };

    template <typename ET>
    struct calcDist_<Vec<ET, 4> >
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
    template <typename T>
    static inline int calcDist(const T &a, const T &b)
    {
        return calcDist_<T>::f(a, b);
    }

    template <typename T>
    static inline T calcBlockMatchingThreshold(const T &blockMatchThrL2, const T &blockSizeSq)
    {
        return blockMatchThrL2 * blockSizeSq;
    }

    template <typename T>
    static inline int calcDist(const Mat& m, int i1, int j1, int i2, int j2)
    {
        const T a = m.at<T>(i1, j1);
        const T b = m.at<T>(i2, j2);
        return calcDist<T>(a, b);
    }
};

#endif
