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

#ifndef OPENCV_DNN_DNN_SHAPE_UTILS_HPP
#define OPENCV_DNN_DNN_SHAPE_UTILS_HPP

#include <opencv2/core.hpp>
#include <opencv2/core/types_c.h>
#include <ostream>

namespace cv {
namespace dnn {
CV__DNN_EXPERIMENTAL_NS_BEGIN

//Useful shortcut
inline std::ostream &operator<< (std::ostream &s, cv::Range &r)
{
    return s << "[" << r.start << ", " << r.end << ")";
}

//Slicing

struct _Range : public cv::Range
{
    _Range(const Range &r) : cv::Range(r) {}
    _Range(int start_, int size_ = 1) : cv::Range(start_, start_ + size_) {}
};

static inline Mat slice(const Mat &m, const _Range &r0)
{
    Range ranges[CV_MAX_DIM];
    for (int i = 1; i < m.dims; i++)
        ranges[i] = Range::all();
    ranges[0] = r0;
    return m(&ranges[0]);
}

static inline Mat slice(const Mat &m, const _Range &r0, const _Range &r1)
{
    CV_Assert(m.dims >= 2);
    Range ranges[CV_MAX_DIM];
    for (int i = 2; i < m.dims; i++)
        ranges[i] = Range::all();
    ranges[0] = r0;
    ranges[1] = r1;
    return m(&ranges[0]);
}

static inline Mat slice(const Mat &m, const _Range &r0, const _Range &r1, const _Range &r2)
{
    CV_Assert(m.dims >= 3);
    Range ranges[CV_MAX_DIM];
    for (int i = 3; i < m.dims; i++)
        ranges[i] = Range::all();
    ranges[0] = r0;
    ranges[1] = r1;
    ranges[2] = r2;
    return m(&ranges[0]);
}

static inline Mat slice(const Mat &m, const _Range &r0, const _Range &r1, const _Range &r2, const _Range &r3)
{
    CV_Assert(m.dims >= 4);
    Range ranges[CV_MAX_DIM];
    for (int i = 4; i < m.dims; i++)
        ranges[i] = Range::all();
    ranges[0] = r0;
    ranges[1] = r1;
    ranges[2] = r2;
    ranges[3] = r3;
    return m(&ranges[0]);
}

static inline Mat getPlane(const Mat &m, int n, int cn)
{
    CV_Assert(m.dims > 2);
    int sz[CV_MAX_DIM];
    for(int i = 2; i < m.dims; i++)
    {
        sz[i-2] = m.size.p[i];
    }
    return Mat(m.dims - 2, sz, m.type(), (void*)m.ptr<float>(n, cn));
}

static inline MatShape shape(const int* dims, const int n = 4)
{
    MatShape shape;
    shape.assign(dims, dims + n);
    return shape;
}

static inline MatShape shape(const Mat& mat)
{
    return shape(mat.size.p, mat.dims);
}

namespace {inline bool is_neg(int i) { return i < 0; }}

static inline MatShape shape(int a0, int a1=-1, int a2=-1, int a3=-1)
{
    int dims[] = {a0, a1, a2, a3};
    MatShape s = shape(dims);
    s.erase(std::remove_if(s.begin(), s.end(), is_neg), s.end());
    return s;
}

static inline int total(const MatShape& shape, int start = -1, int end = -1)
{
    if (start == -1) start = 0;
    if (end == -1) end = (int)shape.size();

    if (shape.empty())
        return 0;

    int elems = 1;
    CV_Assert(start < (int)shape.size() && end <= (int)shape.size() &&
              start <= end);
    for(int i = start; i < end; i++)
    {
        elems *= shape[i];
    }
    return elems;
}

static inline MatShape concat(const MatShape& a, const MatShape& b)
{
    MatShape c = a;
    c.insert(c.end(), b.begin(), b.end());

    return c;
}

inline void print(const MatShape& shape, const String& name = "")
{
    printf("%s: [", name.c_str());
    size_t i, n = shape.size();
    for( i = 0; i < n; i++ )
        printf(" %d", shape[i]);
    printf(" ]\n");
}

inline int clamp(int ax, int dims)
{
    return ax < 0 ? ax + dims : ax;
}

inline int clamp(int ax, const MatShape& shape)
{
    return clamp(ax, (int)shape.size());
}

inline Range clamp(const Range& r, int axisSize)
{
    Range clamped(std::max(r.start, 0),
                  r.end > 0 ? std::min(r.end, axisSize) : axisSize + r.end + 1);
    CV_Assert(clamped.start < clamped.end, clamped.end <= axisSize);
    return clamped;
}

CV__DNN_EXPERIMENTAL_NS_END
}
}
#endif
