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

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/core/types_c.h>  // CV_MAX_DIM
#include <iostream>
#include <ostream>
#include <sstream>

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN

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

static inline MatShape shape(const int* dims, const int n)
{
    MatShape shape;
    shape.assign(dims, dims + n);
    return shape;
}

static inline MatShape shape(const Mat& mat)
{
    return shape(mat.size.p, mat.dims);
}

static inline MatShape shape(const MatSize& sz)
{
    return shape(sz.p, sz.dims());
}

static inline MatShape shape(const UMat& mat)
{
    return shape(mat.size.p, mat.dims);
}

#if 0  // issues with MatExpr wrapped into InputArray
static inline
MatShape shape(InputArray input)
{
    int sz[CV_MAX_DIM];
    int ndims = input.sizend(sz);
    return shape(sz, ndims);
}
#endif

namespace {inline bool is_neg(int i) { return i < 0; }}

static inline MatShape shape(int a0, int a1=-1, int a2=-1, int a3=-1)
{
    int shape_[] = {a0, a1, a2, a3};
    int dims = 1 + (a1 >= 0) + (a1 >= 0 && a2 >= 0) + (a1 >= 0 && a2 >= 0 && a3 >= 0);
    return shape(shape_, dims);
}

static inline int total(const MatShape& shape, int start = -1, int end = -1)
{
    //if (shape.empty())
    //    return 0;

    int dims = (int)shape.size();

    if (start == -1) start = 0;
    if (end == -1) end = dims;

    CV_CheckLE(0, start, "");
    CV_CheckLE(start, end, "");
    CV_CheckLE(end, dims, "");

    int elems = 1;
    for (int i = start; i < end; i++)
    {
        elems *= shape[i];
    }
    return elems;
}

// TODO: rename to countDimsElements()
static inline int total(const Mat& mat, int start = -1, int end = -1)
{
    if (mat.empty())
        return 0;

    int dims = mat.dims;

    if (start == -1) start = 0;
    if (end == -1) end = dims;

    CV_CheckLE(0, start, "");
    CV_CheckLE(start, end, "");
    CV_CheckLE(end, dims, "");

    int elems = 1;
    for (int i = start; i < end; i++)
    {
        elems *= mat.size[i];
    }
    return elems;
}

static inline MatShape concat(const MatShape& a, const MatShape& b)
{
    MatShape c = a;
    size_t a_size = a.size(), b_size = b.size(), c_size = a_size + b_size;
    c.resize(c_size);
    for (size_t i = 0; i < b_size; i++) {
        c[i+a_size] = b[i];
    }
    return c;
}

static inline std::ostream& operator << (std::ostream& strm, const MatShape& shape)
{
    strm << '[';
    if (shape.empty()) {
        strm << "<empty>";
    } else {
        size_t n = shape.size();
        if (n == 0) {
            strm << "<scalar>";
        } else {
            for(size_t i = 0; i < n; ++i)
                strm << (i > 0 ? " x " : "") << shape[i];
        }
    }
    strm << "]";
    return strm;
}

static inline std::string toString(const MatShape& shape, const String& name = "")
{
    std::ostringstream ss;
    if (!name.empty())
        ss << name << ' ';
    ss << shape;
    return ss.str();
}

template<typename _Tp>
static inline std::string toString(const std::vector<_Tp>& shape, const String& name = "")
{
    std::ostringstream ss;
    if (!name.empty())
        ss << name << ' ';
    ss << '[';
    for(size_t i = 0, n = shape.size(); i < n; ++i)
        ss << ' ' << shape[i];
    ss << " ]";
    return ss.str();
}

template<typename _Tp>
static inline void print(const std::vector<_Tp>& shape, const String& name = "")
{
    std::cout << toString(shape, name) << std::endl;
}
template<typename _Tp>
static inline std::ostream& operator<<(std::ostream &out, const std::vector<_Tp>& shape)
{
    out << toString(shape);
    return out;
}

/// @brief Converts axis from `[-dims; dims)` (similar to Python's slice notation) to `[0; dims)` range.
static inline
int normalize_axis(int axis, int dims)
{
    CV_Assert(dims >= 0);
    CV_Check(axis, axis >= -dims && axis <= dims, "");
    axis = (unsigned)axis < (unsigned)dims ? axis : axis < 0 ? axis + dims : axis - dims;
    return axis;
}

static inline
int normalize_axis(int axis, const MatShape& shape)
{
    return normalize_axis(axis, (int)shape.size());
}

static inline
Range normalize_axis_range(const Range& r, int axisSize)
{
    if (r == Range::all() || r == Range(0, INT_MAX))
        return Range(0, axisSize);
    CV_CheckGE(r.start, 0, "");
    Range clamped(r.start,
                  r.end > 0 ? std::min(r.end, axisSize) : axisSize + r.end + 1);
    CV_DbgCheckGE(clamped.start, 0, "");
    CV_CheckLT(clamped.start, clamped.end, "");
    CV_CheckLE(clamped.end, axisSize, "");
    return clamped;
}

static inline
bool isAllOnes(const MatShape &inputShape, int startPos, int endPos)
{
    CV_Assert(!inputShape.empty());

    CV_CheckGE((int) inputShape.size(), startPos, "");
    CV_CheckGE(startPos, 0, "");
    CV_CheckLE(startPos, endPos, "");
    CV_CheckLE((size_t)endPos, inputShape.size(), "");

    for (size_t i = startPos; i < endPos; i++)
    {
        if (inputShape[i] != 1)
            return false;
    }
    return true;
}

CV__DNN_INLINE_NS_END
}
}
#endif
