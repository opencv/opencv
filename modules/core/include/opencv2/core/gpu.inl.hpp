/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

#ifndef __OPENCV_CORE_GPUINL_HPP__
#define __OPENCV_CORE_GPUINL_HPP__

#include "opencv2/core/gpu.hpp"

namespace cv { namespace gpu
{

//////////////////////////////// GpuMat ///////////////////////////////

inline
GpuMat::GpuMat()
    : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0)
{}

inline
GpuMat::GpuMat(int rows_, int cols_, int type_)
    : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0)
{
    if (rows_ > 0 && cols_ > 0)
        create(rows_, cols_, type_);
}

inline
GpuMat::GpuMat(Size size_, int type_)
    : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0)
{
    if (size_.height > 0 && size_.width > 0)
        create(size_.height, size_.width, type_);
}

inline
GpuMat::GpuMat(int rows_, int cols_, int type_, Scalar s_)
    : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0)
{
    if (rows_ > 0 && cols_ > 0)
    {
        create(rows_, cols_, type_);
        setTo(s_);
    }
}

inline
GpuMat::GpuMat(Size size_, int type_, Scalar s_)
    : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0)
{
    if (size_.height > 0 && size_.width > 0)
    {
        create(size_.height, size_.width, type_);
        setTo(s_);
    }
}

inline
GpuMat::~GpuMat()
{
    release();
}

inline
GpuMat GpuMat::clone() const
{
    GpuMat m;
    copyTo(m);
    return m;
}

inline
void GpuMat::assignTo(GpuMat& m, int _type) const
{
    if (_type < 0)
        m = *this;
    else
        convertTo(m, _type);
}

inline
size_t GpuMat::step1() const
{
    return step / elemSize1();
}

inline
bool GpuMat::empty() const
{
    return data == 0;
}

template<typename _Tp> inline
_Tp* GpuMat::ptr(int y)
{
    return (_Tp*)ptr(y);
}

template<typename _Tp> inline
const _Tp* GpuMat::ptr(int y) const
{
    return (const _Tp*)ptr(y);
}

inline
GpuMat GpuMat::row(int y) const
{
    return GpuMat(*this, Range(y, y+1), Range::all());
}

inline
GpuMat GpuMat::col(int x) const
{
    return GpuMat(*this, Range::all(), Range(x, x+1));
}

inline
GpuMat GpuMat::rowRange(int startrow, int endrow) const
{
    return GpuMat(*this, Range(startrow, endrow), Range::all());
}

inline
GpuMat GpuMat::rowRange(Range r) const
{
    return GpuMat(*this, r, Range::all());
}

inline
GpuMat GpuMat::colRange(int startcol, int endcol) const
{
    return GpuMat(*this, Range::all(), Range(startcol, endcol));
}

inline
GpuMat GpuMat::colRange(Range r) const
{
    return GpuMat(*this, Range::all(), r);
}

inline
void GpuMat::create(Size size_, int type_)
{
    create(size_.height, size_.width, type_);
}

inline
GpuMat GpuMat::operator()(Range _rowRange, Range _colRange) const
{
    return GpuMat(*this, _rowRange, _colRange);
}

inline
GpuMat GpuMat::operator()(Rect roi) const
{
    return GpuMat(*this, roi);
}

inline
bool GpuMat::isContinuous() const
{
    return (flags & Mat::CONTINUOUS_FLAG) != 0;
}

inline
size_t GpuMat::elemSize() const
{
    return CV_ELEM_SIZE(flags);
}

inline
size_t GpuMat::elemSize1() const
{
    return CV_ELEM_SIZE1(flags);
}

inline
int GpuMat::type() const
{
    return CV_MAT_TYPE(flags);
}

inline
int GpuMat::depth() const
{
    return CV_MAT_DEPTH(flags);
}

inline
int GpuMat::channels() const
{
    return CV_MAT_CN(flags);
}

inline
Size GpuMat::size() const
{
    return Size(cols, rows);
}

inline
uchar* GpuMat::ptr(int y)
{
    CV_DbgAssert((unsigned)y < (unsigned)rows);
    return data + step * y;
}

inline
const uchar* GpuMat::ptr(int y) const
{
    CV_DbgAssert((unsigned)y < (unsigned)rows);
    return data + step * y;
}

inline
GpuMat& GpuMat::operator = (Scalar s)
{
    setTo(s);
    return *this;
}

template <class T> inline
GpuMat::operator PtrStepSz<T>() const
{
    return PtrStepSz<T>(rows, cols, (T*)data, step);
}

template <class T> inline
GpuMat::operator PtrStep<T>() const
{
    return PtrStep<T>((T*)data, step);
}

static inline
void swap(GpuMat& a, GpuMat& b)
{
    a.swap(b);
}

static inline
GpuMat createContinuous(int rows, int cols, int type)
{
    GpuMat m;
    createContinuous(rows, cols, type, m);
    return m;
}

static inline
void createContinuous(Size size, int type, GpuMat& m)
{
    createContinuous(size.height, size.width, type, m);
}

static inline
GpuMat createContinuous(Size size, int type)
{
    GpuMat m;
    createContinuous(size, type, m);
    return m;
}

static inline
void ensureSizeIsEnough(Size size, int type, GpuMat& m)
{
    ensureSizeIsEnough(size.height, size.width, type, m);
}

}} // namespace cv { namespace gpu

#endif // __OPENCV_CORE_GPUINL_HPP__
