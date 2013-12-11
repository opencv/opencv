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

#pragma once

#ifndef __OPENCV_CUDEV_PTR2D_GPUMAT_DETAIL_HPP__
#define __OPENCV_CUDEV_PTR2D_GPUMAT_DETAIL_HPP__

#include "../gpumat.hpp"

namespace cv { namespace cudev {

template <typename T>
__host__ GpuMat_<T>::GpuMat_()
    : GpuMat()
{
    flags = (flags & ~CV_MAT_TYPE_MASK) | DataType<T>::type;
}

template <typename T>
__host__ GpuMat_<T>::GpuMat_(int arows, int acols)
    : GpuMat(arows, acols, DataType<T>::type)
{
}

template <typename T>
__host__ GpuMat_<T>::GpuMat_(Size asize)
    : GpuMat(asize.height, asize.width, DataType<T>::type)
{
}

template <typename T>
__host__ GpuMat_<T>::GpuMat_(int arows, int acols, Scalar val)
    : GpuMat(arows, acols, DataType<T>::type, val)
{
}

template <typename T>
__host__ GpuMat_<T>::GpuMat_(Size asize, Scalar val)
    : GpuMat(asize.height, asize.width, DataType<T>::type, val)
{
}

template <typename T>
__host__ GpuMat_<T>::GpuMat_(const GpuMat_& m)
    : GpuMat(m)
{
}

template <typename T>
__host__ GpuMat_<T>::GpuMat_(const GpuMat& m)
    : GpuMat()
{
    flags = (flags & ~CV_MAT_TYPE_MASK) | DataType<T>::type;

    if (DataType<T>::type == m.type())
    {
        GpuMat::operator =(m);
        return;
    }

    if (DataType<T>::depth == m.depth())
    {
        GpuMat::operator =(m.reshape(DataType<T>::channels, m.rows));
        return;
    }

    CV_Assert( DataType<T>::channels == m.channels() );
    m.convertTo(*this, type());
}

template <typename T>
__host__ GpuMat_<T>::GpuMat_(int arows, int acols, T* adata, size_t astep)
    : GpuMat(arows, acols, DataType<T>::type, adata, astep)
{
}

template <typename T>
__host__ GpuMat_<T>::GpuMat_(Size asize, T* adata, size_t astep)
    : GpuMat(asize.height, asize.width, DataType<T>::type, adata, astep)
{
}

template <typename T>
__host__ GpuMat_<T>::GpuMat_(const GpuMat_& m, Range arowRange, Range acolRange)
    : GpuMat(m, arowRange, acolRange)
{
}

template <typename T>
__host__ GpuMat_<T>::GpuMat_(const GpuMat_& m, Rect roi)
    : GpuMat(m, roi)
{
}

template <typename T>
__host__ GpuMat_<T>::GpuMat_(InputArray arr)
    : GpuMat()
{
    flags = (flags & ~CV_MAT_TYPE_MASK) | DataType<T>::type;
    upload(arr);
}

template <typename T>
__host__ GpuMat_<T>& GpuMat_<T>::operator =(const GpuMat_& m)
{
    GpuMat::operator =(m);
    return *this;
}

template <typename T>
__host__ void GpuMat_<T>::create(int arows, int acols)
{
    GpuMat::create(arows, acols, DataType<T>::type);
}

template <typename T>
__host__ void GpuMat_<T>::create(Size asize)
{
    GpuMat::create(asize, DataType<T>::type);
}

template <typename T>
__host__ void GpuMat_<T>::swap(GpuMat_& mat)
{
    GpuMat::swap(mat);
}

template <typename T>
__host__ void GpuMat_<T>::upload(InputArray arr)
{
    CV_Assert( arr.type() == DataType<T>::type );
    GpuMat::upload(arr);
}

template <typename T>
__host__ void GpuMat_<T>::upload(InputArray arr, Stream& stream)
{
    CV_Assert( arr.type() == DataType<T>::type );
    GpuMat::upload(arr, stream);
}

template <typename T>
__host__ GpuMat_<T>::operator GlobPtrSz<T>() const
{
    return globPtr((T*) data, step, rows, cols);
}

template <typename T>
__host__ GpuMat_<T>::operator GlobPtr<T>() const
{
    return globPtr((T*) data, step);
}

template <typename T>
__host__ GpuMat_<T> GpuMat_<T>::clone() const
{
    return GpuMat_(GpuMat::clone());
}

template <typename T>
__host__ GpuMat_<T> GpuMat_<T>::row(int y) const
{
    return GpuMat_(*this, Range(y, y+1), Range::all());
}

template <typename T>
__host__ GpuMat_<T> GpuMat_<T>::col(int x) const
{
    return GpuMat_(*this, Range::all(), Range(x, x+1));
}

template <typename T>
__host__ GpuMat_<T> GpuMat_<T>::rowRange(int startrow, int endrow) const
{
    return GpuMat_(*this, Range(startrow, endrow), Range::all());
}

template <typename T>
__host__ GpuMat_<T> GpuMat_<T>::rowRange(Range r) const
{
    return GpuMat_(*this, r, Range::all());
}

template <typename T>
__host__ GpuMat_<T> GpuMat_<T>::colRange(int startcol, int endcol) const
{
    return GpuMat_(*this, Range::all(), Range(startcol, endcol));
}

template <typename T>
__host__ GpuMat_<T> GpuMat_<T>::colRange(Range r) const
{
    return GpuMat_(*this, Range::all(), r);
}

template <typename T>
__host__ GpuMat_<T> GpuMat_<T>::operator ()(Range _rowRange, Range _colRange) const
{
    return GpuMat_(*this, _rowRange, _colRange);
}

template <typename T>
__host__ GpuMat_<T> GpuMat_<T>::operator ()(Rect roi) const
{
    return GpuMat_(*this, roi);
}

template <typename T>
__host__ GpuMat_<T>& GpuMat_<T>::adjustROI(int dtop, int dbottom, int dleft, int dright)
{
    return (GpuMat_<T>&)(GpuMat::adjustROI(dtop, dbottom, dleft, dright));
}

template <typename T>
__host__ size_t GpuMat_<T>::elemSize() const
{
    CV_DbgAssert( GpuMat::elemSize() == sizeof(T) );
    return sizeof(T);
}

template <typename T>
__host__ size_t GpuMat_<T>::elemSize1() const
{
    CV_DbgAssert( GpuMat::elemSize1() == sizeof(T) / DataType<T>::channels );
    return sizeof(T) / DataType<T>::channels;
}

template <typename T>
__host__ int GpuMat_<T>::type() const
{
    CV_DbgAssert( GpuMat::type() == DataType<T>::type );
    return DataType<T>::type;
}

template <typename T>
__host__ int GpuMat_<T>::depth() const
{
    CV_DbgAssert( GpuMat::depth() == DataType<T>::depth );
    return DataType<T>::depth;
}

template <typename T>
__host__ int GpuMat_<T>::channels() const
{
    CV_DbgAssert( GpuMat::channels() == DataType<T>::channels );
    return DataType<T>::channels;
}

template <typename T>
__host__ size_t GpuMat_<T>::stepT() const
{
    return step / elemSize();
}

template <typename T>
__host__ size_t GpuMat_<T>::step1() const
{
    return step / elemSize1();
}

template <typename T>
__host__ T* GpuMat_<T>::operator [](int y)
{
    return (T*)ptr(y);
}

template <typename T>
__host__ const T* GpuMat_<T>::operator [](int y) const
{
    return (const T*)ptr(y);
}

template <typename T> template <class Body>
__host__ GpuMat_<T>::GpuMat_(const Expr<Body>& expr)
    : GpuMat()
{
    flags = (flags & ~CV_MAT_TYPE_MASK) | DataType<T>::type;
    *this = expr;
}

template <typename T> template <class Body>
__host__ GpuMat_<T>& GpuMat_<T>::operator =(const Expr<Body>& expr)
{
    expr.body.assignTo(*this);
    return *this;
}

template <typename T> template <class Body>
__host__ GpuMat_<T>& GpuMat_<T>::assign(const Expr<Body>& expr, Stream& stream)
{
    expr.body.assignTo(*this, stream);
    return *this;
}

}}

// Input / Output Arrays

namespace cv {

template<typename _Tp>
__host__ _InputArray::_InputArray(const cudev::GpuMat_<_Tp>& m)
    : flags(FIXED_TYPE + GPU_MAT + DataType<_Tp>::type), obj((void*)&m)
{}

template<typename _Tp>
__host__ _OutputArray::_OutputArray(cudev::GpuMat_<_Tp>& m)
    : _InputArray(m)
{}

template<typename _Tp>
__host__ _OutputArray::_OutputArray(const cudev::GpuMat_<_Tp>& m)
    : _InputArray(m)
{
    flags |= FIXED_SIZE;
}

}

#endif
