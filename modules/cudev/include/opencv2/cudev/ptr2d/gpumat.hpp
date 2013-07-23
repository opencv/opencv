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

#ifndef __OPENCV_CUDEV_PTR2D_GPUMAT_HPP__
#define __OPENCV_CUDEV_PTR2D_GPUMAT_HPP__

#include "../common.hpp"
#include "../util/vec_traits.hpp"
#include "../expr/expr.hpp"
#include "glob.hpp"

namespace cv { namespace cudev {

template <typename T>
class GpuMat_ : public GpuMat
{
public:
    typedef T value_type;

    //! default constructor
    __host__ GpuMat_();

    //! constructs GpuMat of the specified size
    __host__ GpuMat_(int arows, int acols);
    __host__ explicit GpuMat_(Size asize);

    //! constucts GpuMat and fills it with the specified value
    __host__ GpuMat_(int arows, int acols, Scalar val);
    __host__ GpuMat_(Size asize, Scalar val);

    //! copy constructor
    __host__ GpuMat_(const GpuMat_& m);

    //! copy/conversion contructor. If m is of different type, it's converted
    __host__ explicit GpuMat_(const GpuMat& m);

    //! constructs a matrix on top of user-allocated data. step is in bytes(!!!), regardless of the type
    __host__ GpuMat_(int arows, int acols, T* adata, size_t astep = Mat::AUTO_STEP);
    __host__ GpuMat_(Size asize, T* adata, size_t astep = Mat::AUTO_STEP);

    //! selects a submatrix
    __host__ GpuMat_(const GpuMat_& m, Range arowRange, Range acolRange);
    __host__ GpuMat_(const GpuMat_& m, Rect roi);

    //! builds GpuMat from host memory (Blocking call)
    __host__ explicit GpuMat_(InputArray arr);

    //! assignment operators
    __host__ GpuMat_& operator =(const GpuMat_& m);

    //! allocates new GpuMat data unless the GpuMat already has specified size and type
    __host__ void create(int arows, int acols);
    __host__ void create(Size asize);

    //! swaps with other smart pointer
    __host__ void swap(GpuMat_& mat);

    //! pefroms upload data to GpuMat (Blocking call)
    __host__ void upload(InputArray arr);

    //! pefroms upload data to GpuMat (Non-Blocking call)
    __host__ void upload(InputArray arr, Stream& stream);

    //! convert to GlobPtr
    __host__ operator GlobPtrSz<T>() const;
    __host__ operator GlobPtr<T>() const;

    //! overridden forms of GpuMat::row() etc.
    __host__ GpuMat_ clone() const;
    __host__ GpuMat_ row(int y) const;
    __host__ GpuMat_ col(int x) const;
    __host__ GpuMat_ rowRange(int startrow, int endrow) const;
    __host__ GpuMat_ rowRange(Range r) const;
    __host__ GpuMat_ colRange(int startcol, int endcol) const;
    __host__ GpuMat_ colRange(Range r) const;
    __host__ GpuMat_ operator ()(Range rowRange, Range colRange) const;
    __host__ GpuMat_ operator ()(Rect roi) const;
    __host__ GpuMat_& adjustROI(int dtop, int dbottom, int dleft, int dright);

    //! overridden forms of GpuMat::elemSize() etc.
    __host__ size_t elemSize() const;
    __host__ size_t elemSize1() const;
    __host__ int type() const;
    __host__ int depth() const;
    __host__ int channels() const;
    __host__ size_t step1() const;

    //! returns step()/sizeof(T)
    __host__ size_t stepT() const;

    //! more convenient forms of row and element access operators
    __host__ T* operator [](int y);
    __host__ const T* operator [](int y) const;

    //! expression templates
    template <class Body> __host__ GpuMat_(const Expr<Body>& expr);
    template <class Body> __host__ GpuMat_& operator =(const Expr<Body>& expr);
    template <class Body> __host__ GpuMat_& assign(const Expr<Body>& expr, Stream& stream);
};

//! creates alternative GpuMat header for the same data, with different
//! number of channels and/or different number of rows. see cvReshape.
template <int cn, typename T>
__host__ GpuMat_<typename MakeVec<typename VecTraits<T>::elem_type, cn>::type>
reshape_(const GpuMat_<T>& mat, int rows = 0)
{
    GpuMat_<typename MakeVec<typename VecTraits<T>::elem_type, cn>::type> dst(mat.reshape(cn, rows));
    return dst;
}

template <typename T> struct PtrTraits< GpuMat_<T> > : PtrTraitsBase<GpuMat_<T>, GlobPtr<T> >
{
};

}}

#include "detail/gpumat.hpp"

#endif
