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

#ifndef OPENCV_CUDEV_PTR2D_GLOB_HPP
#define OPENCV_CUDEV_PTR2D_GLOB_HPP

#include "../common.hpp"
#include "traits.hpp"

namespace cv { namespace cudev {

//! @addtogroup cudev
//! @{

/** @brief Structure similar to cv::cudev::GlobPtrSz but containing only a pointer and row step.

Width and height fields are excluded due to performance reasons. The structure is intended
for internal use or for users who write device code.
 */
template <typename T> struct GlobPtr
{
    typedef T   value_type;
    typedef int index_type;

    T* data;

    //! stride between two consecutive rows in bytes. Step is stored always and everywhere in bytes!!!
    size_t step;

    __device__ __forceinline__       T* row(int y)       { return (      T*)( (      uchar*)data + y * step); }
    __device__ __forceinline__ const T* row(int y) const { return (const T*)( (const uchar*)data + y * step); }

    __device__ __forceinline__       T& operator ()(int y, int x)       { return row(y)[x]; }
    __device__ __forceinline__ const T& operator ()(int y, int x) const { return row(y)[x]; }
};

/** @brief Lightweight class encapsulating pitched memory on a GPU and passed to nvcc-compiled code (CUDA
kernels).

Typically, it is used internally by OpenCV and by users who write device code. You can call
its members from both host and device code.
 */
template <typename T> struct GlobPtrSz : GlobPtr<T>
{
    int rows, cols;
};

template <typename T>
__host__ __device__ GlobPtr<T> globPtr(T* data, size_t step)
{
    GlobPtr<T> p;
    p.data = data;
    p.step = step;
    return p;
}

template <typename T>
__host__ __device__ GlobPtrSz<T> globPtr(T* data, size_t step, int rows, int cols)
{
    GlobPtrSz<T> p;
    p.data = data;
    p.step = step;
    p.rows = rows;
    p.cols = cols;
    return p;
}

template <typename T>
__host__ GlobPtrSz<T> globPtr(const GpuMat& mat)
{
    GlobPtrSz<T> p;
    p.data = (T*) mat.data;
    p.step = mat.step;
    p.rows = mat.rows;
    p.cols = mat.cols;
    return p;
}

template <typename T> struct PtrTraits< GlobPtrSz<T> > : PtrTraitsBase<GlobPtrSz<T>, GlobPtr<T> >
{
};

//! @}

}}

#endif
