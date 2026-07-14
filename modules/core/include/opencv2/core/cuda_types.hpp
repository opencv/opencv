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
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

#ifndef OPENCV_CORE_CUDA_TYPES_HPP
#define OPENCV_CORE_CUDA_TYPES_HPP

#ifndef __cplusplus
#  error cuda_types.hpp header must be compiled as C++
#endif

#if defined(__OPENCV_BUILD) && defined(__clang__)
#pragma clang diagnostic ignored "-Winconsistent-missing-override"
#endif
#if defined(__OPENCV_BUILD) && defined(__GNUC__) && __GNUC__ >= 5
#pragma GCC diagnostic ignored "-Wsuggest-override"
#endif

/** @file
 * @deprecated Use @ref cudev instead.
 */

//! @cond IGNORED

#ifdef __CUDACC__
    #define __CV_CUDA_HOST_DEVICE__ __host__ __device__ __forceinline__
#else
    #define __CV_CUDA_HOST_DEVICE__
#endif

#include "opencv2/core/cvdef.h"
#include "opencv2/core.hpp"

namespace cv
{
    namespace cuda
    {

        // Simple lightweight structures that encapsulates information about an image on device.
        // It is intended to pass to nvcc-compiled code. GpuMat depends on headers that nvcc can't compile

        template <typename T> struct DevPtr
        {
            typedef T elem_type;
            typedef int index_type;

            enum { elem_size = sizeof(elem_type) };

            T* data;

            __CV_CUDA_HOST_DEVICE__ DevPtr() : data(0) {}
            __CV_CUDA_HOST_DEVICE__ DevPtr(T* data_) : data(data_) {}

            __CV_CUDA_HOST_DEVICE__ size_t elemSize() const { return elem_size; }
            __CV_CUDA_HOST_DEVICE__ operator       T*()       { return data; }
            __CV_CUDA_HOST_DEVICE__ operator const T*() const { return data; }
        };

        template <typename T> struct PtrSz : public DevPtr<T>
        {
            __CV_CUDA_HOST_DEVICE__ PtrSz() : size(0) {}
            __CV_CUDA_HOST_DEVICE__ PtrSz(T* data_, size_t size_) : DevPtr<T>(data_), size(size_) {}

            size_t size;
        };

        template <typename T> struct PtrStep : public DevPtr<T>
        {
            __CV_CUDA_HOST_DEVICE__ PtrStep() : step(0) {}
            __CV_CUDA_HOST_DEVICE__ PtrStep(T* data_, size_t step_) : DevPtr<T>(data_), step(step_) {}

            size_t step;

            __CV_CUDA_HOST_DEVICE__       T* ptr(int y = 0)       { return (      T*)( (      char*)(((DevPtr<T>*)this)->data) + y * step); }
            __CV_CUDA_HOST_DEVICE__ const T* ptr(int y = 0) const { return (const T*)( (const char*)(((DevPtr<T>*)this)->data) + y * step); }

            __CV_CUDA_HOST_DEVICE__       T& operator ()(int y, int x)       { return ptr(y)[x]; }
            __CV_CUDA_HOST_DEVICE__ const T& operator ()(int y, int x) const { return ptr(y)[x]; }
        };

        template <typename T> struct PtrStepSz : public PtrStep<T>
        {
            __CV_CUDA_HOST_DEVICE__ PtrStepSz() : cols(0), rows(0) {}
            __CV_CUDA_HOST_DEVICE__ PtrStepSz(int rows_, int cols_, T* data_, size_t step_)
                : PtrStep<T>(data_, step_), cols(cols_), rows(rows_) {}

            template <typename U>
            explicit PtrStepSz(const PtrStepSz<U>& d) : PtrStep<T>((T*)d.data, d.step), cols(d.cols), rows(d.rows){}

            int cols;
            int rows;

            CV_NODISCARD_STD __CV_CUDA_HOST_DEVICE__ Size size() const { return {cols, rows}; }
            CV_NODISCARD_STD __CV_CUDA_HOST_DEVICE__ T& operator ()(const Point &pos)       { return (*this)(pos.y, pos.x); }
            CV_NODISCARD_STD __CV_CUDA_HOST_DEVICE__ const T& operator ()(const Point &pos) const { return (*this)(pos.y, pos.x); }
            using PtrStep<T>::operator();
        };

        typedef PtrStepSz<unsigned char> PtrStepSzb;
        typedef PtrStepSz<unsigned short> PtrStepSzus;
        typedef PtrStepSz<float> PtrStepSzf;
        typedef PtrStepSz<int> PtrStepSzi;

        typedef PtrStep<unsigned char> PtrStepb;
        typedef PtrStep<unsigned short> PtrStepus;
        typedef PtrStep<float> PtrStepf;
        typedef PtrStep<int> PtrStepi;

    }
}

//! @endcond

#endif /* OPENCV_CORE_CUDA_TYPES_HPP */
