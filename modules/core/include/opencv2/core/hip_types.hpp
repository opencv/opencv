// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_CORE_HIP_TYPES_HPP
#define OPENCV_CORE_HIP_TYPES_HPP

#ifndef __cplusplus
#  error hip_types.hpp header must be compiled as C++
#endif

#if defined(__OPENCV_BUILD) && defined(__clang__)
#pragma clang diagnostic ignored "-Winconsistent-missing-override"
#endif
#if defined(__OPENCV_BUILD) && defined(__GNUC__) && __GNUC__ >= 5
#pragma GCC diagnostic ignored "-Wsuggest-override"
#endif
#ifdef __HIPCC__
    #include <hip/hip_runtime.h>  // defines __forceinline__, __host__, __device__
    #define __CV_HIP_HOST_DEVICE__ __host__ __device__ __forceinline__
#else
    #define __CV_HIP_HOST_DEVICE__
#endif

#include "opencv2/core/cvdef.h"
#include "opencv2/core.hpp"
namespace cv {
    namespace hip{
        template <typename T> struct DevPtr
        {
            typedef T elem_type;
            typedef int index_type;

            enum { elem_size = sizeof(elem_type) };

            T* data;

            __CV_HIP_HOST_DEVICE__ DevPtr() : data(0) {}
            __CV_HIP_HOST_DEVICE__ DevPtr(T* data_) : data(data_) {}

            __CV_HIP_HOST_DEVICE__ size_t elemSize() const { return elem_size; }
            __CV_HIP_HOST_DEVICE__ operator       T*()       { return data; }
            __CV_HIP_HOST_DEVICE__ operator const T*() const { return data; }
        };
        template<typename T> struct PtrSz : public DevPtr<T>
        {
            __CV_HIP_HOST_DEVICE__ PtrSz() : size(0) {}
            __CV_HIP_HOST_DEVICE__ PtrSz(T* data_, size_t size_) : DevPtr<T>(data_), size(size_) {}

            size_t size;
        };
        template <typename T> struct PtrStep : public DevPtr<T>
        {
            __CV_HIP_HOST_DEVICE__ PtrStep() : step(0) {}
            __CV_HIP_HOST_DEVICE__ PtrStep(T* data_, size_t step_) : DevPtr<T>(data_), step(step_) {}

            size_t step;
            __CV_HIP_HOST_DEVICE__ const T* ptr(int y) const { return (const T*)((const uchar*)this->data + y * step); }
            __CV_HIP_HOST_DEVICE__       T* ptr(int y)       { return (     T*)((      uchar*)this->data + y * step); }     
            __CV_HIP_HOST_DEVICE__       T& operator ()(int y, int x)       { return ptr(y)[x]; }
            __CV_HIP_HOST_DEVICE__ const T& operator ()(int y, int x) const { return ptr(y)[x]; }
        };
        template <typename T> struct PtrStepSz : public PtrStep<T>
        {
            __CV_HIP_HOST_DEVICE__ PtrStepSz() : cols(0), rows(0) {}
            __CV_HIP_HOST_DEVICE__ PtrStepSz(int rows_, int cols_, T* data_, size_t step_) : PtrStep<T>(data_, step_), cols(cols_), rows(rows_) {}

            int cols, rows;
            CV_NODISCARD_STD __CV_HIP_HOST_DEVICE__ Size size() const { return {cols, rows}; }
            CV_NODISCARD_STD __CV_HIP_HOST_DEVICE__ T& operator ()(const Point &pos)       { return (*this)(pos.y, pos.x); }
            CV_NODISCARD_STD __CV_HIP_HOST_DEVICE__ const T& operator ()(const Point &pos) const { return (*this)(pos.y, pos.x); }
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

#endif // OPENCV_CORE_HIP_TYPES_HPP