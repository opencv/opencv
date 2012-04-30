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
//     and/or other GpuMaterials provided with the distribution.
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

#ifndef __OPENCV_CORE_DevMem2D_HPP__
#define __OPENCV_CORE_DevMem2D_HPP__

#ifdef __cplusplus

#ifdef __CUDACC__
    #define __CV_GPU_HOST_DEVICE__ __host__ __device__ __forceinline__
#else
    #define __CV_GPU_HOST_DEVICE__
#endif

namespace cv
{
    namespace gpu
    {
        // Simple lightweight structures that encapsulates information about an image on device.
        // It is intended to pass to nvcc-compiled code. GpuMat depends on headers that nvcc can't compile

        template <bool expr> struct StaticAssert;
        template <> struct StaticAssert<true> {static __CV_GPU_HOST_DEVICE__ void check(){}};

		template<typename T> struct DevPtr
		{
			typedef T elem_type;
			typedef int index_type;

            enum { elem_size = sizeof(elem_type) };

            T* data;

            __CV_GPU_HOST_DEVICE__ DevPtr() : data(0) {}
            __CV_GPU_HOST_DEVICE__ DevPtr(T* data_) : data(data_) {}

            __CV_GPU_HOST_DEVICE__ size_t elemSize() const { return elem_size; }
            __CV_GPU_HOST_DEVICE__ operator       T*()       { return data; }
            __CV_GPU_HOST_DEVICE__ operator const T*() const { return data; }
        };

        template<typename T> struct PtrSz : public DevPtr<T>
        {
            __CV_GPU_HOST_DEVICE__ PtrSz() : size(0) {}
            __CV_GPU_HOST_DEVICE__ PtrSz(T* data_, size_t size_) : DevPtr<T>(data_), size(size_) {}

            size_t size;
        };

        template<typename T> struct PtrStep : public DevPtr<T>
        {
            __CV_GPU_HOST_DEVICE__ PtrStep() : step(0) {}
            __CV_GPU_HOST_DEVICE__ PtrStep(T* data_, size_t step_) : DevPtr<T>(data_), step(step_) {}

            /** \brief stride between two consecutive rows in bytes. Step is stored always and everywhere in bytes!!! */
            size_t step;

            __CV_GPU_HOST_DEVICE__       T* ptr(int y = 0)       { return (      T*)( (      char*)DevPtr<T>::data + y * step); }
            __CV_GPU_HOST_DEVICE__ const T* ptr(int y = 0) const { return (const T*)( (const char*)DevPtr<T>::data + y * step); }

            __CV_GPU_HOST_DEVICE__       T& operator ()(int y, int x)       { return ptr(y)[x]; }
            __CV_GPU_HOST_DEVICE__ const T& operator ()(int y, int x) const { return ptr(y)[x]; }
        };

        template <typename T> struct PtrStepSz : public PtrStep<T>
        {
            __CV_GPU_HOST_DEVICE__ PtrStepSz() : cols(0), rows(0) {}
            __CV_GPU_HOST_DEVICE__ PtrStepSz(int rows_, int cols_, T* data_, size_t step_)
                : PtrStep<T>(data_, step_), cols(cols_), rows(rows_) {}

            int cols;
            int rows;
        };

        template <typename T> struct DevMem2D_ : public PtrStepSz<T>
        {
            DevMem2D_() {}
            DevMem2D_(int rows_, int cols_, T* data_, size_t step_) : PtrStepSz<T>(rows_, cols_, data_, step_) {}

            template <typename U>
            explicit DevMem2D_(const DevMem2D_<U>& d) : PtrStepSz<T>(d.rows, d.cols, (T*)d.data, d.step) {}
        };

        template<typename T> struct PtrElemStep_ : public PtrStep<T>
        {
            PtrElemStep_(const DevMem2D_<T>& mem) : PtrStep<T>(mem.data, mem.step)
            {
                StaticAssert<256 % sizeof(T) == 0>::check();

                PtrStep<T>::step /= PtrStep<T>::elem_size;
            }
            __CV_GPU_HOST_DEVICE__ T* ptr(int y = 0) { return PtrStep<T>::data + y * PtrStep<T>::step; }
            __CV_GPU_HOST_DEVICE__ const T* ptr(int y = 0) const { return PtrStep<T>::data + y * PtrStep<T>::step; }

            __CV_GPU_HOST_DEVICE__ T& operator ()(int y, int x) { return ptr(y)[x]; }
            __CV_GPU_HOST_DEVICE__ const T& operator ()(int y, int x) const { return ptr(y)[x]; }
        };

        template<typename T> struct PtrStep_ : public PtrStep<T>
        {
            PtrStep_() {}
            PtrStep_(const DevMem2D_<T>& mem) : PtrStep<T>(mem.data, mem.step) {}
        };

        typedef DevMem2D_<unsigned char> DevMem2Db;
        typedef DevMem2Db DevMem2D;
        typedef DevMem2D_<float> DevMem2Df;
        typedef DevMem2D_<int> DevMem2Di;

        typedef PtrStep<unsigned char> PtrStepb;
        typedef PtrStep<float> PtrStepf;
        typedef PtrStep<int> PtrStepi;

        typedef PtrElemStep_<unsigned char> PtrElemStep;
        typedef PtrElemStep_<float> PtrElemStepf;
        typedef PtrElemStep_<int> PtrElemStepi;
    }
}

#endif // __cplusplus

#endif /* __OPENCV_GPU_DevMem2D_HPP__ */
