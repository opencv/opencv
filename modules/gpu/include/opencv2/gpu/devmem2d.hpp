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

#ifndef __OPENCV_GPU_DEVMEM2D_HPP__
#define __OPENCV_GPU_DEVMEM2D_HPP__

namespace cv
{    
    namespace gpu
    {
        // Simple lightweight structures that encapsulates information about an image on device.
        // It is intended to pass to nvcc-compiled code. GpuMat depends on headers that nvcc can't compile

#if defined(__CUDACC__) 
    #define __CV_GPU_HOST_DEVICE__ __host__ __device__ 
#else
    #define __CV_GPU_HOST_DEVICE__
#endif
        
        template <typename T> struct DevMem2D_
        {            
            int cols;
            int rows;
            T* data;
            size_t step;
            
            DevMem2D_() : cols(0), rows(0), data(0), step(0) {}
            
            DevMem2D_(int rows_, int cols_, T *data_, size_t step_)
                : cols(cols_), rows(rows_), data(data_), step(step_) {}
            
            template <typename U>            
            explicit DevMem2D_(const DevMem2D_<U>& d)
                : cols(d.cols), rows(d.rows), data((T*)d.data), step(d.step) {}
            
            typedef T elem_type;
            enum { elem_size = sizeof(elem_type) };

            __CV_GPU_HOST_DEVICE__ size_t elemSize() const { return elem_size; }
            __CV_GPU_HOST_DEVICE__ T* ptr(int y = 0) { return (T*)( (char*)data + y * step ); }
            __CV_GPU_HOST_DEVICE__ const T* ptr(int y = 0) const { return (const T*)( (const char*)data + y * step ); }            
        };
 
        template<typename T> struct PtrStep_
        {
            T* data;
            size_t step;

            PtrStep_() : data(0), step(0) {}            
            PtrStep_(const DevMem2D_<T>& mem) : data(mem.data), step(mem.step) {}

            typedef T elem_type;
            enum { elem_size = sizeof(elem_type) };

            __CV_GPU_HOST_DEVICE__ size_t elemSize() const { return elem_size; }
            __CV_GPU_HOST_DEVICE__ T* ptr(int y = 0) { return (T*)( (char*)data + y * step); }
            __CV_GPU_HOST_DEVICE__ const T* ptr(int y = 0) const { return (const T*)( (const char*)data + y * step); }
        };
       
        template<typename T> struct PtrElemStep_ : public PtrStep_<T>
        {                   
            PtrElemStep_(const DevMem2D_<T>& mem) : PtrStep_<T>(mem) 
            {
                step /= elem_size;             
            }
        private:            
            template <bool> struct StaticCheck;
            template <> struct StaticCheck<true>{};            

            StaticCheck<256 % sizeof(T) == 0>  ElemStepTypeCheck;
        };

        typedef DevMem2D_<unsigned char> DevMem2D;
        typedef DevMem2D_<float> DevMem2Df;
        typedef DevMem2D_<int> DevMem2Di;

        typedef PtrStep_<unsigned char> PtrStep;
        typedef PtrStep_<float> PtrStepf;
        typedef PtrStep_<int> PtrStepi;

        typedef PtrElemStep_<unsigned char> PtrElemStep;
        typedef PtrElemStep_<float> PtrElemStepf;
        typedef PtrElemStep_<int> PtrElemStepi;

#undef __CV_GPU_HOST_DEVICE__
    }    
}

#endif /* __OPENCV_GPU_DEVMEM2D_HPP__ */
