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
// any express or bpied warranties, including, but not limited to, the bpied
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

#ifndef __OPENCV_GPU_BORDER_INTERPOLATE_HPP__
#define __OPENCV_GPU_BORDER_INTERPOLATE_HPP__

#include "opencv2/gpu/device/saturate_cast.hpp"
#include "opencv2/gpu/device/vecmath.hpp"

namespace cv { namespace gpu { namespace device
{
    struct BrdReflect101 
    {
        explicit BrdReflect101(int len): last(len - 1) {}

        __device__ __forceinline__ int idx_low(int i) const
        {
            return abs(i);
        }

        __device__ __forceinline__ int idx_high(int i) const 
        {
            return last - abs(last - i);
        }

        __device__ __forceinline__ int idx(int i) const
        {
            return idx_low(idx_high(i));
        }

        bool is_range_safe(int mini, int maxi) const 
        {
            return -last <= mini && maxi <= 2 * last;
        }

    private:
        int last;
    };


    template <typename D>
    struct BrdRowReflect101: BrdReflect101
    {
        explicit BrdRowReflect101(int len): BrdReflect101(len) {}

        template <typename T>
        __device__ __forceinline__ D at_low(int i, const T* data) const 
        {
            return saturate_cast<D>(data[idx_low(i)]);
        }

        template <typename T>
        __device__ __forceinline__ D at_high(int i, const T* data) const 
        {
            return saturate_cast<D>(data[idx_high(i)]);
        }
    };


    template <typename D>
    struct BrdColReflect101: BrdReflect101
    {
        BrdColReflect101(int len, int step): BrdReflect101(len), step(step) {}

        template <typename T>
        __device__ __forceinline__ D at_low(int i, const T* data) const 
        {
            return saturate_cast<D>(data[idx_low(i) * step]);
        }

        template <typename T>
        __device__ __forceinline__ D at_high(int i, const T* data) const 
        {
            return saturate_cast<D>(data[idx_high(i) * step]);
        }

    private:
        int step;
    };


    struct BrdReplicate
    {
        explicit BrdReplicate(int len): last(len - 1) {}

        __device__ __forceinline__ int idx_low(int i) const
        {
            return max(i, 0);
        }

        __device__ __forceinline__ int idx_high(int i) const 
        {
            return min(i, last);
        }

        __device__ __forceinline__ int idx(int i) const
        {
            return idx_low(idx_high(i));
        }

        bool is_range_safe(int mini, int maxi) const 
        {
            return true;
        }

    private:
        int last;
    };


    template <typename D>
    struct BrdRowReplicate: BrdReplicate
    {
        explicit BrdRowReplicate(int len): BrdReplicate(len) {}

        template <typename T>
        __device__ __forceinline__ D at_low(int i, const T* data) const 
        {
            return saturate_cast<D>(data[idx_low(i)]);
        }

        template <typename T>
        __device__ __forceinline__ D at_high(int i, const T* data) const 
        {
            return saturate_cast<D>(data[idx_high(i)]);
        }
    };


    template <typename D>
    struct BrdColReplicate: BrdReplicate
    {
        BrdColReplicate(int len, int step): BrdReplicate(len), step(step) {}

        template <typename T>
        __device__ __forceinline__ D at_low(int i, const T* data) const 
        {
            return saturate_cast<D>(data[idx_low(i) * step]);
        }

        template <typename T>
        __device__ __forceinline__ D at_high(int i, const T* data) const 
        {
            return saturate_cast<D>(data[idx_high(i) * step]);
        }

    private:
        int step;
    };

    template <typename D>
    struct BrdRowConstant
    {
        explicit BrdRowConstant(int len_, const D& val_ = VecTraits<D>::all(0)): len(len_), val(val_) {}

        template <typename T>
        __device__ __forceinline__ D at_low(int i, const T* data) const 
        {
            return i >= 0 ? saturate_cast<D>(data[i]) : val;
        }

        template <typename T>
        __device__ __forceinline__ D at_high(int i, const T* data) const 
        {
            return i < len ? saturate_cast<D>(data[i]) : val;
        }

        bool is_range_safe(int mini, int maxi) const 
        {
            return true;
        }

    private:
        int len;
        D val;
    };

    template <typename D>
    struct BrdColConstant
    {
        BrdColConstant(int len_, int step_, const D& val_ = VecTraits<D>::all(0)): len(len_), step(step_), val(val_) {}

        template <typename T>
        __device__ __forceinline__ D at_low(int i, const T* data) const 
        {
            return i >= 0 ? saturate_cast<D>(data[i * step]) : val;
        }

        template <typename T>
        __device__ __forceinline__ D at_high(int i, const T* data) const 
        {
            return i < len ? saturate_cast<D>(data[i * step]) : val;
        }

        bool is_range_safe(int mini, int maxi) const 
        {
            return true;
        }

    private:
        int len;
        int step;
        D val;
    };
}}}

#endif // __OPENCV_GPU_BORDER_INTERPOLATE_HPP__
