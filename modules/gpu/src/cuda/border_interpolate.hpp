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

namespace cv { namespace gpu {

    struct BrdReflect101 
    {
        BrdReflect101(int len): last(len - 1) {}

        __device__ int idx_low(int i) const
        {
            return abs(i);
        }

        __device__ int idx_high(int i) const 
        {
            return last - abs(last - i);
        }

        __device__ int idx(int i) const
        {
            return abs(idx_high(i));
        }

        bool is_range_safe(int mini, int maxi) const 
        {
            return -last <= mini && maxi <= 2 * last;
        }

        int last;
    };


    template <typename T>
    struct BrdRowReflect101: BrdReflect101
    {
        BrdRowReflect101(int len): BrdReflect101(len) {}

        __device__ float at_low(int i, const T* data) const 
        {
            return data[idx_low(i)];
        }

        __device__ float at_high(int i, const T* data) const 
        {
            return data[idx_high(i)];
        }
    };


    template <typename T>
    struct BrdColReflect101: BrdReflect101
    {
        BrdColReflect101(int len, int step): BrdReflect101(len), step(step) {}

        __device__ float at_low(int i, const T* data) const 
        {
            return data[idx_low(i) * step];
        }

        __device__ float at_high(int i, const T* data) const 
        {
            return data[idx_high(i) * step];
        }

        int step;
    };


    struct BrdReplicate
    {
        BrdReplicate(int len): last(len - 1) {}

        __device__ int idx_low(int i) const
        {
            return max(i, 0);
        }

        __device__ int idx_high(int i) const 
        {
            return min(i, last);
        }

        __device__ int idx(int i) const
        {
            return max(min(i, last), 0);
        }

        bool is_range_safe(int mini, int maxi) const 
        {
            return true;
        }

        int last;
    };


    template <typename T>
    struct BrdRowReplicate: BrdReplicate
    {
        BrdRowReplicate(int len): BrdReplicate(len) {}

        __device__ float at_low(int i, const T* data) const 
        {
            return data[idx_low(i)];
        }

        __device__ float at_high(int i, const T* data) const 
        {
            return data[idx_high(i)];
        }
    };


    template <typename T>
    struct BrdColReplicate: BrdReplicate
    {
        BrdColReplicate(int len, int step): BrdReplicate(len), step(step) {}

        __device__ float at_low(int i, const T* data) const 
        {
            return data[idx_low(i) * step];
        }

        __device__ float at_high(int i, const T* data) const 
        {
            return data[idx_high(i) * step];
        }

        int step;
    };

}}