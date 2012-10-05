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

#ifndef __OPENCV_GPU_SCAN_HPP__
#define __OPENCV_GPU_SCAN_HPP__

#include "common.hpp"

namespace cv { namespace gpu { namespace device
{
    enum ScanKind { EXCLUSIVE = 0,  INCLUSIVE = 1 };

    template <ScanKind Kind, typename T, typename F> struct WarpScan
    {
        __device__ __forceinline__ WarpScan() {}
        __device__ __forceinline__ WarpScan(const WarpScan& other) { (void)other; }

        __device__ __forceinline__ T operator()( volatile T *ptr , const unsigned int idx)
        {
            const unsigned int lane = idx & 31;
            F op;

            if ( lane >=  1) ptr [idx ] = op(ptr [idx -  1], ptr [idx]);
            if ( lane >=  2) ptr [idx ] = op(ptr [idx -  2], ptr [idx]);
            if ( lane >=  4) ptr [idx ] = op(ptr [idx -  4], ptr [idx]);
            if ( lane >=  8) ptr [idx ] = op(ptr [idx -  8], ptr [idx]);
            if ( lane >= 16) ptr [idx ] = op(ptr [idx - 16], ptr [idx]);

            if( Kind == INCLUSIVE )
                return ptr [idx];
            else
                return (lane > 0) ? ptr [idx - 1] : 0;
        }

        __device__ __forceinline__ unsigned int index(const unsigned int tid)
        {
            return tid;
        }

        __device__ __forceinline__ void init(volatile T *ptr){}

        static const int warp_offset      = 0;

        typedef WarpScan<INCLUSIVE, T, F>  merge;
    };

    template <ScanKind Kind , typename T, typename F> struct WarpScanNoComp
    {
        __device__ __forceinline__ WarpScanNoComp() {}
        __device__ __forceinline__ WarpScanNoComp(const WarpScanNoComp& other) { (void)other; }

        __device__ __forceinline__ T operator()( volatile T *ptr , const unsigned int idx)
        {
            const unsigned int lane = threadIdx.x & 31;
            F op;

            ptr [idx ] = op(ptr [idx -  1], ptr [idx]);
            ptr [idx ] = op(ptr [idx -  2], ptr [idx]);
            ptr [idx ] = op(ptr [idx -  4], ptr [idx]);
            ptr [idx ] = op(ptr [idx -  8], ptr [idx]);
            ptr [idx ] = op(ptr [idx - 16], ptr [idx]);

            if( Kind == INCLUSIVE )
                return ptr [idx];
            else
                return (lane > 0) ? ptr [idx - 1] : 0;
        }

        __device__ __forceinline__ unsigned int index(const unsigned int tid)
        {
            return (tid >> warp_log) * warp_smem_stride + 16 + (tid & warp_mask);
        }

        __device__ __forceinline__ void init(volatile T *ptr)
        {
            ptr[threadIdx.x] = 0;
        }

        static const int warp_smem_stride = 32 + 16 + 1;
        static const int warp_offset      = 16;
        static const int warp_log         = 5;
        static const int warp_mask        = 31;

        typedef WarpScanNoComp<INCLUSIVE, T, F> merge;
    };

    template <ScanKind Kind , typename T, typename Sc, typename F> struct BlockScan
    {
        __device__ __forceinline__ BlockScan() {}
        __device__ __forceinline__ BlockScan(const BlockScan& other) { (void)other; }

        __device__ __forceinline__ T operator()(volatile T *ptr)
        {
            const unsigned int tid  = threadIdx.x;
            const unsigned int lane = tid & warp_mask;
            const unsigned int warp = tid >> warp_log;

            Sc scan;
            typename Sc::merge merge_scan;
            const unsigned int idx = scan.index(tid);

            T val = scan(ptr, idx);
            __syncthreads ();

            if( warp == 0)
                scan.init(ptr);
            __syncthreads ();

            if( lane == 31 )
                ptr [scan.warp_offset + warp ] = (Kind == INCLUSIVE) ? val : ptr [idx];
            __syncthreads ();

            if( warp == 0 )
                merge_scan(ptr, idx);
            __syncthreads();

            if ( warp > 0)
                val = ptr [scan.warp_offset + warp - 1] + val;
            __syncthreads ();

            ptr[idx] = val;
            __syncthreads ();

            return val ;
        }

        static const int warp_log  = 5;
        static const int warp_mask = 31;
    };
}}}

#endif // __OPENCV_GPU_SCAN_HPP__
