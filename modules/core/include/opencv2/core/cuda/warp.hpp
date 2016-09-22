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

#ifndef OPENCV_CUDA_DEVICE_WARP_HPP
#define OPENCV_CUDA_DEVICE_WARP_HPP

/** @file
 * @deprecated Use @ref cudev instead.
 */

//! @cond IGNORED

namespace cv { namespace cuda { namespace device
{
    struct Warp
    {
        enum
        {
            LOG_WARP_SIZE = 5,
            WARP_SIZE     = 1 << LOG_WARP_SIZE,
            STRIDE        = WARP_SIZE
        };

        /** \brief Returns the warp lane ID of the calling thread. */
        static __device__ __forceinline__ unsigned int laneId()
        {
            unsigned int ret;
            asm("mov.u32 %0, %laneid;" : "=r"(ret) );
            return ret;
        }

        template<typename It, typename T>
        static __device__ __forceinline__ void fill(It beg, It end, const T& value)
        {
            for(It t = beg + laneId(); t < end; t += STRIDE)
                *t = value;
        }

        template<typename InIt, typename OutIt>
        static __device__ __forceinline__ OutIt copy(InIt beg, InIt end, OutIt out)
        {
            for(InIt t = beg + laneId(); t < end; t += STRIDE, out += STRIDE)
                *out = *t;
            return out;
        }

        template<typename InIt, typename OutIt, class UnOp>
        static __device__ __forceinline__ OutIt transform(InIt beg, InIt end, OutIt out, UnOp op)
        {
            for(InIt t = beg + laneId(); t < end; t += STRIDE, out += STRIDE)
                *out = op(*t);
            return out;
        }

        template<typename InIt1, typename InIt2, typename OutIt, class BinOp>
        static __device__ __forceinline__ OutIt transform(InIt1 beg1, InIt1 end1, InIt2 beg2, OutIt out, BinOp op)
        {
            unsigned int lane = laneId();

            InIt1 t1 = beg1 + lane;
            InIt2 t2 = beg2 + lane;
            for(; t1 < end1; t1 += STRIDE, t2 += STRIDE, out += STRIDE)
                *out = op(*t1, *t2);
            return out;
        }

        template <class T, class BinOp>
        static __device__ __forceinline__ T reduce(volatile T *ptr, BinOp op)
        {
            const unsigned int lane = laneId();

            if (lane < 16)
            {
                T partial = ptr[lane];

                ptr[lane] = partial = op(partial, ptr[lane + 16]);
                ptr[lane] = partial = op(partial, ptr[lane + 8]);
                ptr[lane] = partial = op(partial, ptr[lane + 4]);
                ptr[lane] = partial = op(partial, ptr[lane + 2]);
                ptr[lane] = partial = op(partial, ptr[lane + 1]);
            }

            return *ptr;
        }

        template<typename OutIt, typename T>
        static __device__ __forceinline__ void yota(OutIt beg, OutIt end, T value)
        {
            unsigned int lane = laneId();
            value += lane;

            for(OutIt t = beg + lane; t < end; t += STRIDE, value += STRIDE)
                *t = value;
        }
    };
}}} // namespace cv { namespace cuda { namespace cudev

//! @endcond

#endif /* OPENCV_CUDA_DEVICE_WARP_HPP */
