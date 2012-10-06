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

#ifndef __OPENCV_GPU_TRANSFORM_HPP__
#define __OPENCV_GPU_TRANSFORM_HPP__

#include "common.hpp"
#include "utility.hpp"
#include "detail/transform_detail.hpp"

namespace cv { namespace gpu { namespace device
{
    template <typename T, typename D, typename UnOp, typename Mask>
    static inline void transform(PtrStepSz<T> src, PtrStepSz<D> dst, UnOp op, const Mask& mask, cudaStream_t stream)
    {
        typedef TransformFunctorTraits<UnOp> ft;
        transform_detail::TransformDispatcher<VecTraits<T>::cn == 1 && VecTraits<D>::cn == 1 && ft::smart_shift != 1>::call(src, dst, op, mask, stream);
    }

    template <typename T1, typename T2, typename D, typename BinOp, typename Mask>
    static inline void transform(PtrStepSz<T1> src1, PtrStepSz<T2> src2, PtrStepSz<D> dst, BinOp op, const Mask& mask, cudaStream_t stream)
    {
        typedef TransformFunctorTraits<BinOp> ft;
        transform_detail::TransformDispatcher<VecTraits<T1>::cn == 1 && VecTraits<T2>::cn == 1 && VecTraits<D>::cn == 1 && ft::smart_shift != 1>::call(src1, src2, dst, op, mask, stream);
    }
}}}

#endif // __OPENCV_GPU_TRANSFORM_HPP__
