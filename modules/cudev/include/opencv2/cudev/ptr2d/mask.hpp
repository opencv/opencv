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

#ifndef __OPENCV_CUDEV_PTR2D_MASK_HPP__
#define __OPENCV_CUDEV_PTR2D_MASK_HPP__

#include "../common.hpp"
#include "traits.hpp"

namespace cv { namespace cudev {

struct WithOutMask
{
    typedef bool value_type;
    typedef int  index_type;

    __device__ __forceinline__ bool operator ()(int, int) const
    {
        return true;
    }
};

template <class MaskPtr> struct SingleMaskChannels
{
    typedef typename PtrTraits<MaskPtr>::value_type value_type;
    typedef typename PtrTraits<MaskPtr>::index_type index_type;

    MaskPtr mask;
    int channels;

    __device__ __forceinline__ value_type operator()(index_type y, index_type x) const
    {
        return mask(y, x / channels);
    }

};

template <class MaskPtr> struct SingleMaskChannelsSz : SingleMaskChannels<MaskPtr>
{
    int rows, cols;
};

template <class MaskPtr>
__host__ SingleMaskChannelsSz<typename PtrTraits<MaskPtr>::ptr_type>
singleMaskChannels(const MaskPtr& mask, int channels)
{
    SingleMaskChannelsSz<typename PtrTraits<MaskPtr>::ptr_type> ptr;
    ptr.mask = shrinkPtr(mask);
    ptr.channels = channels;
    ptr.rows = getRows(mask);
    ptr.cols = getCols(mask) * channels;
    return ptr;
}

template <class MaskPtr> struct PtrTraits< SingleMaskChannelsSz<MaskPtr> > : PtrTraitsBase<SingleMaskChannelsSz<MaskPtr>, SingleMaskChannels<MaskPtr> >
{
};

}}

#endif
