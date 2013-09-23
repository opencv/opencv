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

#ifndef __ARITHM_FUNC_TRAITS_HPP__
#define __ARITHM_FUNC_TRAITS_HPP__

#include <cstddef>

namespace arithm
{
    template <size_t src_size, size_t dst_size> struct ArithmFuncTraits
    {
        enum { simple_block_dim_x = 32 };
        enum { simple_block_dim_y = 8 };

        enum { smart_block_dim_x = 32 };
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 1 };
    };

    template <> struct ArithmFuncTraits<1, 1>
    {
        enum { simple_block_dim_x = 32 };
        enum { simple_block_dim_y = 8 };

        enum { smart_block_dim_x = 32 };
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct ArithmFuncTraits<1, 2>
    {
        enum { simple_block_dim_x = 32 };
        enum { simple_block_dim_y = 8 };

        enum { smart_block_dim_x = 32 };
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct ArithmFuncTraits<1, 4>
    {
        enum { simple_block_dim_x = 32 };
        enum { simple_block_dim_y = 8 };

        enum { smart_block_dim_x = 32 };
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };

    template <> struct ArithmFuncTraits<2, 1>
    {
        enum { simple_block_dim_x = 32 };
        enum { simple_block_dim_y = 8 };

        enum { smart_block_dim_x = 32 };
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct ArithmFuncTraits<2, 2>
    {
        enum { simple_block_dim_x = 32 };
        enum { simple_block_dim_y = 8 };

        enum { smart_block_dim_x = 32 };
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct ArithmFuncTraits<2, 4>
    {
        enum { simple_block_dim_x = 32 };
        enum { simple_block_dim_y = 8 };

        enum { smart_block_dim_x = 32 };
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };

    template <> struct ArithmFuncTraits<4, 1>
    {
        enum { simple_block_dim_x = 32 };
        enum { simple_block_dim_y = 8 };

        enum { smart_block_dim_x = 32 };
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct ArithmFuncTraits<4, 2>
    {
        enum { simple_block_dim_x = 32 };
        enum { simple_block_dim_y = 8 };

        enum { smart_block_dim_x = 32 };
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct ArithmFuncTraits<4, 4>
    {
        enum { simple_block_dim_x = 32 };
        enum { simple_block_dim_y = 8 };

        enum { smart_block_dim_x = 32 };
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
}

#endif // __ARITHM_FUNC_TRAITS_HPP__
