/*
 * By downloading, copying, installing or using the software you agree to this license.
 * If you do not agree to this license, do not download, install,
 * copy or use the software.
 *
 *
 *                           License Agreement
 *                For Open Source Computer Vision Library
 *                        (3-clause BSD License)
 *
 * Copyright (C) 2014-2015, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *   * Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 *   * Neither the names of the copyright holders nor the names of the contributors
 *     may be used to endorse or promote products derived from this software
 *     without specific prior written permission.
 *
 * This software is provided by the copyright holders and contributors "as is" and
 * any express or implied warranties, including, but not limited to, the implied
 * warranties of merchantability and fitness for a particular purpose are disclaimed.
 * In no event shall copyright holders or contributors be liable for any direct,
 * indirect, incidental, special, exemplary, or consequential damages
 * (including, but not limited to, procurement of substitute goods or services;
 * loss of use, data, or profits; or business interruption) however caused
 * and on any theory of liability, whether in contract, strict liability,
 * or tort (including negligence or otherwise) arising in any way out of
 * the use of this software, even if advised of the possibility of such damage.
 */

#ifndef CAROTENE_TYPES_HPP
#define CAROTENE_TYPES_HPP

#include <carotene/definitions.hpp>
#include <stdint.h>
#include <cstddef>

#ifndef UINT32_MAX
    #define UINT32_MAX (4294967295U)
#endif

namespace CAROTENE_NS {
    using std::size_t;
    using std::ptrdiff_t;

    typedef int8_t   s8;
    typedef uint8_t  u8;
    typedef int16_t  s16;
    typedef uint16_t u16;
    typedef int32_t  s32;
    typedef uint32_t u32;
    typedef float    f32;
    typedef int64_t  s64;
    typedef uint64_t u64;
    typedef double   f64;

    typedef ptrdiff_t  stride_t;

    enum CONVERT_POLICY
    {
        CONVERT_POLICY_WRAP,
        CONVERT_POLICY_SATURATE
    };

    enum BORDER_MODE
    {
        BORDER_MODE_UNDEFINED,
        BORDER_MODE_CONSTANT,
        BORDER_MODE_REPLICATE,
        BORDER_MODE_REFLECT,
        BORDER_MODE_REFLECT101,
        BORDER_MODE_WRAP
    };

    enum FLIP_MODE
    {
        FLIP_HORIZONTAL_MODE = 1,
        FLIP_VERTICAL_MODE = 2,
        FLIP_BOTH_MODE = FLIP_HORIZONTAL_MODE | FLIP_VERTICAL_MODE
    };

    enum COLOR_SPACE
    {
        COLOR_SPACE_BT601,
        COLOR_SPACE_BT709
    };

    struct Size2D {
        Size2D() : width(0), height(0) {}
        Size2D(size_t width_, size_t height_) : width(width_), height(height_) {}

        size_t width;
        size_t height;

        inline size_t total() const
        {
            return width * height;
        }
    };

    struct Margin {
        Margin() : left(0), right(0), top(0), bottom(0) {}
        Margin(size_t left_, size_t right_, size_t top_, size_t bottom_)
            : left(left_), right(right_), top(top_), bottom(bottom_) {}

        // these are measured in elements
        size_t left, right, top, bottom;
    };

    struct KeypointStore {
        virtual void push(f32 kpX, f32 kpY, f32 kpSize, f32 kpAngle=-1, f32 kpResponse=0, s32 kpOctave=0, s32 kpClass_id=-1) = 0;
        virtual ~KeypointStore() {};
    };
}

#endif
