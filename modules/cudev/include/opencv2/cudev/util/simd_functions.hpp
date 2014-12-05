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

/*
 * Copyright (c) 2013 NVIDIA Corporation. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *   Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 *
 *   Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 *   Neither the name of NVIDIA Corporation nor the names of its contributors
 *   may be used to endorse or promote products derived from this software
 *   without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#ifndef __OPENCV_CUDEV_UTIL_SIMD_FUNCTIONS_HPP__
#define __OPENCV_CUDEV_UTIL_SIMD_FUNCTIONS_HPP__

#include "../common.hpp"

/*
  This header file contains inline functions that implement intra-word SIMD
  operations, that are hardware accelerated on sm_3x (Kepler) GPUs. Efficient
  emulation code paths are provided for earlier architectures (sm_1x, sm_2x)
  to make the code portable across all GPUs supported by CUDA. The following
  functions are currently implemented:

  vadd2(a,b)      per-halfword unsigned addition, with wrap-around: a + b
  vsub2(a,b)      per-halfword unsigned subtraction, with wrap-around: a - b
  vabsdiff2(a,b)  per-halfword unsigned absolute difference: |a - b|
  vavg2(a,b)      per-halfword unsigned average: (a + b) / 2
  vavrg2(a,b)     per-halfword unsigned rounded average: (a + b + 1) / 2
  vseteq2(a,b)    per-halfword unsigned comparison: a == b ? 1 : 0
  vcmpeq2(a,b)    per-halfword unsigned comparison: a == b ? 0xffff : 0
  vsetge2(a,b)    per-halfword unsigned comparison: a >= b ? 1 : 0
  vcmpge2(a,b)    per-halfword unsigned comparison: a >= b ? 0xffff : 0
  vsetgt2(a,b)    per-halfword unsigned comparison: a > b ? 1 : 0
  vcmpgt2(a,b)    per-halfword unsigned comparison: a > b ? 0xffff : 0
  vsetle2(a,b)    per-halfword unsigned comparison: a <= b ? 1 : 0
  vcmple2(a,b)    per-halfword unsigned comparison: a <= b ? 0xffff : 0
  vsetlt2(a,b)    per-halfword unsigned comparison: a < b ? 1 : 0
  vcmplt2(a,b)    per-halfword unsigned comparison: a < b ? 0xffff : 0
  vsetne2(a,b)    per-halfword unsigned comparison: a != b ? 1 : 0
  vcmpne2(a,b)    per-halfword unsigned comparison: a != b ? 0xffff : 0
  vmax2(a,b)      per-halfword unsigned maximum: max(a, b)
  vmin2(a,b)      per-halfword unsigned minimum: min(a, b)

  vadd4(a,b)      per-byte unsigned addition, with wrap-around: a + b
  vsub4(a,b)      per-byte unsigned subtraction, with wrap-around: a - b
  vabsdiff4(a,b)  per-byte unsigned absolute difference: |a - b|
  vavg4(a,b)      per-byte unsigned average: (a + b) / 2
  vavrg4(a,b)     per-byte unsigned rounded average: (a + b + 1) / 2
  vseteq4(a,b)    per-byte unsigned comparison: a == b ? 1 : 0
  vcmpeq4(a,b)    per-byte unsigned comparison: a == b ? 0xff : 0
  vsetge4(a,b)    per-byte unsigned comparison: a >= b ? 1 : 0
  vcmpge4(a,b)    per-byte unsigned comparison: a >= b ? 0xff : 0
  vsetgt4(a,b)    per-byte unsigned comparison: a > b ? 1 : 0
  vcmpgt4(a,b)    per-byte unsigned comparison: a > b ? 0xff : 0
  vsetle4(a,b)    per-byte unsigned comparison: a <= b ? 1 : 0
  vcmple4(a,b)    per-byte unsigned comparison: a <= b ? 0xff : 0
  vsetlt4(a,b)    per-byte unsigned comparison: a < b ? 1 : 0
  vcmplt4(a,b)    per-byte unsigned comparison: a < b ? 0xff : 0
  vsetne4(a,b)    per-byte unsigned comparison: a != b ? 1: 0
  vcmpne4(a,b)    per-byte unsigned comparison: a != b ? 0xff: 0
  vmax4(a,b)      per-byte unsigned maximum: max(a, b)
  vmin4(a,b)      per-byte unsigned minimum: min(a, b)
*/

namespace cv { namespace cudev {

//! @addtogroup cudev
//! @{

// 2

__device__ __forceinline__ uint vadd2(uint a, uint b)
{
    uint r = 0;

#if CV_CUDEV_ARCH >= 300
    asm("vadd2.u32.u32.u32.sat %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
#elif CV_CUDEV_ARCH >= 200
    asm("vadd.u32.u32.u32.sat %0.h0, %1.h0, %2.h0, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
    asm("vadd.u32.u32.u32.sat %0.h1, %1.h1, %2.h1, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
#else
    uint s;
    s = a ^ b;          // sum bits
    r = a + b;          // actual sum
    s = s ^ r;          // determine carry-ins for each bit position
    s = s & 0x00010000; // carry-in to high word (= carry-out from low word)
    r = r - s;          // subtract out carry-out from low word
#endif

    return r;
}

__device__ __forceinline__ uint vsub2(uint a, uint b)
{
    uint r = 0;

#if CV_CUDEV_ARCH >= 300
    asm("vsub2.u32.u32.u32.sat %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
#elif CV_CUDEV_ARCH >= 200
    asm("vsub.u32.u32.u32.sat %0.h0, %1.h0, %2.h0, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
    asm("vsub.u32.u32.u32.sat %0.h1, %1.h1, %2.h1, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
#else
    uint s;
    s = a ^ b;          // sum bits
    r = a - b;          // actual sum
    s = s ^ r;          // determine carry-ins for each bit position
    s = s & 0x00010000; // borrow to high word
    r = r + s;          // compensate for borrow from low word
#endif

    return r;
}

__device__ __forceinline__ uint vabsdiff2(uint a, uint b)
{
    uint r = 0;

#if CV_CUDEV_ARCH >= 300
    asm("vabsdiff2.u32.u32.u32.sat %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
#elif CV_CUDEV_ARCH >= 200
    asm("vabsdiff.u32.u32.u32.sat %0.h0, %1.h0, %2.h0, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
    asm("vabsdiff.u32.u32.u32.sat %0.h1, %1.h1, %2.h1, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
#else
    uint s, t, u, v;
    s = a & 0x0000ffff; // extract low halfword
    r = b & 0x0000ffff; // extract low halfword
    u = ::max(r, s);    // maximum of low halfwords
    v = ::min(r, s);    // minimum of low halfwords
    s = a & 0xffff0000; // extract high halfword
    r = b & 0xffff0000; // extract high halfword
    t = ::max(r, s);    // maximum of high halfwords
    s = ::min(r, s);    // minimum of high halfwords
    r = u | t;          // maximum of both halfwords
    s = v | s;          // minimum of both halfwords
    r = r - s;          // |a - b| = max(a,b) - min(a,b);
#endif

    return r;
}

__device__ __forceinline__ uint vavg2(uint a, uint b)
{
    uint r, s;

    // HAKMEM #23: a + b = 2 * (a & b) + (a ^ b) ==>
    // (a + b) / 2 = (a & b) + ((a ^ b) >> 1)
    s = a ^ b;
    r = a & b;
    s = s & 0xfffefffe; // ensure shift doesn't cross halfword boundaries
    s = s >> 1;
    s = r + s;

    return s;
}

__device__ __forceinline__ uint vavrg2(uint a, uint b)
{
    uint r = 0;

#if CV_CUDEV_ARCH >= 300
    asm("vavrg2.u32.u32.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
#else
    // HAKMEM #23: a + b = 2 * (a | b) - (a ^ b) ==>
    // (a + b + 1) / 2 = (a | b) - ((a ^ b) >> 1)
    uint s;
    s = a ^ b;
    r = a | b;
    s = s & 0xfffefffe; // ensure shift doesn't cross half-word boundaries
    s = s >> 1;
    r = r - s;
#endif

    return r;
}

__device__ __forceinline__ uint vseteq2(uint a, uint b)
{
    uint r = 0;

#if CV_CUDEV_ARCH >= 300
    asm("vset2.u32.u32.eq %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
#else
    // inspired by Alan Mycroft's null-byte detection algorithm:
    // null_byte(x) = ((x - 0x01010101) & (~x & 0x80808080))
    uint c;
    r = a ^ b;          // 0x0000 if a == b
    c = r | 0x80008000; // set msbs, to catch carry out
    r = r ^ c;          // extract msbs, msb = 1 if r < 0x8000
    c = c - 0x00010001; // msb = 0, if r was 0x0000 or 0x8000
    c = r & ~c;         // msb = 1, if r was 0x0000
    r = c >> 15;        // convert to bool
#endif

    return r;
}

__device__ __forceinline__ uint vcmpeq2(uint a, uint b)
{
    uint r, c;

#if CV_CUDEV_ARCH >= 300
    r = vseteq2(a, b);
    c = r << 16;        // convert bool
    r = c - r;          //  into mask
#else
    // inspired by Alan Mycroft's null-byte detection algorithm:
    // null_byte(x) = ((x - 0x01010101) & (~x & 0x80808080))
    r = a ^ b;          // 0x0000 if a == b
    c = r | 0x80008000; // set msbs, to catch carry out
    r = r ^ c;          // extract msbs, msb = 1 if r < 0x8000
    c = c - 0x00010001; // msb = 0, if r was 0x0000 or 0x8000
    c = r & ~c;         // msb = 1, if r was 0x0000
    r = c >> 15;        // convert
    r = c - r;          //  msbs to
    r = c | r;          //   mask
#endif

    return r;
}

__device__ __forceinline__ uint vsetge2(uint a, uint b)
{
    uint r = 0;

#if CV_CUDEV_ARCH >= 300
    asm("vset2.u32.u32.ge %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
#else
    uint c;
    asm("not.b32 %0, %0;" : "+r"(b));
    c = vavrg2(a, b);   // (a + ~b + 1) / 2 = (a - b) / 2
    c = c & 0x80008000; // msb = carry-outs
    r = c >> 15;        // convert to bool
#endif

    return r;
}

__device__ __forceinline__ uint vcmpge2(uint a, uint b)
{
    uint r, c;

#if CV_CUDEV_ARCH >= 300
    r = vsetge2(a, b);
    c = r << 16;        // convert bool
    r = c - r;          //  into mask
#else
    asm("not.b32 %0, %0;" : "+r"(b));
    c = vavrg2(a, b);   // (a + ~b + 1) / 2 = (a - b) / 2
    c = c & 0x80008000; // msb = carry-outs
    r = c >> 15;        // convert
    r = c - r;          //  msbs to
    r = c | r;          //   mask
#endif

    return r;
}

__device__ __forceinline__ uint vsetgt2(uint a, uint b)
{
    uint r = 0;

#if CV_CUDEV_ARCH >= 300
    asm("vset2.u32.u32.gt %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
#else
    uint c;
    asm("not.b32 %0, %0;" : "+r"(b));
    c = vavg2(a, b);    // (a + ~b) / 2 = (a - b) / 2 [rounded down]
    c = c & 0x80008000; // msbs = carry-outs
    r = c >> 15;        // convert to bool
#endif

    return r;
}

__device__ __forceinline__ uint vcmpgt2(uint a, uint b)
{
    uint r, c;

#if CV_CUDEV_ARCH >= 300
    r = vsetgt2(a, b);
    c = r << 16;        // convert bool
    r = c - r;          //  into mask
#else
    asm("not.b32 %0, %0;" : "+r"(b));
    c = vavg2(a, b);    // (a + ~b) / 2 = (a - b) / 2 [rounded down]
    c = c & 0x80008000; // msbs = carry-outs
    r = c >> 15;        // convert
    r = c - r;          //  msbs to
    r = c | r;          //   mask
#endif

    return r;
}

__device__ __forceinline__ uint vsetle2(uint a, uint b)
{
    uint r = 0;

#if CV_CUDEV_ARCH >= 300
    asm("vset2.u32.u32.le %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
#else
    uint c;
    asm("not.b32 %0, %0;" : "+r"(a));
    c = vavrg2(a, b);   // (b + ~a + 1) / 2 = (b - a) / 2
    c = c & 0x80008000; // msb = carry-outs
    r = c >> 15;        // convert to bool
#endif

    return r;
}

__device__ __forceinline__ uint vcmple2(uint a, uint b)
{
    uint r, c;

#if CV_CUDEV_ARCH >= 300
    r = vsetle2(a, b);
    c = r << 16;        // convert bool
    r = c - r;          //  into mask
#else
    asm("not.b32 %0, %0;" : "+r"(a));
    c = vavrg2(a, b);   // (b + ~a + 1) / 2 = (b - a) / 2
    c = c & 0x80008000; // msb = carry-outs
    r = c >> 15;        // convert
    r = c - r;          //  msbs to
    r = c | r;          //   mask
#endif

    return r;
}

__device__ __forceinline__ uint vsetlt2(uint a, uint b)
{
    uint r = 0;

#if CV_CUDEV_ARCH >= 300
    asm("vset2.u32.u32.lt %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
#else
    uint c;
    asm("not.b32 %0, %0;" : "+r"(a));
    c = vavg2(a, b);    // (b + ~a) / 2 = (b - a) / 2 [rounded down]
    c = c & 0x80008000; // msb = carry-outs
    r = c >> 15;        // convert to bool
#endif

    return r;
}

__device__ __forceinline__ uint vcmplt2(uint a, uint b)
{
    uint r, c;

#if CV_CUDEV_ARCH >= 300
    r = vsetlt2(a, b);
    c = r << 16;        // convert bool
    r = c - r;          //  into mask
#else
    asm("not.b32 %0, %0;" : "+r"(a));
    c = vavg2(a, b);    // (b + ~a) / 2 = (b - a) / 2 [rounded down]
    c = c & 0x80008000; // msb = carry-outs
    r = c >> 15;        // convert
    r = c - r;          //  msbs to
    r = c | r;          //   mask
#endif

    return r;
}

__device__ __forceinline__ uint vsetne2(uint a, uint b)
{
    uint r = 0;

#if CV_CUDEV_ARCH >= 300
    asm ("vset2.u32.u32.ne %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
#else
    // inspired by Alan Mycroft's null-byte detection algorithm:
    // null_byte(x) = ((x - 0x01010101) & (~x & 0x80808080))
    uint c;
    r = a ^ b;          // 0x0000 if a == b
    c = r | 0x80008000; // set msbs, to catch carry out
    c = c - 0x00010001; // msb = 0, if r was 0x0000 or 0x8000
    c = r | c;          // msb = 1, if r was not 0x0000
    c = c & 0x80008000; // extract msbs
    r = c >> 15;        // convert to bool
#endif

    return r;
}

__device__ __forceinline__ uint vcmpne2(uint a, uint b)
{
    uint r, c;

#if CV_CUDEV_ARCH >= 300
    r = vsetne2(a, b);
    c = r << 16;        // convert bool
    r = c - r;          //  into mask
#else
    // inspired by Alan Mycroft's null-byte detection algorithm:
    // null_byte(x) = ((x - 0x01010101) & (~x & 0x80808080))
    r = a ^ b;          // 0x0000 if a == b
    c = r | 0x80008000; // set msbs, to catch carry out
    c = c - 0x00010001; // msb = 0, if r was 0x0000 or 0x8000
    c = r | c;          // msb = 1, if r was not 0x0000
    c = c & 0x80008000; // extract msbs
    r = c >> 15;        // convert
    r = c - r;          //  msbs to
    r = c | r;          //   mask
#endif

    return r;
}

__device__ __forceinline__ uint vmax2(uint a, uint b)
{
    uint r = 0;

#if CV_CUDEV_ARCH >= 300
    asm("vmax2.u32.u32.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
#elif CV_CUDEV_ARCH >= 200
    asm("vmax.u32.u32.u32 %0.h0, %1.h0, %2.h0, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
    asm("vmax.u32.u32.u32 %0.h1, %1.h1, %2.h1, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
#else
    uint s, t, u;
    r = a & 0x0000ffff; // extract low halfword
    s = b & 0x0000ffff; // extract low halfword
    t = ::max(r, s);    // maximum of low halfwords
    r = a & 0xffff0000; // extract high halfword
    s = b & 0xffff0000; // extract high halfword
    u = ::max(r, s);    // maximum of high halfwords
    r = t | u;          // combine halfword maximums
#endif

    return r;
}

__device__ __forceinline__ uint vmin2(uint a, uint b)
{
    uint r = 0;

#if CV_CUDEV_ARCH >= 300
    asm("vmin2.u32.u32.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
#elif CV_CUDEV_ARCH >= 200
    asm("vmin.u32.u32.u32 %0.h0, %1.h0, %2.h0, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
    asm("vmin.u32.u32.u32 %0.h1, %1.h1, %2.h1, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
#else
    uint s, t, u;
    r = a & 0x0000ffff; // extract low halfword
    s = b & 0x0000ffff; // extract low halfword
    t = ::min(r, s);    // minimum of low halfwords
    r = a & 0xffff0000; // extract high halfword
    s = b & 0xffff0000; // extract high halfword
    u = ::min(r, s);    // minimum of high halfwords
    r = t | u;          // combine halfword minimums
#endif

    return r;
}

// 4

__device__ __forceinline__ uint vadd4(uint a, uint b)
{
    uint r = 0;

#if CV_CUDEV_ARCH >= 300
    asm("vadd4.u32.u32.u32.sat %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
#elif CV_CUDEV_ARCH >= 200
    asm("vadd.u32.u32.u32.sat %0.b0, %1.b0, %2.b0, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
    asm("vadd.u32.u32.u32.sat %0.b1, %1.b1, %2.b1, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
    asm("vadd.u32.u32.u32.sat %0.b2, %1.b2, %2.b2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
    asm("vadd.u32.u32.u32.sat %0.b3, %1.b3, %2.b3, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
#else
    uint s, t;
    s = a ^ b;          // sum bits
    r = a & 0x7f7f7f7f; // clear msbs
    t = b & 0x7f7f7f7f; // clear msbs
    s = s & 0x80808080; // msb sum bits
    r = r + t;          // add without msbs, record carry-out in msbs
    r = r ^ s;          // sum of msb sum and carry-in bits, w/o carry-out
#endif /* CV_CUDEV_ARCH >= 300 */

    return r;
}

__device__ __forceinline__ uint vsub4(uint a, uint b)
{
    uint r = 0;

#if CV_CUDEV_ARCH >= 300
    asm("vsub4.u32.u32.u32.sat %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
#elif CV_CUDEV_ARCH >= 200
    asm("vsub.u32.u32.u32.sat %0.b0, %1.b0, %2.b0, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
    asm("vsub.u32.u32.u32.sat %0.b1, %1.b1, %2.b1, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
    asm("vsub.u32.u32.u32.sat %0.b2, %1.b2, %2.b2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
    asm("vsub.u32.u32.u32.sat %0.b3, %1.b3, %2.b3, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
#else
    uint s, t;
    s = a ^ ~b;         // inverted sum bits
    r = a | 0x80808080; // set msbs
    t = b & 0x7f7f7f7f; // clear msbs
    s = s & 0x80808080; // inverted msb sum bits
    r = r - t;          // subtract w/o msbs, record inverted borrows in msb
    r = r ^ s;          // combine inverted msb sum bits and borrows
#endif

    return r;
}

__device__ __forceinline__ uint vavg4(uint a, uint b)
{
    uint r, s;

    // HAKMEM #23: a + b = 2 * (a & b) + (a ^ b) ==>
    // (a + b) / 2 = (a & b) + ((a ^ b) >> 1)
    s = a ^ b;
    r = a & b;
    s = s & 0xfefefefe; // ensure following shift doesn't cross byte boundaries
    s = s >> 1;
    s = r + s;

    return s;
}

__device__ __forceinline__ uint vavrg4(uint a, uint b)
{
    uint r = 0;

#if CV_CUDEV_ARCH >= 300
    asm("vavrg4.u32.u32.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
#else
    // HAKMEM #23: a + b = 2 * (a | b) - (a ^ b) ==>
    // (a + b + 1) / 2 = (a | b) - ((a ^ b) >> 1)
    uint c;
    c = a ^ b;
    r = a | b;
    c = c & 0xfefefefe; // ensure following shift doesn't cross byte boundaries
    c = c >> 1;
    r = r - c;
#endif

    return r;
}

__device__ __forceinline__ uint vseteq4(uint a, uint b)
{
    uint r = 0;

#if CV_CUDEV_ARCH >= 300
    asm("vset4.u32.u32.eq %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
#else
    // inspired by Alan Mycroft's null-byte detection algorithm:
    // null_byte(x) = ((x - 0x01010101) & (~x & 0x80808080))
    uint c;
    r = a ^ b;          // 0x00 if a == b
    c = r | 0x80808080; // set msbs, to catch carry out
    r = r ^ c;          // extract msbs, msb = 1 if r < 0x80
    c = c - 0x01010101; // msb = 0, if r was 0x00 or 0x80
    c = r & ~c;         // msb = 1, if r was 0x00
    r = c >> 7;         // convert to bool
#endif

    return r;
}

__device__ __forceinline__ uint vcmpeq4(uint a, uint b)
{
    uint r, t;

#if CV_CUDEV_ARCH >= 300
    r = vseteq4(a, b);
    t = r << 8;         // convert bool
    r = t - r;          //  to mask
#else
    // inspired by Alan Mycroft's null-byte detection algorithm:
    // null_byte(x) = ((x - 0x01010101) & (~x & 0x80808080))
    t = a ^ b;          // 0x00 if a == b
    r = t | 0x80808080; // set msbs, to catch carry out
    t = t ^ r;          // extract msbs, msb = 1 if t < 0x80
    r = r - 0x01010101; // msb = 0, if t was 0x00 or 0x80
    r = t & ~r;         // msb = 1, if t was 0x00
    t = r >> 7;         // build mask
    t = r - t;          //  from
    r = t | r;          //   msbs
#endif

    return r;
}

__device__ __forceinline__ uint vsetle4(uint a, uint b)
{
    uint r = 0;

#if CV_CUDEV_ARCH >= 300
    asm("vset4.u32.u32.le %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
#else
    uint c;
    asm("not.b32 %0, %0;" : "+r"(a));
    c = vavrg4(a, b);   // (b + ~a + 1) / 2 = (b - a) / 2
    c = c & 0x80808080; // msb = carry-outs
    r = c >> 7;         // convert to bool
#endif

    return r;
}

__device__ __forceinline__ uint vcmple4(uint a, uint b)
{
    uint r, c;

#if CV_CUDEV_ARCH >= 300
    r = vsetle4(a, b);
    c = r << 8;         // convert bool
    r = c - r;          //  to mask
#else
    asm("not.b32 %0, %0;" : "+r"(a));
    c = vavrg4(a, b);   // (b + ~a + 1) / 2 = (b - a) / 2
    c = c & 0x80808080; // msbs = carry-outs
    r = c >> 7;         // convert
    r = c - r;          //  msbs to
    r = c | r;          //   mask
#endif

    return r;
}

__device__ __forceinline__ uint vsetlt4(uint a, uint b)
{
    uint r = 0;

#if CV_CUDEV_ARCH >= 300
    asm("vset4.u32.u32.lt %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
#else
    uint c;
    asm("not.b32 %0, %0;" : "+r"(a));
    c = vavg4(a, b);    // (b + ~a) / 2 = (b - a) / 2 [rounded down]
    c = c & 0x80808080; // msb = carry-outs
    r = c >> 7;         // convert to bool
#endif

    return r;
}

__device__ __forceinline__ uint vcmplt4(uint a, uint b)
{
    uint r, c;

#if CV_CUDEV_ARCH >= 300
    r = vsetlt4(a, b);
    c = r << 8;         // convert bool
    r = c - r;          //  to mask
#else
    asm("not.b32 %0, %0;" : "+r"(a));
    c = vavg4(a, b);    // (b + ~a) / 2 = (b - a) / 2 [rounded down]
    c = c & 0x80808080; // msbs = carry-outs
    r = c >> 7;         // convert
    r = c - r;          //  msbs to
    r = c | r;          //   mask
#endif

    return r;
}

__device__ __forceinline__ uint vsetge4(uint a, uint b)
{
    uint r = 0;

#if CV_CUDEV_ARCH >= 300
    asm("vset4.u32.u32.ge %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
#else
    uint c;
    asm("not.b32 %0, %0;" : "+r"(b));
    c = vavrg4(a, b);   // (a + ~b + 1) / 2 = (a - b) / 2
    c = c & 0x80808080; // msb = carry-outs
    r = c >> 7;         // convert to bool
#endif

    return r;
}

__device__ __forceinline__ uint vcmpge4(uint a, uint b)
{
    uint r, s;

#if CV_CUDEV_ARCH >= 300
    r = vsetge4(a, b);
    s = r << 8;         // convert bool
    r = s - r;          //  to mask
#else
    asm ("not.b32 %0,%0;" : "+r"(b));
    r = vavrg4 (a, b);  // (a + ~b + 1) / 2 = (a - b) / 2
    r = r & 0x80808080; // msb = carry-outs
    s = r >> 7;         // build mask
    s = r - s;          //  from
    r = s | r;          //   msbs
#endif

    return r;
}

__device__ __forceinline__ uint vsetgt4(uint a, uint b)
{
    uint r = 0;

#if CV_CUDEV_ARCH >= 300
    asm("vset4.u32.u32.gt %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
#else
    uint c;
    asm("not.b32 %0, %0;" : "+r"(b));
    c = vavg4(a, b);    // (a + ~b) / 2 = (a - b) / 2 [rounded down]
    c = c & 0x80808080; // msb = carry-outs
    r = c >> 7;         // convert to bool
#endif

    return r;
}

__device__ __forceinline__ uint vcmpgt4(uint a, uint b)
{
    uint r, c;

#if CV_CUDEV_ARCH >= 300
    r = vsetgt4(a, b);
    c = r << 8;         // convert bool
    r = c - r;          //  to mask
#else
    asm("not.b32 %0, %0;" : "+r"(b));
    c = vavg4(a, b);    // (a + ~b) / 2 = (a - b) / 2 [rounded down]
    c = c & 0x80808080; // msb = carry-outs
    r = c >> 7;         // convert
    r = c - r;          //  msbs to
    r = c | r;          //   mask
#endif

    return r;
}

__device__ __forceinline__ uint vsetne4(uint a, uint b)
{
    uint r = 0;

#if CV_CUDEV_ARCH >= 300
    asm("vset4.u32.u32.ne %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
#else
    // inspired by Alan Mycroft's null-byte detection algorithm:
    // null_byte(x) = ((x - 0x01010101) & (~x & 0x80808080))
    uint c;
    r = a ^ b;          // 0x00 if a == b
    c = r | 0x80808080; // set msbs, to catch carry out
    c = c - 0x01010101; // msb = 0, if r was 0x00 or 0x80
    c = r | c;          // msb = 1, if r was not 0x00
    c = c & 0x80808080; // extract msbs
    r = c >> 7;         // convert to bool
#endif

    return r;
}

__device__ __forceinline__ uint vcmpne4(uint a, uint b)
{
    uint r, c;

#if CV_CUDEV_ARCH >= 300
    r = vsetne4(a, b);
    c = r << 8;         // convert bool
    r = c - r;          //  to mask
#else
    // inspired by Alan Mycroft's null-byte detection algorithm:
    // null_byte(x) = ((x - 0x01010101) & (~x & 0x80808080))
    r = a ^ b;          // 0x00 if a == b
    c = r | 0x80808080; // set msbs, to catch carry out
    c = c - 0x01010101; // msb = 0, if r was 0x00 or 0x80
    c = r | c;          // msb = 1, if r was not 0x00
    c = c & 0x80808080; // extract msbs
    r = c >> 7;         // convert
    r = c - r;          //  msbs to
    r = c | r;          //   mask
#endif

    return r;
}

__device__ __forceinline__ uint vabsdiff4(uint a, uint b)
{
    uint r = 0;

#if CV_CUDEV_ARCH >= 300
    asm("vabsdiff4.u32.u32.u32.sat %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
#elif CV_CUDEV_ARCH >= 200
    asm("vabsdiff.u32.u32.u32.sat %0.b0, %1.b0, %2.b0, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
    asm("vabsdiff.u32.u32.u32.sat %0.b1, %1.b1, %2.b1, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
    asm("vabsdiff.u32.u32.u32.sat %0.b2, %1.b2, %2.b2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
    asm("vabsdiff.u32.u32.u32.sat %0.b3, %1.b3, %2.b3, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
#else
    uint s;
    s = vcmpge4(a, b);  // mask = 0xff if a >= b
    r = a ^ b;          //
    s = (r &  s) ^ b;   // select a when a >= b, else select b => max(a,b)
    r = s ^ r;          // select a when b >= a, else select b => min(a,b)
    r = s - r;          // |a - b| = max(a,b) - min(a,b);
#endif

    return r;
}

__device__ __forceinline__ uint vmax4(uint a, uint b)
{
    uint r = 0;

#if CV_CUDEV_ARCH >= 300
    asm("vmax4.u32.u32.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
#elif CV_CUDEV_ARCH >= 200
    asm("vmax.u32.u32.u32 %0.b0, %1.b0, %2.b0, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
    asm("vmax.u32.u32.u32 %0.b1, %1.b1, %2.b1, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
    asm("vmax.u32.u32.u32 %0.b2, %1.b2, %2.b2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
    asm("vmax.u32.u32.u32 %0.b3, %1.b3, %2.b3, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
#else
    uint s;
    s = vcmpge4(a, b);  // mask = 0xff if a >= b
    r = a & s;          // select a when b >= a
    s = b & ~s;         // select b when b < a
    r = r | s;          // combine byte selections
#endif

    return r;           // byte-wise unsigned maximum
}

__device__ __forceinline__ uint vmin4(uint a, uint b)
{
    uint r = 0;

#if CV_CUDEV_ARCH >= 300
    asm("vmin4.u32.u32.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
#elif CV_CUDEV_ARCH >= 200
    asm("vmin.u32.u32.u32 %0.b0, %1.b0, %2.b0, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
    asm("vmin.u32.u32.u32 %0.b1, %1.b1, %2.b1, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
    asm("vmin.u32.u32.u32 %0.b2, %1.b2, %2.b2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
    asm("vmin.u32.u32.u32 %0.b3, %1.b3, %2.b3, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
#else
    uint s;
    s = vcmpge4(b, a);  // mask = 0xff if a >= b
    r = a & s;          // select a when b >= a
    s = b & ~s;         // select b when b < a
    r = r | s;          // combine byte selections
#endif

    return r;
}

//! @}

}}

#endif
