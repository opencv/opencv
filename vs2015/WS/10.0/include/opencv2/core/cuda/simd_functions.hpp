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

#ifndef __OPENCV_CUDA_SIMD_FUNCTIONS_HPP__
#define __OPENCV_CUDA_SIMD_FUNCTIONS_HPP__

#include "common.hpp"

/** @file
 * @deprecated Use @ref cudev instead.
 */

//! @cond IGNORED

namespace cv { namespace cuda { namespace device
{
    // 2

    static __device__ __forceinline__ unsigned int vadd2(unsigned int a, unsigned int b)
    {
        unsigned int r = 0;

    #if __CUDA_ARCH__ >= 300
        asm("vadd2.u32.u32.u32.sat %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
    #elif __CUDA_ARCH__ >= 200
        asm("vadd.u32.u32.u32.sat %0.h0, %1.h0, %2.h0, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
        asm("vadd.u32.u32.u32.sat %0.h1, %1.h1, %2.h1, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
    #else
        unsigned int s;
        s = a ^ b;          // sum bits
        r = a + b;          // actual sum
        s = s ^ r;          // determine carry-ins for each bit position
        s = s & 0x00010000; // carry-in to high word (= carry-out from low word)
        r = r - s;          // subtract out carry-out from low word
    #endif

        return r;
    }

    static __device__ __forceinline__ unsigned int vsub2(unsigned int a, unsigned int b)
    {
        unsigned int r = 0;

    #if __CUDA_ARCH__ >= 300
        asm("vsub2.u32.u32.u32.sat %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
    #elif __CUDA_ARCH__ >= 200
        asm("vsub.u32.u32.u32.sat %0.h0, %1.h0, %2.h0, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
        asm("vsub.u32.u32.u32.sat %0.h1, %1.h1, %2.h1, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
    #else
        unsigned int s;
        s = a ^ b;          // sum bits
        r = a - b;          // actual sum
        s = s ^ r;          // determine carry-ins for each bit position
        s = s & 0x00010000; // borrow to high word
        r = r + s;          // compensate for borrow from low word
    #endif

        return r;
    }

    static __device__ __forceinline__ unsigned int vabsdiff2(unsigned int a, unsigned int b)
    {
        unsigned int r = 0;

    #if __CUDA_ARCH__ >= 300
        asm("vabsdiff2.u32.u32.u32.sat %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
    #elif __CUDA_ARCH__ >= 200
        asm("vabsdiff.u32.u32.u32.sat %0.h0, %1.h0, %2.h0, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
        asm("vabsdiff.u32.u32.u32.sat %0.h1, %1.h1, %2.h1, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
    #else
        unsigned int s, t, u, v;
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

    static __device__ __forceinline__ unsigned int vavg2(unsigned int a, unsigned int b)
    {
        unsigned int r, s;

        // HAKMEM #23: a + b = 2 * (a & b) + (a ^ b) ==>
        // (a + b) / 2 = (a & b) + ((a ^ b) >> 1)
        s = a ^ b;
        r = a & b;
        s = s & 0xfffefffe; // ensure shift doesn't cross halfword boundaries
        s = s >> 1;
        s = r + s;

        return s;
    }

    static __device__ __forceinline__ unsigned int vavrg2(unsigned int a, unsigned int b)
    {
        unsigned int r = 0;

    #if __CUDA_ARCH__ >= 300
        asm("vavrg2.u32.u32.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
    #else
        // HAKMEM #23: a + b = 2 * (a | b) - (a ^ b) ==>
        // (a + b + 1) / 2 = (a | b) - ((a ^ b) >> 1)
        unsigned int s;
        s = a ^ b;
        r = a | b;
        s = s & 0xfffefffe; // ensure shift doesn't cross half-word boundaries
        s = s >> 1;
        r = r - s;
    #endif

        return r;
    }

    static __device__ __forceinline__ unsigned int vseteq2(unsigned int a, unsigned int b)
    {
        unsigned int r = 0;

    #if __CUDA_ARCH__ >= 300
        asm("vset2.u32.u32.eq %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
    #else
        // inspired by Alan Mycroft's null-byte detection algorithm:
        // null_byte(x) = ((x - 0x01010101) & (~x & 0x80808080))
        unsigned int c;
        r = a ^ b;          // 0x0000 if a == b
        c = r | 0x80008000; // set msbs, to catch carry out
        r = r ^ c;          // extract msbs, msb = 1 if r < 0x8000
        c = c - 0x00010001; // msb = 0, if r was 0x0000 or 0x8000
        c = r & ~c;         // msb = 1, if r was 0x0000
        r = c >> 15;        // convert to bool
    #endif

        return r;
    }

    static __device__ __forceinline__ unsigned int vcmpeq2(unsigned int a, unsigned int b)
    {
        unsigned int r, c;

    #if __CUDA_ARCH__ >= 300
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

    static __device__ __forceinline__ unsigned int vsetge2(unsigned int a, unsigned int b)
    {
        unsigned int r = 0;

    #if __CUDA_ARCH__ >= 300
        asm("vset2.u32.u32.ge %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
    #else
        unsigned int c;
        asm("not.b32 %0, %0;" : "+r"(b));
        c = vavrg2(a, b);   // (a + ~b + 1) / 2 = (a - b) / 2
        c = c & 0x80008000; // msb = carry-outs
        r = c >> 15;        // convert to bool
    #endif

        return r;
    }

    static __device__ __forceinline__ unsigned int vcmpge2(unsigned int a, unsigned int b)
    {
        unsigned int r, c;

    #if __CUDA_ARCH__ >= 300
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

    static __device__ __forceinline__ unsigned int vsetgt2(unsigned int a, unsigned int b)
    {
        unsigned int r = 0;

    #if __CUDA_ARCH__ >= 300
        asm("vset2.u32.u32.gt %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
    #else
        unsigned int c;
        asm("not.b32 %0, %0;" : "+r"(b));
        c = vavg2(a, b);    // (a + ~b) / 2 = (a - b) / 2 [rounded down]
        c = c & 0x80008000; // msbs = carry-outs
        r = c >> 15;        // convert to bool
    #endif

        return r;
    }

    static __device__ __forceinline__ unsigned int vcmpgt2(unsigned int a, unsigned int b)
    {
        unsigned int r, c;

    #if __CUDA_ARCH__ >= 300
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

    static __device__ __forceinline__ unsigned int vsetle2(unsigned int a, unsigned int b)
    {
        unsigned int r = 0;

    #if __CUDA_ARCH__ >= 300
        asm("vset2.u32.u32.le %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
    #else
        unsigned int c;
        asm("not.b32 %0, %0;" : "+r"(a));
        c = vavrg2(a, b);   // (b + ~a + 1) / 2 = (b - a) / 2
        c = c & 0x80008000; // msb = carry-outs
        r = c >> 15;        // convert to bool
    #endif

        return r;
    }

    static __device__ __forceinline__ unsigned int vcmple2(unsigned int a, unsigned int b)
    {
        unsigned int r, c;

    #if __CUDA_ARCH__ >= 300
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

    static __device__ __forceinline__ unsigned int vsetlt2(unsigned int a, unsigned int b)
    {
        unsigned int r = 0;

    #if __CUDA_ARCH__ >= 300
        asm("vset2.u32.u32.lt %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
    #else
        unsigned int c;
        asm("not.b32 %0, %0;" : "+r"(a));
        c = vavg2(a, b);    // (b + ~a) / 2 = (b - a) / 2 [rounded down]
        c = c & 0x80008000; // msb = carry-outs
        r = c >> 15;        // convert to bool
    #endif

        return r;
    }

    static __device__ __forceinline__ unsigned int vcmplt2(unsigned int a, unsigned int b)
    {
        unsigned int r, c;

    #if __CUDA_ARCH__ >= 300
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

    static __device__ __forceinline__ unsigned int vsetne2(unsigned int a, unsigned int b)
    {
        unsigned int r = 0;

    #if __CUDA_ARCH__ >= 300
        asm ("vset2.u32.u32.ne %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
    #else
        // inspired by Alan Mycroft's null-byte detection algorithm:
        // null_byte(x) = ((x - 0x01010101) & (~x & 0x80808080))
        unsigned int c;
        r = a ^ b;          // 0x0000 if a == b
        c = r | 0x80008000; // set msbs, to catch carry out
        c = c - 0x00010001; // msb = 0, if r was 0x0000 or 0x8000
        c = r | c;          // msb = 1, if r was not 0x0000
        c = c & 0x80008000; // extract msbs
        r = c >> 15;        // convert to bool
    #endif

        return r;
    }

    static __device__ __forceinline__ unsigned int vcmpne2(unsigned int a, unsigned int b)
    {
        unsigned int r, c;

    #if __CUDA_ARCH__ >= 300
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

    static __device__ __forceinline__ unsigned int vmax2(unsigned int a, unsigned int b)
    {
        unsigned int r = 0;

    #if __CUDA_ARCH__ >= 300
        asm("vmax2.u32.u32.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
    #elif __CUDA_ARCH__ >= 200
        asm("vmax.u32.u32.u32 %0.h0, %1.h0, %2.h0, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
        asm("vmax.u32.u32.u32 %0.h1, %1.h1, %2.h1, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
    #else
        unsigned int s, t, u;
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

    static __device__ __forceinline__ unsigned int vmin2(unsigned int a, unsigned int b)
    {
        unsigned int r = 0;

    #if __CUDA_ARCH__ >= 300
        asm("vmin2.u32.u32.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
    #elif __CUDA_ARCH__ >= 200
        asm("vmin.u32.u32.u32 %0.h0, %1.h0, %2.h0, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
        asm("vmin.u32.u32.u32 %0.h1, %1.h1, %2.h1, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
    #else
        unsigned int s, t, u;
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

    static __device__ __forceinline__ unsigned int vadd4(unsigned int a, unsigned int b)
    {
        unsigned int r = 0;

    #if __CUDA_ARCH__ >= 300
        asm("vadd4.u32.u32.u32.sat %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
    #elif __CUDA_ARCH__ >= 200
        asm("vadd.u32.u32.u32.sat %0.b0, %1.b0, %2.b0, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
        asm("vadd.u32.u32.u32.sat %0.b1, %1.b1, %2.b1, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
        asm("vadd.u32.u32.u32.sat %0.b2, %1.b2, %2.b2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
        asm("vadd.u32.u32.u32.sat %0.b3, %1.b3, %2.b3, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
    #else
        unsigned int s, t;
        s = a ^ b;          // sum bits
        r = a & 0x7f7f7f7f; // clear msbs
        t = b & 0x7f7f7f7f; // clear msbs
        s = s & 0x80808080; // msb sum bits
        r = r + t;          // add without msbs, record carry-out in msbs
        r = r ^ s;          // sum of msb sum and carry-in bits, w/o carry-out
    #endif /* __CUDA_ARCH__ >= 300 */

        return r;
    }

    static __device__ __forceinline__ unsigned int vsub4(unsigned int a, unsigned int b)
    {
        unsigned int r = 0;

    #if __CUDA_ARCH__ >= 300
        asm("vsub4.u32.u32.u32.sat %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
    #elif __CUDA_ARCH__ >= 200
        asm("vsub.u32.u32.u32.sat %0.b0, %1.b0, %2.b0, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
        asm("vsub.u32.u32.u32.sat %0.b1, %1.b1, %2.b1, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
        asm("vsub.u32.u32.u32.sat %0.b2, %1.b2, %2.b2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
        asm("vsub.u32.u32.u32.sat %0.b3, %1.b3, %2.b3, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
    #else
        unsigned int s, t;
        s = a ^ ~b;         // inverted sum bits
        r = a | 0x80808080; // set msbs
        t = b & 0x7f7f7f7f; // clear msbs
        s = s & 0x80808080; // inverted msb sum bits
        r = r - t;          // subtract w/o msbs, record inverted borrows in msb
        r = r ^ s;          // combine inverted msb sum bits and borrows
    #endif

        return r;
    }

    static __device__ __forceinline__ unsigned int vavg4(unsigned int a, unsigned int b)
    {
        unsigned int r, s;

        // HAKMEM #23: a + b = 2 * (a & b) + (a ^ b) ==>
        // (a + b) / 2 = (a & b) + ((a ^ b) >> 1)
        s = a ^ b;
        r = a & b;
        s = s & 0xfefefefe; // ensure following shift doesn't cross byte boundaries
        s = s >> 1;
        s = r + s;

        return s;
    }

    static __device__ __forceinline__ unsigned int vavrg4(unsigned int a, unsigned int b)
    {
        unsigned int r = 0;

    #if __CUDA_ARCH__ >= 300
        asm("vavrg4.u32.u32.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
    #else
        // HAKMEM #23: a + b = 2 * (a | b) - (a ^ b) ==>
        // (a + b + 1) / 2 = (a | b) - ((a ^ b) >> 1)
        unsigned int c;
        c = a ^ b;
        r = a | b;
        c = c & 0xfefefefe; // ensure following shift doesn't cross byte boundaries
        c = c >> 1;
        r = r - c;
    #endif

        return r;
    }

    static __device__ __forceinline__ unsigned int vseteq4(unsigned int a, unsigned int b)
    {
        unsigned int r = 0;

    #if __CUDA_ARCH__ >= 300
        asm("vset4.u32.u32.eq %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
    #else
        // inspired by Alan Mycroft's null-byte detection algorithm:
        // null_byte(x) = ((x - 0x01010101) & (~x & 0x80808080))
        unsigned int c;
        r = a ^ b;          // 0x00 if a == b
        c = r | 0x80808080; // set msbs, to catch carry out
        r = r ^ c;          // extract msbs, msb = 1 if r < 0x80
        c = c - 0x01010101; // msb = 0, if r was 0x00 or 0x80
        c = r & ~c;         // msb = 1, if r was 0x00
        r = c >> 7;         // convert to bool
    #endif

        return r;
    }

    static __device__ __forceinline__ unsigned int vcmpeq4(unsigned int a, unsigned int b)
    {
        unsigned int r, t;

    #if __CUDA_ARCH__ >= 300
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

    static __device__ __forceinline__ unsigned int vsetle4(unsigned int a, unsigned int b)
    {
        unsigned int r = 0;

    #if __CUDA_ARCH__ >= 300
        asm("vset4.u32.u32.le %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
    #else
        unsigned int c;
        asm("not.b32 %0, %0;" : "+r"(a));
        c = vavrg4(a, b);   // (b + ~a + 1) / 2 = (b - a) / 2
        c = c & 0x80808080; // msb = carry-outs
        r = c >> 7;         // convert to bool
    #endif

        return r;
    }

    static __device__ __forceinline__ unsigned int vcmple4(unsigned int a, unsigned int b)
    {
        unsigned int r, c;

    #if __CUDA_ARCH__ >= 300
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

    static __device__ __forceinline__ unsigned int vsetlt4(unsigned int a, unsigned int b)
    {
        unsigned int r = 0;

    #if __CUDA_ARCH__ >= 300
        asm("vset4.u32.u32.lt %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
    #else
        unsigned int c;
        asm("not.b32 %0, %0;" : "+r"(a));
        c = vavg4(a, b);    // (b + ~a) / 2 = (b - a) / 2 [rounded down]
        c = c & 0x80808080; // msb = carry-outs
        r = c >> 7;         // convert to bool
    #endif

        return r;
    }

    static __device__ __forceinline__ unsigned int vcmplt4(unsigned int a, unsigned int b)
    {
        unsigned int r, c;

    #if __CUDA_ARCH__ >= 300
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

    static __device__ __forceinline__ unsigned int vsetge4(unsigned int a, unsigned int b)
    {
        unsigned int r = 0;

    #if __CUDA_ARCH__ >= 300
        asm("vset4.u32.u32.ge %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
    #else
        unsigned int c;
        asm("not.b32 %0, %0;" : "+r"(b));
        c = vavrg4(a, b);   // (a + ~b + 1) / 2 = (a - b) / 2
        c = c & 0x80808080; // msb = carry-outs
        r = c >> 7;         // convert to bool
    #endif

        return r;
    }

    static __device__ __forceinline__ unsigned int vcmpge4(unsigned int a, unsigned int b)
    {
        unsigned int r, s;

    #if __CUDA_ARCH__ >= 300
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

    static __device__ __forceinline__ unsigned int vsetgt4(unsigned int a, unsigned int b)
    {
        unsigned int r = 0;

    #if __CUDA_ARCH__ >= 300
        asm("vset4.u32.u32.gt %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
    #else
        unsigned int c;
        asm("not.b32 %0, %0;" : "+r"(b));
        c = vavg4(a, b);    // (a + ~b) / 2 = (a - b) / 2 [rounded down]
        c = c & 0x80808080; // msb = carry-outs
        r = c >> 7;         // convert to bool
    #endif

        return r;
    }

    static __device__ __forceinline__ unsigned int vcmpgt4(unsigned int a, unsigned int b)
    {
        unsigned int r, c;

    #if __CUDA_ARCH__ >= 300
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

    static __device__ __forceinline__ unsigned int vsetne4(unsigned int a, unsigned int b)
    {
        unsigned int r = 0;

    #if __CUDA_ARCH__ >= 300
        asm("vset4.u32.u32.ne %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
    #else
        // inspired by Alan Mycroft's null-byte detection algorithm:
        // null_byte(x) = ((x - 0x01010101) & (~x & 0x80808080))
        unsigned int c;
        r = a ^ b;          // 0x00 if a == b
        c = r | 0x80808080; // set msbs, to catch carry out
        c = c - 0x01010101; // msb = 0, if r was 0x00 or 0x80
        c = r | c;          // msb = 1, if r was not 0x00
        c = c & 0x80808080; // extract msbs
        r = c >> 7;         // convert to bool
    #endif

        return r;
    }

    static __device__ __forceinline__ unsigned int vcmpne4(unsigned int a, unsigned int b)
    {
        unsigned int r, c;

    #if __CUDA_ARCH__ >= 300
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

    static __device__ __forceinline__ unsigned int vabsdiff4(unsigned int a, unsigned int b)
    {
        unsigned int r = 0;

    #if __CUDA_ARCH__ >= 300
        asm("vabsdiff4.u32.u32.u32.sat %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
    #elif __CUDA_ARCH__ >= 200
        asm("vabsdiff.u32.u32.u32.sat %0.b0, %1.b0, %2.b0, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
        asm("vabsdiff.u32.u32.u32.sat %0.b1, %1.b1, %2.b1, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
        asm("vabsdiff.u32.u32.u32.sat %0.b2, %1.b2, %2.b2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
        asm("vabsdiff.u32.u32.u32.sat %0.b3, %1.b3, %2.b3, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
    #else
        unsigned int s;
        s = vcmpge4(a, b);  // mask = 0xff if a >= b
        r = a ^ b;          //
        s = (r &  s) ^ b;   // select a when a >= b, else select b => max(a,b)
        r = s ^ r;          // select a when b >= a, else select b => min(a,b)
        r = s - r;          // |a - b| = max(a,b) - min(a,b);
    #endif

        return r;
    }

    static __device__ __forceinline__ unsigned int vmax4(unsigned int a, unsigned int b)
    {
        unsigned int r = 0;

    #if __CUDA_ARCH__ >= 300
        asm("vmax4.u32.u32.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
    #elif __CUDA_ARCH__ >= 200
        asm("vmax.u32.u32.u32 %0.b0, %1.b0, %2.b0, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
        asm("vmax.u32.u32.u32 %0.b1, %1.b1, %2.b1, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
        asm("vmax.u32.u32.u32 %0.b2, %1.b2, %2.b2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
        asm("vmax.u32.u32.u32 %0.b3, %1.b3, %2.b3, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
    #else
        unsigned int s;
        s = vcmpge4(a, b);  // mask = 0xff if a >= b
        r = a & s;          // select a when b >= a
        s = b & ~s;         // select b when b < a
        r = r | s;          // combine byte selections
    #endif

        return r;           // byte-wise unsigned maximum
    }

    static __device__ __forceinline__ unsigned int vmin4(unsigned int a, unsigned int b)
    {
        unsigned int r = 0;

    #if __CUDA_ARCH__ >= 300
        asm("vmin4.u32.u32.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
    #elif __CUDA_ARCH__ >= 200
        asm("vmin.u32.u32.u32 %0.b0, %1.b0, %2.b0, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
        asm("vmin.u32.u32.u32 %0.b1, %1.b1, %2.b1, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
        asm("vmin.u32.u32.u32 %0.b2, %1.b2, %2.b2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
        asm("vmin.u32.u32.u32 %0.b3, %1.b3, %2.b3, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(r));
    #else
        unsigned int s;
        s = vcmpge4(b, a);  // mask = 0xff if a >= b
        r = a & s;          // select a when b >= a
        s = b & ~s;         // select b when b < a
        r = r | s;          // combine byte selections
    #endif

        return r;
    }
}}}

//! @endcond

#endif // __OPENCV_CUDA_SIMD_FUNCTIONS_HPP__
