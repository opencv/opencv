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
// Copyright (C) 2015, Itseez Inc., all rights reserved.
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

#ifndef OPENCV_CORE_SSE_UTILS_HPP
#define OPENCV_CORE_SSE_UTILS_HPP

#ifndef __cplusplus
#  error sse_utils.hpp header must be compiled as C++
#endif

#include "opencv2/core/cvdef.h"

//! @addtogroup core_utils_sse
//! @{

#if CV_SSE2

inline void _mm_deinterleave_epi8(__m128i & v_r0, __m128i & v_r1, __m128i & v_g0, __m128i & v_g1)
{
    __m128i layer1_chunk0 = _mm_unpacklo_epi8(v_r0, v_g0);
    __m128i layer1_chunk1 = _mm_unpackhi_epi8(v_r0, v_g0);
    __m128i layer1_chunk2 = _mm_unpacklo_epi8(v_r1, v_g1);
    __m128i layer1_chunk3 = _mm_unpackhi_epi8(v_r1, v_g1);

    __m128i layer2_chunk0 = _mm_unpacklo_epi8(layer1_chunk0, layer1_chunk2);
    __m128i layer2_chunk1 = _mm_unpackhi_epi8(layer1_chunk0, layer1_chunk2);
    __m128i layer2_chunk2 = _mm_unpacklo_epi8(layer1_chunk1, layer1_chunk3);
    __m128i layer2_chunk3 = _mm_unpackhi_epi8(layer1_chunk1, layer1_chunk3);

    __m128i layer3_chunk0 = _mm_unpacklo_epi8(layer2_chunk0, layer2_chunk2);
    __m128i layer3_chunk1 = _mm_unpackhi_epi8(layer2_chunk0, layer2_chunk2);
    __m128i layer3_chunk2 = _mm_unpacklo_epi8(layer2_chunk1, layer2_chunk3);
    __m128i layer3_chunk3 = _mm_unpackhi_epi8(layer2_chunk1, layer2_chunk3);

    __m128i layer4_chunk0 = _mm_unpacklo_epi8(layer3_chunk0, layer3_chunk2);
    __m128i layer4_chunk1 = _mm_unpackhi_epi8(layer3_chunk0, layer3_chunk2);
    __m128i layer4_chunk2 = _mm_unpacklo_epi8(layer3_chunk1, layer3_chunk3);
    __m128i layer4_chunk3 = _mm_unpackhi_epi8(layer3_chunk1, layer3_chunk3);

    v_r0 = _mm_unpacklo_epi8(layer4_chunk0, layer4_chunk2);
    v_r1 = _mm_unpackhi_epi8(layer4_chunk0, layer4_chunk2);
    v_g0 = _mm_unpacklo_epi8(layer4_chunk1, layer4_chunk3);
    v_g1 = _mm_unpackhi_epi8(layer4_chunk1, layer4_chunk3);
}

inline void _mm_deinterleave_epi8(__m128i & v_r0, __m128i & v_r1, __m128i & v_g0,
                                  __m128i & v_g1, __m128i & v_b0, __m128i & v_b1)
{
    __m128i layer1_chunk0 = _mm_unpacklo_epi8(v_r0, v_g1);
    __m128i layer1_chunk1 = _mm_unpackhi_epi8(v_r0, v_g1);
    __m128i layer1_chunk2 = _mm_unpacklo_epi8(v_r1, v_b0);
    __m128i layer1_chunk3 = _mm_unpackhi_epi8(v_r1, v_b0);
    __m128i layer1_chunk4 = _mm_unpacklo_epi8(v_g0, v_b1);
    __m128i layer1_chunk5 = _mm_unpackhi_epi8(v_g0, v_b1);

    __m128i layer2_chunk0 = _mm_unpacklo_epi8(layer1_chunk0, layer1_chunk3);
    __m128i layer2_chunk1 = _mm_unpackhi_epi8(layer1_chunk0, layer1_chunk3);
    __m128i layer2_chunk2 = _mm_unpacklo_epi8(layer1_chunk1, layer1_chunk4);
    __m128i layer2_chunk3 = _mm_unpackhi_epi8(layer1_chunk1, layer1_chunk4);
    __m128i layer2_chunk4 = _mm_unpacklo_epi8(layer1_chunk2, layer1_chunk5);
    __m128i layer2_chunk5 = _mm_unpackhi_epi8(layer1_chunk2, layer1_chunk5);

    __m128i layer3_chunk0 = _mm_unpacklo_epi8(layer2_chunk0, layer2_chunk3);
    __m128i layer3_chunk1 = _mm_unpackhi_epi8(layer2_chunk0, layer2_chunk3);
    __m128i layer3_chunk2 = _mm_unpacklo_epi8(layer2_chunk1, layer2_chunk4);
    __m128i layer3_chunk3 = _mm_unpackhi_epi8(layer2_chunk1, layer2_chunk4);
    __m128i layer3_chunk4 = _mm_unpacklo_epi8(layer2_chunk2, layer2_chunk5);
    __m128i layer3_chunk5 = _mm_unpackhi_epi8(layer2_chunk2, layer2_chunk5);

    __m128i layer4_chunk0 = _mm_unpacklo_epi8(layer3_chunk0, layer3_chunk3);
    __m128i layer4_chunk1 = _mm_unpackhi_epi8(layer3_chunk0, layer3_chunk3);
    __m128i layer4_chunk2 = _mm_unpacklo_epi8(layer3_chunk1, layer3_chunk4);
    __m128i layer4_chunk3 = _mm_unpackhi_epi8(layer3_chunk1, layer3_chunk4);
    __m128i layer4_chunk4 = _mm_unpacklo_epi8(layer3_chunk2, layer3_chunk5);
    __m128i layer4_chunk5 = _mm_unpackhi_epi8(layer3_chunk2, layer3_chunk5);

    v_r0 = _mm_unpacklo_epi8(layer4_chunk0, layer4_chunk3);
    v_r1 = _mm_unpackhi_epi8(layer4_chunk0, layer4_chunk3);
    v_g0 = _mm_unpacklo_epi8(layer4_chunk1, layer4_chunk4);
    v_g1 = _mm_unpackhi_epi8(layer4_chunk1, layer4_chunk4);
    v_b0 = _mm_unpacklo_epi8(layer4_chunk2, layer4_chunk5);
    v_b1 = _mm_unpackhi_epi8(layer4_chunk2, layer4_chunk5);
}

inline void _mm_deinterleave_epi8(__m128i & v_r0, __m128i & v_r1, __m128i & v_g0, __m128i & v_g1,
                                  __m128i & v_b0, __m128i & v_b1, __m128i & v_a0, __m128i & v_a1)
{
    __m128i layer1_chunk0 = _mm_unpacklo_epi8(v_r0, v_b0);
    __m128i layer1_chunk1 = _mm_unpackhi_epi8(v_r0, v_b0);
    __m128i layer1_chunk2 = _mm_unpacklo_epi8(v_r1, v_b1);
    __m128i layer1_chunk3 = _mm_unpackhi_epi8(v_r1, v_b1);
    __m128i layer1_chunk4 = _mm_unpacklo_epi8(v_g0, v_a0);
    __m128i layer1_chunk5 = _mm_unpackhi_epi8(v_g0, v_a0);
    __m128i layer1_chunk6 = _mm_unpacklo_epi8(v_g1, v_a1);
    __m128i layer1_chunk7 = _mm_unpackhi_epi8(v_g1, v_a1);

    __m128i layer2_chunk0 = _mm_unpacklo_epi8(layer1_chunk0, layer1_chunk4);
    __m128i layer2_chunk1 = _mm_unpackhi_epi8(layer1_chunk0, layer1_chunk4);
    __m128i layer2_chunk2 = _mm_unpacklo_epi8(layer1_chunk1, layer1_chunk5);
    __m128i layer2_chunk3 = _mm_unpackhi_epi8(layer1_chunk1, layer1_chunk5);
    __m128i layer2_chunk4 = _mm_unpacklo_epi8(layer1_chunk2, layer1_chunk6);
    __m128i layer2_chunk5 = _mm_unpackhi_epi8(layer1_chunk2, layer1_chunk6);
    __m128i layer2_chunk6 = _mm_unpacklo_epi8(layer1_chunk3, layer1_chunk7);
    __m128i layer2_chunk7 = _mm_unpackhi_epi8(layer1_chunk3, layer1_chunk7);

    __m128i layer3_chunk0 = _mm_unpacklo_epi8(layer2_chunk0, layer2_chunk4);
    __m128i layer3_chunk1 = _mm_unpackhi_epi8(layer2_chunk0, layer2_chunk4);
    __m128i layer3_chunk2 = _mm_unpacklo_epi8(layer2_chunk1, layer2_chunk5);
    __m128i layer3_chunk3 = _mm_unpackhi_epi8(layer2_chunk1, layer2_chunk5);
    __m128i layer3_chunk4 = _mm_unpacklo_epi8(layer2_chunk2, layer2_chunk6);
    __m128i layer3_chunk5 = _mm_unpackhi_epi8(layer2_chunk2, layer2_chunk6);
    __m128i layer3_chunk6 = _mm_unpacklo_epi8(layer2_chunk3, layer2_chunk7);
    __m128i layer3_chunk7 = _mm_unpackhi_epi8(layer2_chunk3, layer2_chunk7);

    __m128i layer4_chunk0 = _mm_unpacklo_epi8(layer3_chunk0, layer3_chunk4);
    __m128i layer4_chunk1 = _mm_unpackhi_epi8(layer3_chunk0, layer3_chunk4);
    __m128i layer4_chunk2 = _mm_unpacklo_epi8(layer3_chunk1, layer3_chunk5);
    __m128i layer4_chunk3 = _mm_unpackhi_epi8(layer3_chunk1, layer3_chunk5);
    __m128i layer4_chunk4 = _mm_unpacklo_epi8(layer3_chunk2, layer3_chunk6);
    __m128i layer4_chunk5 = _mm_unpackhi_epi8(layer3_chunk2, layer3_chunk6);
    __m128i layer4_chunk6 = _mm_unpacklo_epi8(layer3_chunk3, layer3_chunk7);
    __m128i layer4_chunk7 = _mm_unpackhi_epi8(layer3_chunk3, layer3_chunk7);

    v_r0 = _mm_unpacklo_epi8(layer4_chunk0, layer4_chunk4);
    v_r1 = _mm_unpackhi_epi8(layer4_chunk0, layer4_chunk4);
    v_g0 = _mm_unpacklo_epi8(layer4_chunk1, layer4_chunk5);
    v_g1 = _mm_unpackhi_epi8(layer4_chunk1, layer4_chunk5);
    v_b0 = _mm_unpacklo_epi8(layer4_chunk2, layer4_chunk6);
    v_b1 = _mm_unpackhi_epi8(layer4_chunk2, layer4_chunk6);
    v_a0 = _mm_unpacklo_epi8(layer4_chunk3, layer4_chunk7);
    v_a1 = _mm_unpackhi_epi8(layer4_chunk3, layer4_chunk7);
}

inline void _mm_interleave_epi8(__m128i & v_r0, __m128i & v_r1, __m128i & v_g0, __m128i & v_g1)
{
    __m128i v_mask = _mm_set1_epi16(0x00ff);

    __m128i layer4_chunk0 = _mm_packus_epi16(_mm_and_si128(v_r0, v_mask), _mm_and_si128(v_r1, v_mask));
    __m128i layer4_chunk2 = _mm_packus_epi16(_mm_srli_epi16(v_r0, 8), _mm_srli_epi16(v_r1, 8));
    __m128i layer4_chunk1 = _mm_packus_epi16(_mm_and_si128(v_g0, v_mask), _mm_and_si128(v_g1, v_mask));
    __m128i layer4_chunk3 = _mm_packus_epi16(_mm_srli_epi16(v_g0, 8), _mm_srli_epi16(v_g1, 8));

    __m128i layer3_chunk0 = _mm_packus_epi16(_mm_and_si128(layer4_chunk0, v_mask), _mm_and_si128(layer4_chunk1, v_mask));
    __m128i layer3_chunk2 = _mm_packus_epi16(_mm_srli_epi16(layer4_chunk0, 8), _mm_srli_epi16(layer4_chunk1, 8));
    __m128i layer3_chunk1 = _mm_packus_epi16(_mm_and_si128(layer4_chunk2, v_mask), _mm_and_si128(layer4_chunk3, v_mask));
    __m128i layer3_chunk3 = _mm_packus_epi16(_mm_srli_epi16(layer4_chunk2, 8), _mm_srli_epi16(layer4_chunk3, 8));

    __m128i layer2_chunk0 = _mm_packus_epi16(_mm_and_si128(layer3_chunk0, v_mask), _mm_and_si128(layer3_chunk1, v_mask));
    __m128i layer2_chunk2 = _mm_packus_epi16(_mm_srli_epi16(layer3_chunk0, 8), _mm_srli_epi16(layer3_chunk1, 8));
    __m128i layer2_chunk1 = _mm_packus_epi16(_mm_and_si128(layer3_chunk2, v_mask), _mm_and_si128(layer3_chunk3, v_mask));
    __m128i layer2_chunk3 = _mm_packus_epi16(_mm_srli_epi16(layer3_chunk2, 8), _mm_srli_epi16(layer3_chunk3, 8));

    __m128i layer1_chunk0 = _mm_packus_epi16(_mm_and_si128(layer2_chunk0, v_mask), _mm_and_si128(layer2_chunk1, v_mask));
    __m128i layer1_chunk2 = _mm_packus_epi16(_mm_srli_epi16(layer2_chunk0, 8), _mm_srli_epi16(layer2_chunk1, 8));
    __m128i layer1_chunk1 = _mm_packus_epi16(_mm_and_si128(layer2_chunk2, v_mask), _mm_and_si128(layer2_chunk3, v_mask));
    __m128i layer1_chunk3 = _mm_packus_epi16(_mm_srli_epi16(layer2_chunk2, 8), _mm_srli_epi16(layer2_chunk3, 8));

    v_r0 = _mm_packus_epi16(_mm_and_si128(layer1_chunk0, v_mask), _mm_and_si128(layer1_chunk1, v_mask));
    v_g0 = _mm_packus_epi16(_mm_srli_epi16(layer1_chunk0, 8), _mm_srli_epi16(layer1_chunk1, 8));
    v_r1 = _mm_packus_epi16(_mm_and_si128(layer1_chunk2, v_mask), _mm_and_si128(layer1_chunk3, v_mask));
    v_g1 = _mm_packus_epi16(_mm_srli_epi16(layer1_chunk2, 8), _mm_srli_epi16(layer1_chunk3, 8));
}

inline void _mm_interleave_epi8(__m128i & v_r0, __m128i & v_r1, __m128i & v_g0,
                                __m128i & v_g1, __m128i & v_b0, __m128i & v_b1)
{
    __m128i v_mask = _mm_set1_epi16(0x00ff);

    __m128i layer4_chunk0 = _mm_packus_epi16(_mm_and_si128(v_r0, v_mask), _mm_and_si128(v_r1, v_mask));
    __m128i layer4_chunk3 = _mm_packus_epi16(_mm_srli_epi16(v_r0, 8), _mm_srli_epi16(v_r1, 8));
    __m128i layer4_chunk1 = _mm_packus_epi16(_mm_and_si128(v_g0, v_mask), _mm_and_si128(v_g1, v_mask));
    __m128i layer4_chunk4 = _mm_packus_epi16(_mm_srli_epi16(v_g0, 8), _mm_srli_epi16(v_g1, 8));
    __m128i layer4_chunk2 = _mm_packus_epi16(_mm_and_si128(v_b0, v_mask), _mm_and_si128(v_b1, v_mask));
    __m128i layer4_chunk5 = _mm_packus_epi16(_mm_srli_epi16(v_b0, 8), _mm_srli_epi16(v_b1, 8));

    __m128i layer3_chunk0 = _mm_packus_epi16(_mm_and_si128(layer4_chunk0, v_mask), _mm_and_si128(layer4_chunk1, v_mask));
    __m128i layer3_chunk3 = _mm_packus_epi16(_mm_srli_epi16(layer4_chunk0, 8), _mm_srli_epi16(layer4_chunk1, 8));
    __m128i layer3_chunk1 = _mm_packus_epi16(_mm_and_si128(layer4_chunk2, v_mask), _mm_and_si128(layer4_chunk3, v_mask));
    __m128i layer3_chunk4 = _mm_packus_epi16(_mm_srli_epi16(layer4_chunk2, 8), _mm_srli_epi16(layer4_chunk3, 8));
    __m128i layer3_chunk2 = _mm_packus_epi16(_mm_and_si128(layer4_chunk4, v_mask), _mm_and_si128(layer4_chunk5, v_mask));
    __m128i layer3_chunk5 = _mm_packus_epi16(_mm_srli_epi16(layer4_chunk4, 8), _mm_srli_epi16(layer4_chunk5, 8));

    __m128i layer2_chunk0 = _mm_packus_epi16(_mm_and_si128(layer3_chunk0, v_mask), _mm_and_si128(layer3_chunk1, v_mask));
    __m128i layer2_chunk3 = _mm_packus_epi16(_mm_srli_epi16(layer3_chunk0, 8), _mm_srli_epi16(layer3_chunk1, 8));
    __m128i layer2_chunk1 = _mm_packus_epi16(_mm_and_si128(layer3_chunk2, v_mask), _mm_and_si128(layer3_chunk3, v_mask));
    __m128i layer2_chunk4 = _mm_packus_epi16(_mm_srli_epi16(layer3_chunk2, 8), _mm_srli_epi16(layer3_chunk3, 8));
    __m128i layer2_chunk2 = _mm_packus_epi16(_mm_and_si128(layer3_chunk4, v_mask), _mm_and_si128(layer3_chunk5, v_mask));
    __m128i layer2_chunk5 = _mm_packus_epi16(_mm_srli_epi16(layer3_chunk4, 8), _mm_srli_epi16(layer3_chunk5, 8));

    __m128i layer1_chunk0 = _mm_packus_epi16(_mm_and_si128(layer2_chunk0, v_mask), _mm_and_si128(layer2_chunk1, v_mask));
    __m128i layer1_chunk3 = _mm_packus_epi16(_mm_srli_epi16(layer2_chunk0, 8), _mm_srli_epi16(layer2_chunk1, 8));
    __m128i layer1_chunk1 = _mm_packus_epi16(_mm_and_si128(layer2_chunk2, v_mask), _mm_and_si128(layer2_chunk3, v_mask));
    __m128i layer1_chunk4 = _mm_packus_epi16(_mm_srli_epi16(layer2_chunk2, 8), _mm_srli_epi16(layer2_chunk3, 8));
    __m128i layer1_chunk2 = _mm_packus_epi16(_mm_and_si128(layer2_chunk4, v_mask), _mm_and_si128(layer2_chunk5, v_mask));
    __m128i layer1_chunk5 = _mm_packus_epi16(_mm_srli_epi16(layer2_chunk4, 8), _mm_srli_epi16(layer2_chunk5, 8));

    v_r0 = _mm_packus_epi16(_mm_and_si128(layer1_chunk0, v_mask), _mm_and_si128(layer1_chunk1, v_mask));
    v_g1 = _mm_packus_epi16(_mm_srli_epi16(layer1_chunk0, 8), _mm_srli_epi16(layer1_chunk1, 8));
    v_r1 = _mm_packus_epi16(_mm_and_si128(layer1_chunk2, v_mask), _mm_and_si128(layer1_chunk3, v_mask));
    v_b0 = _mm_packus_epi16(_mm_srli_epi16(layer1_chunk2, 8), _mm_srli_epi16(layer1_chunk3, 8));
    v_g0 = _mm_packus_epi16(_mm_and_si128(layer1_chunk4, v_mask), _mm_and_si128(layer1_chunk5, v_mask));
    v_b1 = _mm_packus_epi16(_mm_srli_epi16(layer1_chunk4, 8), _mm_srli_epi16(layer1_chunk5, 8));
}

inline void _mm_interleave_epi8(__m128i & v_r0, __m128i & v_r1, __m128i & v_g0, __m128i & v_g1,
                                __m128i & v_b0, __m128i & v_b1, __m128i & v_a0, __m128i & v_a1)
{
    __m128i v_mask = _mm_set1_epi16(0x00ff);

    __m128i layer4_chunk0 = _mm_packus_epi16(_mm_and_si128(v_r0, v_mask), _mm_and_si128(v_r1, v_mask));
    __m128i layer4_chunk4 = _mm_packus_epi16(_mm_srli_epi16(v_r0, 8), _mm_srli_epi16(v_r1, 8));
    __m128i layer4_chunk1 = _mm_packus_epi16(_mm_and_si128(v_g0, v_mask), _mm_and_si128(v_g1, v_mask));
    __m128i layer4_chunk5 = _mm_packus_epi16(_mm_srli_epi16(v_g0, 8), _mm_srli_epi16(v_g1, 8));
    __m128i layer4_chunk2 = _mm_packus_epi16(_mm_and_si128(v_b0, v_mask), _mm_and_si128(v_b1, v_mask));
    __m128i layer4_chunk6 = _mm_packus_epi16(_mm_srli_epi16(v_b0, 8), _mm_srli_epi16(v_b1, 8));
    __m128i layer4_chunk3 = _mm_packus_epi16(_mm_and_si128(v_a0, v_mask), _mm_and_si128(v_a1, v_mask));
    __m128i layer4_chunk7 = _mm_packus_epi16(_mm_srli_epi16(v_a0, 8), _mm_srli_epi16(v_a1, 8));

    __m128i layer3_chunk0 = _mm_packus_epi16(_mm_and_si128(layer4_chunk0, v_mask), _mm_and_si128(layer4_chunk1, v_mask));
    __m128i layer3_chunk4 = _mm_packus_epi16(_mm_srli_epi16(layer4_chunk0, 8), _mm_srli_epi16(layer4_chunk1, 8));
    __m128i layer3_chunk1 = _mm_packus_epi16(_mm_and_si128(layer4_chunk2, v_mask), _mm_and_si128(layer4_chunk3, v_mask));
    __m128i layer3_chunk5 = _mm_packus_epi16(_mm_srli_epi16(layer4_chunk2, 8), _mm_srli_epi16(layer4_chunk3, 8));
    __m128i layer3_chunk2 = _mm_packus_epi16(_mm_and_si128(layer4_chunk4, v_mask), _mm_and_si128(layer4_chunk5, v_mask));
    __m128i layer3_chunk6 = _mm_packus_epi16(_mm_srli_epi16(layer4_chunk4, 8), _mm_srli_epi16(layer4_chunk5, 8));
    __m128i layer3_chunk3 = _mm_packus_epi16(_mm_and_si128(layer4_chunk6, v_mask), _mm_and_si128(layer4_chunk7, v_mask));
    __m128i layer3_chunk7 = _mm_packus_epi16(_mm_srli_epi16(layer4_chunk6, 8), _mm_srli_epi16(layer4_chunk7, 8));

    __m128i layer2_chunk0 = _mm_packus_epi16(_mm_and_si128(layer3_chunk0, v_mask), _mm_and_si128(layer3_chunk1, v_mask));
    __m128i layer2_chunk4 = _mm_packus_epi16(_mm_srli_epi16(layer3_chunk0, 8), _mm_srli_epi16(layer3_chunk1, 8));
    __m128i layer2_chunk1 = _mm_packus_epi16(_mm_and_si128(layer3_chunk2, v_mask), _mm_and_si128(layer3_chunk3, v_mask));
    __m128i layer2_chunk5 = _mm_packus_epi16(_mm_srli_epi16(layer3_chunk2, 8), _mm_srli_epi16(layer3_chunk3, 8));
    __m128i layer2_chunk2 = _mm_packus_epi16(_mm_and_si128(layer3_chunk4, v_mask), _mm_and_si128(layer3_chunk5, v_mask));
    __m128i layer2_chunk6 = _mm_packus_epi16(_mm_srli_epi16(layer3_chunk4, 8), _mm_srli_epi16(layer3_chunk5, 8));
    __m128i layer2_chunk3 = _mm_packus_epi16(_mm_and_si128(layer3_chunk6, v_mask), _mm_and_si128(layer3_chunk7, v_mask));
    __m128i layer2_chunk7 = _mm_packus_epi16(_mm_srli_epi16(layer3_chunk6, 8), _mm_srli_epi16(layer3_chunk7, 8));

    __m128i layer1_chunk0 = _mm_packus_epi16(_mm_and_si128(layer2_chunk0, v_mask), _mm_and_si128(layer2_chunk1, v_mask));
    __m128i layer1_chunk4 = _mm_packus_epi16(_mm_srli_epi16(layer2_chunk0, 8), _mm_srli_epi16(layer2_chunk1, 8));
    __m128i layer1_chunk1 = _mm_packus_epi16(_mm_and_si128(layer2_chunk2, v_mask), _mm_and_si128(layer2_chunk3, v_mask));
    __m128i layer1_chunk5 = _mm_packus_epi16(_mm_srli_epi16(layer2_chunk2, 8), _mm_srli_epi16(layer2_chunk3, 8));
    __m128i layer1_chunk2 = _mm_packus_epi16(_mm_and_si128(layer2_chunk4, v_mask), _mm_and_si128(layer2_chunk5, v_mask));
    __m128i layer1_chunk6 = _mm_packus_epi16(_mm_srli_epi16(layer2_chunk4, 8), _mm_srli_epi16(layer2_chunk5, 8));
    __m128i layer1_chunk3 = _mm_packus_epi16(_mm_and_si128(layer2_chunk6, v_mask), _mm_and_si128(layer2_chunk7, v_mask));
    __m128i layer1_chunk7 = _mm_packus_epi16(_mm_srli_epi16(layer2_chunk6, 8), _mm_srli_epi16(layer2_chunk7, 8));

    v_r0 = _mm_packus_epi16(_mm_and_si128(layer1_chunk0, v_mask), _mm_and_si128(layer1_chunk1, v_mask));
    v_b0 = _mm_packus_epi16(_mm_srli_epi16(layer1_chunk0, 8), _mm_srli_epi16(layer1_chunk1, 8));
    v_r1 = _mm_packus_epi16(_mm_and_si128(layer1_chunk2, v_mask), _mm_and_si128(layer1_chunk3, v_mask));
    v_b1 = _mm_packus_epi16(_mm_srli_epi16(layer1_chunk2, 8), _mm_srli_epi16(layer1_chunk3, 8));
    v_g0 = _mm_packus_epi16(_mm_and_si128(layer1_chunk4, v_mask), _mm_and_si128(layer1_chunk5, v_mask));
    v_a0 = _mm_packus_epi16(_mm_srli_epi16(layer1_chunk4, 8), _mm_srli_epi16(layer1_chunk5, 8));
    v_g1 = _mm_packus_epi16(_mm_and_si128(layer1_chunk6, v_mask), _mm_and_si128(layer1_chunk7, v_mask));
    v_a1 = _mm_packus_epi16(_mm_srli_epi16(layer1_chunk6, 8), _mm_srli_epi16(layer1_chunk7, 8));
}

inline void _mm_deinterleave_epi16(__m128i & v_r0, __m128i & v_r1, __m128i & v_g0, __m128i & v_g1)
{
    __m128i layer1_chunk0 = _mm_unpacklo_epi16(v_r0, v_g0);
    __m128i layer1_chunk1 = _mm_unpackhi_epi16(v_r0, v_g0);
    __m128i layer1_chunk2 = _mm_unpacklo_epi16(v_r1, v_g1);
    __m128i layer1_chunk3 = _mm_unpackhi_epi16(v_r1, v_g1);

    __m128i layer2_chunk0 = _mm_unpacklo_epi16(layer1_chunk0, layer1_chunk2);
    __m128i layer2_chunk1 = _mm_unpackhi_epi16(layer1_chunk0, layer1_chunk2);
    __m128i layer2_chunk2 = _mm_unpacklo_epi16(layer1_chunk1, layer1_chunk3);
    __m128i layer2_chunk3 = _mm_unpackhi_epi16(layer1_chunk1, layer1_chunk3);

    __m128i layer3_chunk0 = _mm_unpacklo_epi16(layer2_chunk0, layer2_chunk2);
    __m128i layer3_chunk1 = _mm_unpackhi_epi16(layer2_chunk0, layer2_chunk2);
    __m128i layer3_chunk2 = _mm_unpacklo_epi16(layer2_chunk1, layer2_chunk3);
    __m128i layer3_chunk3 = _mm_unpackhi_epi16(layer2_chunk1, layer2_chunk3);

    v_r0 = _mm_unpacklo_epi16(layer3_chunk0, layer3_chunk2);
    v_r1 = _mm_unpackhi_epi16(layer3_chunk0, layer3_chunk2);
    v_g0 = _mm_unpacklo_epi16(layer3_chunk1, layer3_chunk3);
    v_g1 = _mm_unpackhi_epi16(layer3_chunk1, layer3_chunk3);
}

inline void _mm_deinterleave_epi16(__m128i & v_r0, __m128i & v_r1, __m128i & v_g0,
                                   __m128i & v_g1, __m128i & v_b0, __m128i & v_b1)
{
    __m128i layer1_chunk0 = _mm_unpacklo_epi16(v_r0, v_g1);
    __m128i layer1_chunk1 = _mm_unpackhi_epi16(v_r0, v_g1);
    __m128i layer1_chunk2 = _mm_unpacklo_epi16(v_r1, v_b0);
    __m128i layer1_chunk3 = _mm_unpackhi_epi16(v_r1, v_b0);
    __m128i layer1_chunk4 = _mm_unpacklo_epi16(v_g0, v_b1);
    __m128i layer1_chunk5 = _mm_unpackhi_epi16(v_g0, v_b1);

    __m128i layer2_chunk0 = _mm_unpacklo_epi16(layer1_chunk0, layer1_chunk3);
    __m128i layer2_chunk1 = _mm_unpackhi_epi16(layer1_chunk0, layer1_chunk3);
    __m128i layer2_chunk2 = _mm_unpacklo_epi16(layer1_chunk1, layer1_chunk4);
    __m128i layer2_chunk3 = _mm_unpackhi_epi16(layer1_chunk1, layer1_chunk4);
    __m128i layer2_chunk4 = _mm_unpacklo_epi16(layer1_chunk2, layer1_chunk5);
    __m128i layer2_chunk5 = _mm_unpackhi_epi16(layer1_chunk2, layer1_chunk5);

    __m128i layer3_chunk0 = _mm_unpacklo_epi16(layer2_chunk0, layer2_chunk3);
    __m128i layer3_chunk1 = _mm_unpackhi_epi16(layer2_chunk0, layer2_chunk3);
    __m128i layer3_chunk2 = _mm_unpacklo_epi16(layer2_chunk1, layer2_chunk4);
    __m128i layer3_chunk3 = _mm_unpackhi_epi16(layer2_chunk1, layer2_chunk4);
    __m128i layer3_chunk4 = _mm_unpacklo_epi16(layer2_chunk2, layer2_chunk5);
    __m128i layer3_chunk5 = _mm_unpackhi_epi16(layer2_chunk2, layer2_chunk5);

    v_r0 = _mm_unpacklo_epi16(layer3_chunk0, layer3_chunk3);
    v_r1 = _mm_unpackhi_epi16(layer3_chunk0, layer3_chunk3);
    v_g0 = _mm_unpacklo_epi16(layer3_chunk1, layer3_chunk4);
    v_g1 = _mm_unpackhi_epi16(layer3_chunk1, layer3_chunk4);
    v_b0 = _mm_unpacklo_epi16(layer3_chunk2, layer3_chunk5);
    v_b1 = _mm_unpackhi_epi16(layer3_chunk2, layer3_chunk5);
}

inline void _mm_deinterleave_epi16(__m128i & v_r0, __m128i & v_r1, __m128i & v_g0, __m128i & v_g1,
                                   __m128i & v_b0, __m128i & v_b1, __m128i & v_a0, __m128i & v_a1)
{
    __m128i layer1_chunk0 = _mm_unpacklo_epi16(v_r0, v_b0);
    __m128i layer1_chunk1 = _mm_unpackhi_epi16(v_r0, v_b0);
    __m128i layer1_chunk2 = _mm_unpacklo_epi16(v_r1, v_b1);
    __m128i layer1_chunk3 = _mm_unpackhi_epi16(v_r1, v_b1);
    __m128i layer1_chunk4 = _mm_unpacklo_epi16(v_g0, v_a0);
    __m128i layer1_chunk5 = _mm_unpackhi_epi16(v_g0, v_a0);
    __m128i layer1_chunk6 = _mm_unpacklo_epi16(v_g1, v_a1);
    __m128i layer1_chunk7 = _mm_unpackhi_epi16(v_g1, v_a1);

    __m128i layer2_chunk0 = _mm_unpacklo_epi16(layer1_chunk0, layer1_chunk4);
    __m128i layer2_chunk1 = _mm_unpackhi_epi16(layer1_chunk0, layer1_chunk4);
    __m128i layer2_chunk2 = _mm_unpacklo_epi16(layer1_chunk1, layer1_chunk5);
    __m128i layer2_chunk3 = _mm_unpackhi_epi16(layer1_chunk1, layer1_chunk5);
    __m128i layer2_chunk4 = _mm_unpacklo_epi16(layer1_chunk2, layer1_chunk6);
    __m128i layer2_chunk5 = _mm_unpackhi_epi16(layer1_chunk2, layer1_chunk6);
    __m128i layer2_chunk6 = _mm_unpacklo_epi16(layer1_chunk3, layer1_chunk7);
    __m128i layer2_chunk7 = _mm_unpackhi_epi16(layer1_chunk3, layer1_chunk7);

    __m128i layer3_chunk0 = _mm_unpacklo_epi16(layer2_chunk0, layer2_chunk4);
    __m128i layer3_chunk1 = _mm_unpackhi_epi16(layer2_chunk0, layer2_chunk4);
    __m128i layer3_chunk2 = _mm_unpacklo_epi16(layer2_chunk1, layer2_chunk5);
    __m128i layer3_chunk3 = _mm_unpackhi_epi16(layer2_chunk1, layer2_chunk5);
    __m128i layer3_chunk4 = _mm_unpacklo_epi16(layer2_chunk2, layer2_chunk6);
    __m128i layer3_chunk5 = _mm_unpackhi_epi16(layer2_chunk2, layer2_chunk6);
    __m128i layer3_chunk6 = _mm_unpacklo_epi16(layer2_chunk3, layer2_chunk7);
    __m128i layer3_chunk7 = _mm_unpackhi_epi16(layer2_chunk3, layer2_chunk7);

    v_r0 = _mm_unpacklo_epi16(layer3_chunk0, layer3_chunk4);
    v_r1 = _mm_unpackhi_epi16(layer3_chunk0, layer3_chunk4);
    v_g0 = _mm_unpacklo_epi16(layer3_chunk1, layer3_chunk5);
    v_g1 = _mm_unpackhi_epi16(layer3_chunk1, layer3_chunk5);
    v_b0 = _mm_unpacklo_epi16(layer3_chunk2, layer3_chunk6);
    v_b1 = _mm_unpackhi_epi16(layer3_chunk2, layer3_chunk6);
    v_a0 = _mm_unpacklo_epi16(layer3_chunk3, layer3_chunk7);
    v_a1 = _mm_unpackhi_epi16(layer3_chunk3, layer3_chunk7);
}

#if CV_SSE4_1

inline void _mm_interleave_epi16(__m128i & v_r0, __m128i & v_r1, __m128i & v_g0, __m128i & v_g1)
{
    __m128i v_mask = _mm_set1_epi32(0x0000ffff);

    __m128i layer3_chunk0 = _mm_packus_epi32(_mm_and_si128(v_r0, v_mask), _mm_and_si128(v_r1, v_mask));
    __m128i layer3_chunk2 = _mm_packus_epi32(_mm_srli_epi32(v_r0, 16), _mm_srli_epi32(v_r1, 16));
    __m128i layer3_chunk1 = _mm_packus_epi32(_mm_and_si128(v_g0, v_mask), _mm_and_si128(v_g1, v_mask));
    __m128i layer3_chunk3 = _mm_packus_epi32(_mm_srli_epi32(v_g0, 16), _mm_srli_epi32(v_g1, 16));

    __m128i layer2_chunk0 = _mm_packus_epi32(_mm_and_si128(layer3_chunk0, v_mask), _mm_and_si128(layer3_chunk1, v_mask));
    __m128i layer2_chunk2 = _mm_packus_epi32(_mm_srli_epi32(layer3_chunk0, 16), _mm_srli_epi32(layer3_chunk1, 16));
    __m128i layer2_chunk1 = _mm_packus_epi32(_mm_and_si128(layer3_chunk2, v_mask), _mm_and_si128(layer3_chunk3, v_mask));
    __m128i layer2_chunk3 = _mm_packus_epi32(_mm_srli_epi32(layer3_chunk2, 16), _mm_srli_epi32(layer3_chunk3, 16));

    __m128i layer1_chunk0 = _mm_packus_epi32(_mm_and_si128(layer2_chunk0, v_mask), _mm_and_si128(layer2_chunk1, v_mask));
    __m128i layer1_chunk2 = _mm_packus_epi32(_mm_srli_epi32(layer2_chunk0, 16), _mm_srli_epi32(layer2_chunk1, 16));
    __m128i layer1_chunk1 = _mm_packus_epi32(_mm_and_si128(layer2_chunk2, v_mask), _mm_and_si128(layer2_chunk3, v_mask));
    __m128i layer1_chunk3 = _mm_packus_epi32(_mm_srli_epi32(layer2_chunk2, 16), _mm_srli_epi32(layer2_chunk3, 16));

    v_r0 = _mm_packus_epi32(_mm_and_si128(layer1_chunk0, v_mask), _mm_and_si128(layer1_chunk1, v_mask));
    v_g0 = _mm_packus_epi32(_mm_srli_epi32(layer1_chunk0, 16), _mm_srli_epi32(layer1_chunk1, 16));
    v_r1 = _mm_packus_epi32(_mm_and_si128(layer1_chunk2, v_mask), _mm_and_si128(layer1_chunk3, v_mask));
    v_g1 = _mm_packus_epi32(_mm_srli_epi32(layer1_chunk2, 16), _mm_srli_epi32(layer1_chunk3, 16));
}

inline void _mm_interleave_epi16(__m128i & v_r0, __m128i & v_r1, __m128i & v_g0,
                                 __m128i & v_g1, __m128i & v_b0, __m128i & v_b1)
{
    __m128i v_mask = _mm_set1_epi32(0x0000ffff);

    __m128i layer3_chunk0 = _mm_packus_epi32(_mm_and_si128(v_r0, v_mask), _mm_and_si128(v_r1, v_mask));
    __m128i layer3_chunk3 = _mm_packus_epi32(_mm_srli_epi32(v_r0, 16), _mm_srli_epi32(v_r1, 16));
    __m128i layer3_chunk1 = _mm_packus_epi32(_mm_and_si128(v_g0, v_mask), _mm_and_si128(v_g1, v_mask));
    __m128i layer3_chunk4 = _mm_packus_epi32(_mm_srli_epi32(v_g0, 16), _mm_srli_epi32(v_g1, 16));
    __m128i layer3_chunk2 = _mm_packus_epi32(_mm_and_si128(v_b0, v_mask), _mm_and_si128(v_b1, v_mask));
    __m128i layer3_chunk5 = _mm_packus_epi32(_mm_srli_epi32(v_b0, 16), _mm_srli_epi32(v_b1, 16));

    __m128i layer2_chunk0 = _mm_packus_epi32(_mm_and_si128(layer3_chunk0, v_mask), _mm_and_si128(layer3_chunk1, v_mask));
    __m128i layer2_chunk3 = _mm_packus_epi32(_mm_srli_epi32(layer3_chunk0, 16), _mm_srli_epi32(layer3_chunk1, 16));
    __m128i layer2_chunk1 = _mm_packus_epi32(_mm_and_si128(layer3_chunk2, v_mask), _mm_and_si128(layer3_chunk3, v_mask));
    __m128i layer2_chunk4 = _mm_packus_epi32(_mm_srli_epi32(layer3_chunk2, 16), _mm_srli_epi32(layer3_chunk3, 16));
    __m128i layer2_chunk2 = _mm_packus_epi32(_mm_and_si128(layer3_chunk4, v_mask), _mm_and_si128(layer3_chunk5, v_mask));
    __m128i layer2_chunk5 = _mm_packus_epi32(_mm_srli_epi32(layer3_chunk4, 16), _mm_srli_epi32(layer3_chunk5, 16));

    __m128i layer1_chunk0 = _mm_packus_epi32(_mm_and_si128(layer2_chunk0, v_mask), _mm_and_si128(layer2_chunk1, v_mask));
    __m128i layer1_chunk3 = _mm_packus_epi32(_mm_srli_epi32(layer2_chunk0, 16), _mm_srli_epi32(layer2_chunk1, 16));
    __m128i layer1_chunk1 = _mm_packus_epi32(_mm_and_si128(layer2_chunk2, v_mask), _mm_and_si128(layer2_chunk3, v_mask));
    __m128i layer1_chunk4 = _mm_packus_epi32(_mm_srli_epi32(layer2_chunk2, 16), _mm_srli_epi32(layer2_chunk3, 16));
    __m128i layer1_chunk2 = _mm_packus_epi32(_mm_and_si128(layer2_chunk4, v_mask), _mm_and_si128(layer2_chunk5, v_mask));
    __m128i layer1_chunk5 = _mm_packus_epi32(_mm_srli_epi32(layer2_chunk4, 16), _mm_srli_epi32(layer2_chunk5, 16));

    v_r0 = _mm_packus_epi32(_mm_and_si128(layer1_chunk0, v_mask), _mm_and_si128(layer1_chunk1, v_mask));
    v_g1 = _mm_packus_epi32(_mm_srli_epi32(layer1_chunk0, 16), _mm_srli_epi32(layer1_chunk1, 16));
    v_r1 = _mm_packus_epi32(_mm_and_si128(layer1_chunk2, v_mask), _mm_and_si128(layer1_chunk3, v_mask));
    v_b0 = _mm_packus_epi32(_mm_srli_epi32(layer1_chunk2, 16), _mm_srli_epi32(layer1_chunk3, 16));
    v_g0 = _mm_packus_epi32(_mm_and_si128(layer1_chunk4, v_mask), _mm_and_si128(layer1_chunk5, v_mask));
    v_b1 = _mm_packus_epi32(_mm_srli_epi32(layer1_chunk4, 16), _mm_srli_epi32(layer1_chunk5, 16));
}

inline void _mm_interleave_epi16(__m128i & v_r0, __m128i & v_r1, __m128i & v_g0, __m128i & v_g1,
                                 __m128i & v_b0, __m128i & v_b1, __m128i & v_a0, __m128i & v_a1)
{
    __m128i v_mask = _mm_set1_epi32(0x0000ffff);

    __m128i layer3_chunk0 = _mm_packus_epi32(_mm_and_si128(v_r0, v_mask), _mm_and_si128(v_r1, v_mask));
    __m128i layer3_chunk4 = _mm_packus_epi32(_mm_srli_epi32(v_r0, 16), _mm_srli_epi32(v_r1, 16));
    __m128i layer3_chunk1 = _mm_packus_epi32(_mm_and_si128(v_g0, v_mask), _mm_and_si128(v_g1, v_mask));
    __m128i layer3_chunk5 = _mm_packus_epi32(_mm_srli_epi32(v_g0, 16), _mm_srli_epi32(v_g1, 16));
    __m128i layer3_chunk2 = _mm_packus_epi32(_mm_and_si128(v_b0, v_mask), _mm_and_si128(v_b1, v_mask));
    __m128i layer3_chunk6 = _mm_packus_epi32(_mm_srli_epi32(v_b0, 16), _mm_srli_epi32(v_b1, 16));
    __m128i layer3_chunk3 = _mm_packus_epi32(_mm_and_si128(v_a0, v_mask), _mm_and_si128(v_a1, v_mask));
    __m128i layer3_chunk7 = _mm_packus_epi32(_mm_srli_epi32(v_a0, 16), _mm_srli_epi32(v_a1, 16));

    __m128i layer2_chunk0 = _mm_packus_epi32(_mm_and_si128(layer3_chunk0, v_mask), _mm_and_si128(layer3_chunk1, v_mask));
    __m128i layer2_chunk4 = _mm_packus_epi32(_mm_srli_epi32(layer3_chunk0, 16), _mm_srli_epi32(layer3_chunk1, 16));
    __m128i layer2_chunk1 = _mm_packus_epi32(_mm_and_si128(layer3_chunk2, v_mask), _mm_and_si128(layer3_chunk3, v_mask));
    __m128i layer2_chunk5 = _mm_packus_epi32(_mm_srli_epi32(layer3_chunk2, 16), _mm_srli_epi32(layer3_chunk3, 16));
    __m128i layer2_chunk2 = _mm_packus_epi32(_mm_and_si128(layer3_chunk4, v_mask), _mm_and_si128(layer3_chunk5, v_mask));
    __m128i layer2_chunk6 = _mm_packus_epi32(_mm_srli_epi32(layer3_chunk4, 16), _mm_srli_epi32(layer3_chunk5, 16));
    __m128i layer2_chunk3 = _mm_packus_epi32(_mm_and_si128(layer3_chunk6, v_mask), _mm_and_si128(layer3_chunk7, v_mask));
    __m128i layer2_chunk7 = _mm_packus_epi32(_mm_srli_epi32(layer3_chunk6, 16), _mm_srli_epi32(layer3_chunk7, 16));

    __m128i layer1_chunk0 = _mm_packus_epi32(_mm_and_si128(layer2_chunk0, v_mask), _mm_and_si128(layer2_chunk1, v_mask));
    __m128i layer1_chunk4 = _mm_packus_epi32(_mm_srli_epi32(layer2_chunk0, 16), _mm_srli_epi32(layer2_chunk1, 16));
    __m128i layer1_chunk1 = _mm_packus_epi32(_mm_and_si128(layer2_chunk2, v_mask), _mm_and_si128(layer2_chunk3, v_mask));
    __m128i layer1_chunk5 = _mm_packus_epi32(_mm_srli_epi32(layer2_chunk2, 16), _mm_srli_epi32(layer2_chunk3, 16));
    __m128i layer1_chunk2 = _mm_packus_epi32(_mm_and_si128(layer2_chunk4, v_mask), _mm_and_si128(layer2_chunk5, v_mask));
    __m128i layer1_chunk6 = _mm_packus_epi32(_mm_srli_epi32(layer2_chunk4, 16), _mm_srli_epi32(layer2_chunk5, 16));
    __m128i layer1_chunk3 = _mm_packus_epi32(_mm_and_si128(layer2_chunk6, v_mask), _mm_and_si128(layer2_chunk7, v_mask));
    __m128i layer1_chunk7 = _mm_packus_epi32(_mm_srli_epi32(layer2_chunk6, 16), _mm_srli_epi32(layer2_chunk7, 16));

    v_r0 = _mm_packus_epi32(_mm_and_si128(layer1_chunk0, v_mask), _mm_and_si128(layer1_chunk1, v_mask));
    v_b0 = _mm_packus_epi32(_mm_srli_epi32(layer1_chunk0, 16), _mm_srli_epi32(layer1_chunk1, 16));
    v_r1 = _mm_packus_epi32(_mm_and_si128(layer1_chunk2, v_mask), _mm_and_si128(layer1_chunk3, v_mask));
    v_b1 = _mm_packus_epi32(_mm_srli_epi32(layer1_chunk2, 16), _mm_srli_epi32(layer1_chunk3, 16));
    v_g0 = _mm_packus_epi32(_mm_and_si128(layer1_chunk4, v_mask), _mm_and_si128(layer1_chunk5, v_mask));
    v_a0 = _mm_packus_epi32(_mm_srli_epi32(layer1_chunk4, 16), _mm_srli_epi32(layer1_chunk5, 16));
    v_g1 = _mm_packus_epi32(_mm_and_si128(layer1_chunk6, v_mask), _mm_and_si128(layer1_chunk7, v_mask));
    v_a1 = _mm_packus_epi32(_mm_srli_epi32(layer1_chunk6, 16), _mm_srli_epi32(layer1_chunk7, 16));
}

#endif // CV_SSE4_1

inline void _mm_deinterleave_ps(__m128 & v_r0, __m128 & v_r1, __m128 & v_g0, __m128 & v_g1)
{
    __m128 layer1_chunk0 = _mm_unpacklo_ps(v_r0, v_g0);
    __m128 layer1_chunk1 = _mm_unpackhi_ps(v_r0, v_g0);
    __m128 layer1_chunk2 = _mm_unpacklo_ps(v_r1, v_g1);
    __m128 layer1_chunk3 = _mm_unpackhi_ps(v_r1, v_g1);

    __m128 layer2_chunk0 = _mm_unpacklo_ps(layer1_chunk0, layer1_chunk2);
    __m128 layer2_chunk1 = _mm_unpackhi_ps(layer1_chunk0, layer1_chunk2);
    __m128 layer2_chunk2 = _mm_unpacklo_ps(layer1_chunk1, layer1_chunk3);
    __m128 layer2_chunk3 = _mm_unpackhi_ps(layer1_chunk1, layer1_chunk3);

    v_r0 = _mm_unpacklo_ps(layer2_chunk0, layer2_chunk2);
    v_r1 = _mm_unpackhi_ps(layer2_chunk0, layer2_chunk2);
    v_g0 = _mm_unpacklo_ps(layer2_chunk1, layer2_chunk3);
    v_g1 = _mm_unpackhi_ps(layer2_chunk1, layer2_chunk3);
}

inline void _mm_deinterleave_ps(__m128 & v_r0, __m128 & v_r1, __m128 & v_g0,
                                __m128 & v_g1, __m128 & v_b0, __m128 & v_b1)
{
    __m128 layer1_chunk0 = _mm_unpacklo_ps(v_r0, v_g1);
    __m128 layer1_chunk1 = _mm_unpackhi_ps(v_r0, v_g1);
    __m128 layer1_chunk2 = _mm_unpacklo_ps(v_r1, v_b0);
    __m128 layer1_chunk3 = _mm_unpackhi_ps(v_r1, v_b0);
    __m128 layer1_chunk4 = _mm_unpacklo_ps(v_g0, v_b1);
    __m128 layer1_chunk5 = _mm_unpackhi_ps(v_g0, v_b1);

    __m128 layer2_chunk0 = _mm_unpacklo_ps(layer1_chunk0, layer1_chunk3);
    __m128 layer2_chunk1 = _mm_unpackhi_ps(layer1_chunk0, layer1_chunk3);
    __m128 layer2_chunk2 = _mm_unpacklo_ps(layer1_chunk1, layer1_chunk4);
    __m128 layer2_chunk3 = _mm_unpackhi_ps(layer1_chunk1, layer1_chunk4);
    __m128 layer2_chunk4 = _mm_unpacklo_ps(layer1_chunk2, layer1_chunk5);
    __m128 layer2_chunk5 = _mm_unpackhi_ps(layer1_chunk2, layer1_chunk5);

    v_r0 = _mm_unpacklo_ps(layer2_chunk0, layer2_chunk3);
    v_r1 = _mm_unpackhi_ps(layer2_chunk0, layer2_chunk3);
    v_g0 = _mm_unpacklo_ps(layer2_chunk1, layer2_chunk4);
    v_g1 = _mm_unpackhi_ps(layer2_chunk1, layer2_chunk4);
    v_b0 = _mm_unpacklo_ps(layer2_chunk2, layer2_chunk5);
    v_b1 = _mm_unpackhi_ps(layer2_chunk2, layer2_chunk5);
}

inline void _mm_deinterleave_ps(__m128 & v_r0, __m128 & v_r1, __m128 & v_g0, __m128 & v_g1,
                                __m128 & v_b0, __m128 & v_b1, __m128 & v_a0, __m128 & v_a1)
{
    __m128 layer1_chunk0 = _mm_unpacklo_ps(v_r0, v_b0);
    __m128 layer1_chunk1 = _mm_unpackhi_ps(v_r0, v_b0);
    __m128 layer1_chunk2 = _mm_unpacklo_ps(v_r1, v_b1);
    __m128 layer1_chunk3 = _mm_unpackhi_ps(v_r1, v_b1);
    __m128 layer1_chunk4 = _mm_unpacklo_ps(v_g0, v_a0);
    __m128 layer1_chunk5 = _mm_unpackhi_ps(v_g0, v_a0);
    __m128 layer1_chunk6 = _mm_unpacklo_ps(v_g1, v_a1);
    __m128 layer1_chunk7 = _mm_unpackhi_ps(v_g1, v_a1);

    __m128 layer2_chunk0 = _mm_unpacklo_ps(layer1_chunk0, layer1_chunk4);
    __m128 layer2_chunk1 = _mm_unpackhi_ps(layer1_chunk0, layer1_chunk4);
    __m128 layer2_chunk2 = _mm_unpacklo_ps(layer1_chunk1, layer1_chunk5);
    __m128 layer2_chunk3 = _mm_unpackhi_ps(layer1_chunk1, layer1_chunk5);
    __m128 layer2_chunk4 = _mm_unpacklo_ps(layer1_chunk2, layer1_chunk6);
    __m128 layer2_chunk5 = _mm_unpackhi_ps(layer1_chunk2, layer1_chunk6);
    __m128 layer2_chunk6 = _mm_unpacklo_ps(layer1_chunk3, layer1_chunk7);
    __m128 layer2_chunk7 = _mm_unpackhi_ps(layer1_chunk3, layer1_chunk7);

    v_r0 = _mm_unpacklo_ps(layer2_chunk0, layer2_chunk4);
    v_r1 = _mm_unpackhi_ps(layer2_chunk0, layer2_chunk4);
    v_g0 = _mm_unpacklo_ps(layer2_chunk1, layer2_chunk5);
    v_g1 = _mm_unpackhi_ps(layer2_chunk1, layer2_chunk5);
    v_b0 = _mm_unpacklo_ps(layer2_chunk2, layer2_chunk6);
    v_b1 = _mm_unpackhi_ps(layer2_chunk2, layer2_chunk6);
    v_a0 = _mm_unpacklo_ps(layer2_chunk3, layer2_chunk7);
    v_a1 = _mm_unpackhi_ps(layer2_chunk3, layer2_chunk7);
}

inline void _mm_interleave_ps(__m128 & v_r0, __m128 & v_r1, __m128 & v_g0, __m128 & v_g1)
{
    const int mask_lo = _MM_SHUFFLE(2, 0, 2, 0), mask_hi = _MM_SHUFFLE(3, 1, 3, 1);

    __m128 layer2_chunk0 = _mm_shuffle_ps(v_r0, v_r1, mask_lo);
    __m128 layer2_chunk2 = _mm_shuffle_ps(v_r0, v_r1, mask_hi);
    __m128 layer2_chunk1 = _mm_shuffle_ps(v_g0, v_g1, mask_lo);
    __m128 layer2_chunk3 = _mm_shuffle_ps(v_g0, v_g1, mask_hi);

    __m128 layer1_chunk0 = _mm_shuffle_ps(layer2_chunk0, layer2_chunk1, mask_lo);
    __m128 layer1_chunk2 = _mm_shuffle_ps(layer2_chunk0, layer2_chunk1, mask_hi);
    __m128 layer1_chunk1 = _mm_shuffle_ps(layer2_chunk2, layer2_chunk3, mask_lo);
    __m128 layer1_chunk3 = _mm_shuffle_ps(layer2_chunk2, layer2_chunk3, mask_hi);

    v_r0 = _mm_shuffle_ps(layer1_chunk0, layer1_chunk1, mask_lo);
    v_g0 = _mm_shuffle_ps(layer1_chunk0, layer1_chunk1, mask_hi);
    v_r1 = _mm_shuffle_ps(layer1_chunk2, layer1_chunk3, mask_lo);
    v_g1 = _mm_shuffle_ps(layer1_chunk2, layer1_chunk3, mask_hi);
}

inline void _mm_interleave_ps(__m128 & v_r0, __m128 & v_r1, __m128 & v_g0,
                              __m128 & v_g1, __m128 & v_b0, __m128 & v_b1)
{
    const int mask_lo = _MM_SHUFFLE(2, 0, 2, 0), mask_hi = _MM_SHUFFLE(3, 1, 3, 1);

    __m128 layer2_chunk0 = _mm_shuffle_ps(v_r0, v_r1, mask_lo);
    __m128 layer2_chunk3 = _mm_shuffle_ps(v_r0, v_r1, mask_hi);
    __m128 layer2_chunk1 = _mm_shuffle_ps(v_g0, v_g1, mask_lo);
    __m128 layer2_chunk4 = _mm_shuffle_ps(v_g0, v_g1, mask_hi);
    __m128 layer2_chunk2 = _mm_shuffle_ps(v_b0, v_b1, mask_lo);
    __m128 layer2_chunk5 = _mm_shuffle_ps(v_b0, v_b1, mask_hi);

    __m128 layer1_chunk0 = _mm_shuffle_ps(layer2_chunk0, layer2_chunk1, mask_lo);
    __m128 layer1_chunk3 = _mm_shuffle_ps(layer2_chunk0, layer2_chunk1, mask_hi);
    __m128 layer1_chunk1 = _mm_shuffle_ps(layer2_chunk2, layer2_chunk3, mask_lo);
    __m128 layer1_chunk4 = _mm_shuffle_ps(layer2_chunk2, layer2_chunk3, mask_hi);
    __m128 layer1_chunk2 = _mm_shuffle_ps(layer2_chunk4, layer2_chunk5, mask_lo);
    __m128 layer1_chunk5 = _mm_shuffle_ps(layer2_chunk4, layer2_chunk5, mask_hi);

    v_r0 = _mm_shuffle_ps(layer1_chunk0, layer1_chunk1, mask_lo);
    v_g1 = _mm_shuffle_ps(layer1_chunk0, layer1_chunk1, mask_hi);
    v_r1 = _mm_shuffle_ps(layer1_chunk2, layer1_chunk3, mask_lo);
    v_b0 = _mm_shuffle_ps(layer1_chunk2, layer1_chunk3, mask_hi);
    v_g0 = _mm_shuffle_ps(layer1_chunk4, layer1_chunk5, mask_lo);
    v_b1 = _mm_shuffle_ps(layer1_chunk4, layer1_chunk5, mask_hi);
}

inline void _mm_interleave_ps(__m128 & v_r0, __m128 & v_r1, __m128 & v_g0, __m128 & v_g1,
                              __m128 & v_b0, __m128 & v_b1, __m128 & v_a0, __m128 & v_a1)
{
    const int mask_lo = _MM_SHUFFLE(2, 0, 2, 0), mask_hi = _MM_SHUFFLE(3, 1, 3, 1);

    __m128 layer2_chunk0 = _mm_shuffle_ps(v_r0, v_r1, mask_lo);
    __m128 layer2_chunk4 = _mm_shuffle_ps(v_r0, v_r1, mask_hi);
    __m128 layer2_chunk1 = _mm_shuffle_ps(v_g0, v_g1, mask_lo);
    __m128 layer2_chunk5 = _mm_shuffle_ps(v_g0, v_g1, mask_hi);
    __m128 layer2_chunk2 = _mm_shuffle_ps(v_b0, v_b1, mask_lo);
    __m128 layer2_chunk6 = _mm_shuffle_ps(v_b0, v_b1, mask_hi);
    __m128 layer2_chunk3 = _mm_shuffle_ps(v_a0, v_a1, mask_lo);
    __m128 layer2_chunk7 = _mm_shuffle_ps(v_a0, v_a1, mask_hi);

    __m128 layer1_chunk0 = _mm_shuffle_ps(layer2_chunk0, layer2_chunk1, mask_lo);
    __m128 layer1_chunk4 = _mm_shuffle_ps(layer2_chunk0, layer2_chunk1, mask_hi);
    __m128 layer1_chunk1 = _mm_shuffle_ps(layer2_chunk2, layer2_chunk3, mask_lo);
    __m128 layer1_chunk5 = _mm_shuffle_ps(layer2_chunk2, layer2_chunk3, mask_hi);
    __m128 layer1_chunk2 = _mm_shuffle_ps(layer2_chunk4, layer2_chunk5, mask_lo);
    __m128 layer1_chunk6 = _mm_shuffle_ps(layer2_chunk4, layer2_chunk5, mask_hi);
    __m128 layer1_chunk3 = _mm_shuffle_ps(layer2_chunk6, layer2_chunk7, mask_lo);
    __m128 layer1_chunk7 = _mm_shuffle_ps(layer2_chunk6, layer2_chunk7, mask_hi);

    v_r0 = _mm_shuffle_ps(layer1_chunk0, layer1_chunk1, mask_lo);
    v_b0 = _mm_shuffle_ps(layer1_chunk0, layer1_chunk1, mask_hi);
    v_r1 = _mm_shuffle_ps(layer1_chunk2, layer1_chunk3, mask_lo);
    v_b1 = _mm_shuffle_ps(layer1_chunk2, layer1_chunk3, mask_hi);
    v_g0 = _mm_shuffle_ps(layer1_chunk4, layer1_chunk5, mask_lo);
    v_a0 = _mm_shuffle_ps(layer1_chunk4, layer1_chunk5, mask_hi);
    v_g1 = _mm_shuffle_ps(layer1_chunk6, layer1_chunk7, mask_lo);
    v_a1 = _mm_shuffle_ps(layer1_chunk6, layer1_chunk7, mask_hi);
}

#endif // CV_SSE2

//! @}

#endif //OPENCV_CORE_SSE_UTILS_HPP
