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

#ifndef CAROTENE_SRC_VTRANSFORM_HPP
#define CAROTENE_SRC_VTRANSFORM_HPP

#include "common.hpp"

#include <carotene/types.hpp>

#ifdef CAROTENE_NEON

namespace CAROTENE_NS { namespace internal {

////////////////////////////// Type Traits ///////////////////////

template <typename T, int cn = 1>
struct VecTraits;

template <> struct VecTraits< u8, 1> { typedef  uint8x16_t vec128; typedef   uint8x8_t vec64; typedef VecTraits<  u8, 1> unsign; };
template <> struct VecTraits< s8, 1> { typedef   int8x16_t vec128; typedef    int8x8_t vec64; typedef VecTraits<  u8, 1> unsign; };
template <> struct VecTraits<u16, 1> { typedef  uint16x8_t vec128; typedef  uint16x4_t vec64; typedef VecTraits< u16, 1> unsign; };
template <> struct VecTraits<s16, 1> { typedef   int16x8_t vec128; typedef   int16x4_t vec64; typedef VecTraits< u16, 1> unsign; };
template <> struct VecTraits<s32, 1> { typedef   int32x4_t vec128; typedef   int32x2_t vec64; typedef VecTraits< u32, 1> unsign; };
template <> struct VecTraits<u32, 1> { typedef  uint32x4_t vec128; typedef  uint32x2_t vec64; typedef VecTraits< u32, 1> unsign; };
template <> struct VecTraits<s64, 1> { typedef   int64x2_t vec128; typedef   int64x1_t vec64; typedef VecTraits< u64, 1> unsign; };
template <> struct VecTraits<u64, 1> { typedef  uint64x2_t vec128; typedef  uint64x1_t vec64; typedef VecTraits< u64, 1> unsign; };
template <> struct VecTraits<f32, 1> { typedef float32x4_t vec128; typedef float32x2_t vec64; typedef VecTraits< u32, 1> unsign; };

template <> struct VecTraits< u8, 2> { typedef  uint8x16x2_t vec128; typedef   uint8x8x2_t vec64; typedef VecTraits<  u8, 2> unsign; };
template <> struct VecTraits< s8, 2> { typedef   int8x16x2_t vec128; typedef    int8x8x2_t vec64; typedef VecTraits<  u8, 2> unsign; };
template <> struct VecTraits<u16, 2> { typedef  uint16x8x2_t vec128; typedef  uint16x4x2_t vec64; typedef VecTraits< u16, 2> unsign; };
template <> struct VecTraits<s16, 2> { typedef   int16x8x2_t vec128; typedef   int16x4x2_t vec64; typedef VecTraits< u16, 2> unsign; };
template <> struct VecTraits<s32, 2> { typedef   int32x4x2_t vec128; typedef   int32x2x2_t vec64; typedef VecTraits< u32, 2> unsign; };
template <> struct VecTraits<u32, 2> { typedef  uint32x4x2_t vec128; typedef  uint32x2x2_t vec64; typedef VecTraits< u32, 2> unsign; };
template <> struct VecTraits<s64, 2> { typedef   int64x2x2_t vec128; typedef   int64x1x2_t vec64; typedef VecTraits< u64, 2> unsign; };
template <> struct VecTraits<u64, 2> { typedef  uint64x2x2_t vec128; typedef  uint64x1x2_t vec64; typedef VecTraits< u64, 2> unsign; };
template <> struct VecTraits<f32, 2> { typedef float32x4x2_t vec128; typedef float32x2x2_t vec64; typedef VecTraits< u32, 2> unsign; };

template <> struct VecTraits< u8, 3> { typedef  uint8x16x3_t vec128; typedef   uint8x8x3_t vec64; typedef VecTraits<  u8, 3> unsign; };
template <> struct VecTraits< s8, 3> { typedef   int8x16x3_t vec128; typedef    int8x8x3_t vec64; typedef VecTraits<  u8, 3> unsign; };
template <> struct VecTraits<u16, 3> { typedef  uint16x8x3_t vec128; typedef  uint16x4x3_t vec64; typedef VecTraits< u16, 3> unsign; };
template <> struct VecTraits<s16, 3> { typedef   int16x8x3_t vec128; typedef   int16x4x3_t vec64; typedef VecTraits< u16, 3> unsign; };
template <> struct VecTraits<s32, 3> { typedef   int32x4x3_t vec128; typedef   int32x2x3_t vec64; typedef VecTraits< u32, 3> unsign; };
template <> struct VecTraits<u32, 3> { typedef  uint32x4x3_t vec128; typedef  uint32x2x3_t vec64; typedef VecTraits< u32, 3> unsign; };
template <> struct VecTraits<s64, 3> { typedef   int64x2x3_t vec128; typedef   int64x1x3_t vec64; typedef VecTraits< u64, 2> unsign; };
template <> struct VecTraits<u64, 3> { typedef  uint64x2x3_t vec128; typedef  uint64x1x3_t vec64; typedef VecTraits< u64, 2> unsign; };
template <> struct VecTraits<f32, 3> { typedef float32x4x3_t vec128; typedef float32x2x3_t vec64; typedef VecTraits< u32, 3> unsign; };

template <> struct VecTraits< u8, 4> { typedef  uint8x16x4_t vec128; typedef   uint8x8x4_t vec64; typedef VecTraits<  u8, 3> unsign; };
template <> struct VecTraits< s8, 4> { typedef   int8x16x4_t vec128; typedef    int8x8x4_t vec64; typedef VecTraits<  u8, 3> unsign; };
template <> struct VecTraits<u16, 4> { typedef  uint16x8x4_t vec128; typedef  uint16x4x4_t vec64; typedef VecTraits< u16, 3> unsign; };
template <> struct VecTraits<s16, 4> { typedef   int16x8x4_t vec128; typedef   int16x4x4_t vec64; typedef VecTraits< u16, 3> unsign; };
template <> struct VecTraits<s32, 4> { typedef   int32x4x4_t vec128; typedef   int32x2x4_t vec64; typedef VecTraits< u32, 3> unsign; };
template <> struct VecTraits<u32, 4> { typedef  uint32x4x4_t vec128; typedef  uint32x2x4_t vec64; typedef VecTraits< u32, 3> unsign; };
template <> struct VecTraits<s64, 4> { typedef   int64x2x4_t vec128; typedef   int64x1x4_t vec64; typedef VecTraits< u64, 2> unsign; };
template <> struct VecTraits<u64, 4> { typedef  uint64x2x4_t vec128; typedef  uint64x1x4_t vec64; typedef VecTraits< u64, 2> unsign; };
template <> struct VecTraits<f32, 4> { typedef float32x4x4_t vec128; typedef float32x2x4_t vec64; typedef VecTraits< u32, 3> unsign; };

////////////////////////////// vld1q ///////////////////////

inline  uint8x16_t vld1q(const u8  * ptr) { return  vld1q_u8(ptr); }
inline   int8x16_t vld1q(const s8  * ptr) { return  vld1q_s8(ptr); }
inline  uint16x8_t vld1q(const u16 * ptr) { return vld1q_u16(ptr); }
inline   int16x8_t vld1q(const s16 * ptr) { return vld1q_s16(ptr); }
inline  uint32x4_t vld1q(const u32 * ptr) { return vld1q_u32(ptr); }
inline   int32x4_t vld1q(const s32 * ptr) { return vld1q_s32(ptr); }
inline float32x4_t vld1q(const f32 * ptr) { return vld1q_f32(ptr); }

////////////////////////////// vld1 ///////////////////////

inline   uint8x8_t vld1(const u8  * ptr) { return  vld1_u8(ptr); }
inline    int8x8_t vld1(const s8  * ptr) { return  vld1_s8(ptr); }
inline  uint16x4_t vld1(const u16 * ptr) { return vld1_u16(ptr); }
inline   int16x4_t vld1(const s16 * ptr) { return vld1_s16(ptr); }
inline  uint32x2_t vld1(const u32 * ptr) { return vld1_u32(ptr); }
inline   int32x2_t vld1(const s32 * ptr) { return vld1_s32(ptr); }
inline float32x2_t vld1(const f32 * ptr) { return vld1_f32(ptr); }

////////////////////////////// vld2q ///////////////////////

inline  uint8x16x2_t vld2q(const u8  * ptr) { return  vld2q_u8(ptr); }
inline   int8x16x2_t vld2q(const s8  * ptr) { return  vld2q_s8(ptr); }
inline  uint16x8x2_t vld2q(const u16 * ptr) { return vld2q_u16(ptr); }
inline   int16x8x2_t vld2q(const s16 * ptr) { return vld2q_s16(ptr); }
inline  uint32x4x2_t vld2q(const u32 * ptr) { return vld2q_u32(ptr); }
inline   int32x4x2_t vld2q(const s32 * ptr) { return vld2q_s32(ptr); }
inline float32x4x2_t vld2q(const f32 * ptr) { return vld2q_f32(ptr); }

////////////////////////////// vld2 ///////////////////////

inline   uint8x8x2_t vld2(const u8  * ptr) { return  vld2_u8(ptr); }
inline    int8x8x2_t vld2(const s8  * ptr) { return  vld2_s8(ptr); }
inline  uint16x4x2_t vld2(const u16 * ptr) { return vld2_u16(ptr); }
inline   int16x4x2_t vld2(const s16 * ptr) { return vld2_s16(ptr); }
inline  uint32x2x2_t vld2(const u32 * ptr) { return vld2_u32(ptr); }
inline   int32x2x2_t vld2(const s32 * ptr) { return vld2_s32(ptr); }
inline float32x2x2_t vld2(const f32 * ptr) { return vld2_f32(ptr); }

////////////////////////////// vld3q ///////////////////////

inline  uint8x16x3_t vld3q(const u8  * ptr) { return  vld3q_u8(ptr); }
inline   int8x16x3_t vld3q(const s8  * ptr) { return  vld3q_s8(ptr); }
inline  uint16x8x3_t vld3q(const u16 * ptr) { return vld3q_u16(ptr); }
inline   int16x8x3_t vld3q(const s16 * ptr) { return vld3q_s16(ptr); }
inline  uint32x4x3_t vld3q(const u32 * ptr) { return vld3q_u32(ptr); }
inline   int32x4x3_t vld3q(const s32 * ptr) { return vld3q_s32(ptr); }
inline float32x4x3_t vld3q(const f32 * ptr) { return vld3q_f32(ptr); }

////////////////////////////// vld3 ///////////////////////

inline   uint8x8x3_t vld3(const u8  * ptr) { return  vld3_u8(ptr); }
inline    int8x8x3_t vld3(const s8  * ptr) { return  vld3_s8(ptr); }
inline  uint16x4x3_t vld3(const u16 * ptr) { return vld3_u16(ptr); }
inline   int16x4x3_t vld3(const s16 * ptr) { return vld3_s16(ptr); }
inline  uint32x2x3_t vld3(const u32 * ptr) { return vld3_u32(ptr); }
inline   int32x2x3_t vld3(const s32 * ptr) { return vld3_s32(ptr); }
inline float32x2x3_t vld3(const f32 * ptr) { return vld3_f32(ptr); }

////////////////////////////// vld4q ///////////////////////

inline  uint8x16x4_t vld4q(const u8  * ptr) { return  vld4q_u8(ptr); }
inline   int8x16x4_t vld4q(const s8  * ptr) { return  vld4q_s8(ptr); }
inline  uint16x8x4_t vld4q(const u16 * ptr) { return vld4q_u16(ptr); }
inline   int16x8x4_t vld4q(const s16 * ptr) { return vld4q_s16(ptr); }
inline  uint32x4x4_t vld4q(const u32 * ptr) { return vld4q_u32(ptr); }
inline   int32x4x4_t vld4q(const s32 * ptr) { return vld4q_s32(ptr); }
inline float32x4x4_t vld4q(const f32 * ptr) { return vld4q_f32(ptr); }

////////////////////////////// vld4 ///////////////////////

inline   uint8x8x4_t vld4(const u8  * ptr) { return  vld4_u8(ptr); }
inline    int8x8x4_t vld4(const s8  * ptr) { return  vld4_s8(ptr); }
inline  uint16x4x4_t vld4(const u16 * ptr) { return vld4_u16(ptr); }
inline   int16x4x4_t vld4(const s16 * ptr) { return vld4_s16(ptr); }
inline  uint32x2x4_t vld4(const u32 * ptr) { return vld4_u32(ptr); }
inline   int32x2x4_t vld4(const s32 * ptr) { return vld4_s32(ptr); }
inline float32x2x4_t vld4(const f32 * ptr) { return vld4_f32(ptr); }

////////////////////////////// vst1q ///////////////////////

inline void vst1q(u8  * ptr, const uint8x16_t  & v) { return vst1q_u8(ptr,  v); }
inline void vst1q(s8  * ptr, const int8x16_t   & v) { return vst1q_s8(ptr,  v); }
inline void vst1q(u16 * ptr, const uint16x8_t  & v) { return vst1q_u16(ptr, v); }
inline void vst1q(s16 * ptr, const int16x8_t   & v) { return vst1q_s16(ptr, v); }
inline void vst1q(u32 * ptr, const uint32x4_t  & v) { return vst1q_u32(ptr, v); }
inline void vst1q(s32 * ptr, const int32x4_t   & v) { return vst1q_s32(ptr, v); }
inline void vst1q(f32 * ptr, const float32x4_t & v) { return vst1q_f32(ptr, v); }

////////////////////////////// vst1 ///////////////////////

inline void vst1(u8  * ptr, const uint8x8_t   & v) { return vst1_u8(ptr,  v); }
inline void vst1(s8  * ptr, const int8x8_t    & v) { return vst1_s8(ptr,  v); }
inline void vst1(u16 * ptr, const uint16x4_t  & v) { return vst1_u16(ptr, v); }
inline void vst1(s16 * ptr, const int16x4_t   & v) { return vst1_s16(ptr, v); }
inline void vst1(u32 * ptr, const uint32x2_t  & v) { return vst1_u32(ptr, v); }
inline void vst1(s32 * ptr, const int32x2_t   & v) { return vst1_s32(ptr, v); }
inline void vst1(f32 * ptr, const float32x2_t & v) { return vst1_f32(ptr, v); }

////////////////////////////// vst2q ///////////////////////

inline void vst2q(u8  * ptr, const uint8x16x2_t  & v) { return vst2q_u8(ptr,  v); }
inline void vst2q(s8  * ptr, const int8x16x2_t   & v) { return vst2q_s8(ptr,  v); }
inline void vst2q(u16 * ptr, const uint16x8x2_t  & v) { return vst2q_u16(ptr, v); }
inline void vst2q(s16 * ptr, const int16x8x2_t   & v) { return vst2q_s16(ptr, v); }
inline void vst2q(u32 * ptr, const uint32x4x2_t  & v) { return vst2q_u32(ptr, v); }
inline void vst2q(s32 * ptr, const int32x4x2_t   & v) { return vst2q_s32(ptr, v); }
inline void vst2q(f32 * ptr, const float32x4x2_t & v) { return vst2q_f32(ptr, v); }

////////////////////////////// vst2 ///////////////////////

inline void vst2(u8  * ptr, const uint8x8x2_t   & v) { return vst2_u8(ptr,  v); }
inline void vst2(s8  * ptr, const int8x8x2_t    & v) { return vst2_s8(ptr,  v); }
inline void vst2(u16 * ptr, const uint16x4x2_t  & v) { return vst2_u16(ptr, v); }
inline void vst2(s16 * ptr, const int16x4x2_t   & v) { return vst2_s16(ptr, v); }
inline void vst2(u32 * ptr, const uint32x2x2_t  & v) { return vst2_u32(ptr, v); }
inline void vst2(s32 * ptr, const int32x2x2_t   & v) { return vst2_s32(ptr, v); }
inline void vst2(f32 * ptr, const float32x2x2_t & v) { return vst2_f32(ptr, v); }

////////////////////////////// vst3q ///////////////////////

inline void vst3q(u8  * ptr, const uint8x16x3_t  & v) { return vst3q_u8(ptr,  v); }
inline void vst3q(s8  * ptr, const int8x16x3_t   & v) { return vst3q_s8(ptr,  v); }
inline void vst3q(u16 * ptr, const uint16x8x3_t  & v) { return vst3q_u16(ptr, v); }
inline void vst3q(s16 * ptr, const int16x8x3_t   & v) { return vst3q_s16(ptr, v); }
inline void vst3q(u32 * ptr, const uint32x4x3_t  & v) { return vst3q_u32(ptr, v); }
inline void vst3q(s32 * ptr, const int32x4x3_t   & v) { return vst3q_s32(ptr, v); }
inline void vst3q(f32 * ptr, const float32x4x3_t & v) { return vst3q_f32(ptr, v); }

////////////////////////////// vst3 ///////////////////////

inline void vst3(u8  * ptr, const uint8x8x3_t   & v) { return vst3_u8(ptr,  v); }
inline void vst3(s8  * ptr, const int8x8x3_t    & v) { return vst3_s8(ptr,  v); }
inline void vst3(u16 * ptr, const uint16x4x3_t  & v) { return vst3_u16(ptr, v); }
inline void vst3(s16 * ptr, const int16x4x3_t   & v) { return vst3_s16(ptr, v); }
inline void vst3(u32 * ptr, const uint32x2x3_t  & v) { return vst3_u32(ptr, v); }
inline void vst3(s32 * ptr, const int32x2x3_t   & v) { return vst3_s32(ptr, v); }
inline void vst3(f32 * ptr, const float32x2x3_t & v) { return vst3_f32(ptr, v); }

////////////////////////////// vst4q ///////////////////////

inline void vst4q(u8  * ptr, const uint8x16x4_t  & v) { return vst4q_u8(ptr,  v); }
inline void vst4q(s8  * ptr, const int8x16x4_t   & v) { return vst4q_s8(ptr,  v); }
inline void vst4q(u16 * ptr, const uint16x8x4_t  & v) { return vst4q_u16(ptr, v); }
inline void vst4q(s16 * ptr, const int16x8x4_t   & v) { return vst4q_s16(ptr, v); }
inline void vst4q(u32 * ptr, const uint32x4x4_t  & v) { return vst4q_u32(ptr, v); }
inline void vst4q(s32 * ptr, const int32x4x4_t   & v) { return vst4q_s32(ptr, v); }
inline void vst4q(f32 * ptr, const float32x4x4_t & v) { return vst4q_f32(ptr, v); }

////////////////////////////// vst4 ///////////////////////

inline void vst4(u8  * ptr, const uint8x8x4_t   & v) { return vst4_u8(ptr,  v); }
inline void vst4(s8  * ptr, const int8x8x4_t    & v) { return vst4_s8(ptr,  v); }
inline void vst4(u16 * ptr, const uint16x4x4_t  & v) { return vst4_u16(ptr, v); }
inline void vst4(s16 * ptr, const int16x4x4_t   & v) { return vst4_s16(ptr, v); }
inline void vst4(u32 * ptr, const uint32x2x4_t  & v) { return vst4_u32(ptr, v); }
inline void vst4(s32 * ptr, const int32x2x4_t   & v) { return vst4_s32(ptr, v); }
inline void vst4(f32 * ptr, const float32x2x4_t & v) { return vst4_f32(ptr, v); }

////////////////////////////// vabdq ///////////////////////

inline  uint8x16_t vabdq(const uint8x16_t  & v0, const uint8x16_t  & v1) { return vabdq_u8 (v0, v1); }
inline   int8x16_t vabdq(const int8x16_t   & v0, const int8x16_t   & v1) { return vabdq_s8 (v0, v1); }
inline  uint16x8_t vabdq(const uint16x8_t  & v0, const uint16x8_t  & v1) { return vabdq_u16(v0, v1); }
inline   int16x8_t vabdq(const int16x8_t   & v0, const int16x8_t   & v1) { return vabdq_s16(v0, v1); }
inline  uint32x4_t vabdq(const uint32x4_t  & v0, const uint32x4_t  & v1) { return vabdq_u32(v0, v1); }
inline   int32x4_t vabdq(const int32x4_t   & v0, const int32x4_t   & v1) { return vabdq_s32(v0, v1); }
inline float32x4_t vabdq(const float32x4_t & v0, const float32x4_t & v1) { return vabdq_f32(v0, v1); }

////////////////////////////// vabd ///////////////////////

inline   uint8x8_t vabd(const uint8x8_t   & v0, const uint8x8_t   & v1) { return vabd_u8 (v0, v1); }
inline    int8x8_t vabd(const int8x8_t    & v0, const int8x8_t    & v1) { return vabd_s8 (v0, v1); }
inline  uint16x4_t vabd(const uint16x4_t  & v0, const uint16x4_t  & v1) { return vabd_u16(v0, v1); }
inline   int16x4_t vabd(const int16x4_t   & v0, const int16x4_t   & v1) { return vabd_s16(v0, v1); }
inline  uint32x2_t vabd(const uint32x2_t  & v0, const uint32x2_t  & v1) { return vabd_u32(v0, v1); }
inline   int32x2_t vabd(const int32x2_t   & v0, const int32x2_t   & v1) { return vabd_s32(v0, v1); }
inline float32x2_t vabd(const float32x2_t & v0, const float32x2_t & v1) { return vabd_f32(v0, v1); }

////////////////////////////// vminq ///////////////////////

inline  uint8x16_t vminq(const uint8x16_t  & v0, const uint8x16_t  & v1) { return vminq_u8 (v0, v1); }
inline   int8x16_t vminq(const int8x16_t   & v0, const int8x16_t   & v1) { return vminq_s8 (v0, v1); }
inline  uint16x8_t vminq(const uint16x8_t  & v0, const uint16x8_t  & v1) { return vminq_u16(v0, v1); }
inline   int16x8_t vminq(const int16x8_t   & v0, const int16x8_t   & v1) { return vminq_s16(v0, v1); }
inline  uint32x4_t vminq(const uint32x4_t  & v0, const uint32x4_t  & v1) { return vminq_u32(v0, v1); }
inline   int32x4_t vminq(const int32x4_t   & v0, const int32x4_t   & v1) { return vminq_s32(v0, v1); }
inline float32x4_t vminq(const float32x4_t & v0, const float32x4_t & v1) { return vminq_f32(v0, v1); }

////////////////////////////// vmin ///////////////////////

inline   uint8x8_t vmin(const uint8x8_t   & v0, const uint8x8_t   & v1) { return vmin_u8 (v0, v1); }
inline    int8x8_t vmin(const int8x8_t    & v0, const int8x8_t    & v1) { return vmin_s8 (v0, v1); }
inline  uint16x4_t vmin(const uint16x4_t  & v0, const uint16x4_t  & v1) { return vmin_u16(v0, v1); }
inline   int16x4_t vmin(const int16x4_t   & v0, const int16x4_t   & v1) { return vmin_s16(v0, v1); }
inline  uint32x2_t vmin(const uint32x2_t  & v0, const uint32x2_t  & v1) { return vmin_u32(v0, v1); }
inline   int32x2_t vmin(const int32x2_t   & v0, const int32x2_t   & v1) { return vmin_s32(v0, v1); }
inline float32x2_t vmin(const float32x2_t & v0, const float32x2_t & v1) { return vmin_f32(v0, v1); }

////////////////////////////// vmaxq ///////////////////////

inline  uint8x16_t vmaxq(const uint8x16_t  & v0, const uint8x16_t  & v1) { return vmaxq_u8 (v0, v1); }
inline   int8x16_t vmaxq(const int8x16_t   & v0, const int8x16_t   & v1) { return vmaxq_s8 (v0, v1); }
inline  uint16x8_t vmaxq(const uint16x8_t  & v0, const uint16x8_t  & v1) { return vmaxq_u16(v0, v1); }
inline   int16x8_t vmaxq(const int16x8_t   & v0, const int16x8_t   & v1) { return vmaxq_s16(v0, v1); }
inline  uint32x4_t vmaxq(const uint32x4_t  & v0, const uint32x4_t  & v1) { return vmaxq_u32(v0, v1); }
inline   int32x4_t vmaxq(const int32x4_t   & v0, const int32x4_t   & v1) { return vmaxq_s32(v0, v1); }
inline float32x4_t vmaxq(const float32x4_t & v0, const float32x4_t & v1) { return vmaxq_f32(v0, v1); }

////////////////////////////// vmax ///////////////////////

inline   uint8x8_t vmax(const uint8x8_t   & v0, const uint8x8_t   & v1) { return vmax_u8 (v0, v1); }
inline    int8x8_t vmax(const int8x8_t    & v0, const int8x8_t    & v1) { return vmax_s8 (v0, v1); }
inline  uint16x4_t vmax(const uint16x4_t  & v0, const uint16x4_t  & v1) { return vmax_u16(v0, v1); }
inline   int16x4_t vmax(const int16x4_t   & v0, const int16x4_t   & v1) { return vmax_s16(v0, v1); }
inline  uint32x2_t vmax(const uint32x2_t  & v0, const uint32x2_t  & v1) { return vmax_u32(v0, v1); }
inline   int32x2_t vmax(const int32x2_t   & v0, const int32x2_t   & v1) { return vmax_s32(v0, v1); }
inline float32x2_t vmax(const float32x2_t & v0, const float32x2_t & v1) { return vmax_f32(v0, v1); }

////////////////////////////// vdupq_n ///////////////////////

inline  uint8x16_t vdupq_n(const u8  & val) { return  vdupq_n_u8(val); }
inline   int8x16_t vdupq_n(const s8  & val) { return  vdupq_n_s8(val); }
inline  uint16x8_t vdupq_n(const u16 & val) { return vdupq_n_u16(val); }
inline   int16x8_t vdupq_n(const s16 & val) { return vdupq_n_s16(val); }
inline  uint32x4_t vdupq_n(const u32 & val) { return vdupq_n_u32(val); }
inline   int32x4_t vdupq_n(const s32 & val) { return vdupq_n_s32(val); }
inline  uint64x2_t vdupq_n(const u64 & val) { return vdupq_n_u64(val); }
inline   int64x2_t vdupq_n(const s64 & val) { return vdupq_n_s64(val); }
inline float32x4_t vdupq_n(const f32 & val) { return vdupq_n_f32(val); }

////////////////////////////// vdup_n ///////////////////////

inline   uint8x8_t vdup_n(const u8  & val) { return  vdup_n_u8(val); }
inline    int8x8_t vdup_n(const s8  & val) { return  vdup_n_s8(val); }
inline  uint16x4_t vdup_n(const u16 & val) { return vdup_n_u16(val); }
inline   int16x4_t vdup_n(const s16 & val) { return vdup_n_s16(val); }
inline  uint32x2_t vdup_n(const u32 & val) { return vdup_n_u32(val); }
inline   int32x2_t vdup_n(const s32 & val) { return vdup_n_s32(val); }
inline  uint64x1_t vdup_n(const u64 & val) { return vdup_n_u64(val); }
inline   int64x1_t vdup_n(const s64 & val) { return vdup_n_s64(val); }
inline float32x2_t vdup_n(const f32 & val) { return vdup_n_f32(val); }

////////////////////////////// vget_low ///////////////////////

inline uint8x8_t   vget_low(const uint8x16_t  & v) { return vget_low_u8 (v); }
inline int8x8_t    vget_low(const int8x16_t   & v) { return vget_low_s8 (v); }
inline uint16x4_t  vget_low(const uint16x8_t  & v) { return vget_low_u16(v); }
inline int16x4_t   vget_low(const int16x8_t   & v) { return vget_low_s16(v); }
inline uint32x2_t  vget_low(const uint32x4_t  & v) { return vget_low_u32(v); }
inline int32x2_t   vget_low(const int32x4_t   & v) { return vget_low_s32(v); }
inline float32x2_t vget_low(const float32x4_t & v) { return vget_low_f32(v); }

////////////////////////////// vget_high ///////////////////////

inline uint8x8_t   vget_high(const uint8x16_t  & v) { return vget_high_u8 (v); }
inline int8x8_t    vget_high(const int8x16_t   & v) { return vget_high_s8 (v); }
inline uint16x4_t  vget_high(const uint16x8_t  & v) { return vget_high_u16(v); }
inline int16x4_t   vget_high(const int16x8_t   & v) { return vget_high_s16(v); }
inline uint32x2_t  vget_high(const uint32x4_t  & v) { return vget_high_u32(v); }
inline int32x2_t   vget_high(const int32x4_t   & v) { return vget_high_s32(v); }
inline float32x2_t vget_high(const float32x4_t & v) { return vget_high_f32(v); }

////////////////////////////// vcombine ///////////////////////

inline   uint8x16_t vcombine(const uint8x8_t   & v0, const uint8x8_t   & v1) { return vcombine_u8 (v0, v1); }
inline    int8x16_t vcombine(const int8x8_t    & v0, const int8x8_t    & v1) { return vcombine_s8 (v0, v1); }
inline  uint16x8_t  vcombine(const uint16x4_t  & v0, const uint16x4_t  & v1) { return vcombine_u16(v0, v1); }
inline   int16x8_t  vcombine(const int16x4_t   & v0, const int16x4_t   & v1) { return vcombine_s16(v0, v1); }
inline  uint32x4_t  vcombine(const uint32x2_t  & v0, const uint32x2_t  & v1) { return vcombine_u32(v0, v1); }
inline   int32x4_t  vcombine(const int32x2_t   & v0, const int32x2_t   & v1) { return vcombine_s32(v0, v1); }
inline float32x4_t  vcombine(const float32x2_t & v0, const float32x2_t & v1) { return vcombine_f32(v0, v1); }

////////////////////////////// vaddq ///////////////////////

inline  uint8x16_t vaddq(const uint8x16_t  & v0, const uint8x16_t  & v1) { return vaddq_u8 (v0, v1); }
inline   int8x16_t vaddq(const int8x16_t   & v0, const int8x16_t   & v1) { return vaddq_s8 (v0, v1); }
inline  uint16x8_t vaddq(const uint16x8_t  & v0, const uint16x8_t  & v1) { return vaddq_u16(v0, v1); }
inline   int16x8_t vaddq(const int16x8_t   & v0, const int16x8_t   & v1) { return vaddq_s16(v0, v1); }
inline  uint32x4_t vaddq(const uint32x4_t  & v0, const uint32x4_t  & v1) { return vaddq_u32(v0, v1); }
inline   int32x4_t vaddq(const int32x4_t   & v0, const int32x4_t   & v1) { return vaddq_s32(v0, v1); }
inline float32x4_t vaddq(const float32x4_t & v0, const float32x4_t & v1) { return vaddq_f32(v0, v1); }

////////////////////////////// vadd ///////////////////////

inline   uint8x8_t vadd(const uint8x8_t   & v0, const uint8x8_t   & v1) { return vadd_u8 (v0, v1); }
inline    int8x8_t vadd(const int8x8_t    & v0, const int8x8_t    & v1) { return vadd_s8 (v0, v1); }
inline  uint16x4_t vadd(const uint16x4_t  & v0, const uint16x4_t  & v1) { return vadd_u16(v0, v1); }
inline   int16x4_t vadd(const int16x4_t   & v0, const int16x4_t   & v1) { return vadd_s16(v0, v1); }
inline  uint32x2_t vadd(const uint32x2_t  & v0, const uint32x2_t  & v1) { return vadd_u32(v0, v1); }
inline   int32x2_t vadd(const int32x2_t   & v0, const int32x2_t   & v1) { return vadd_s32(v0, v1); }
inline float32x2_t vadd(const float32x2_t & v0, const float32x2_t & v1) { return vadd_f32(v0, v1); }

////////////////////////////// vqaddq ///////////////////////

inline  uint8x16_t vqaddq(const uint8x16_t  & v0, const uint8x16_t  & v1) { return vqaddq_u8 (v0, v1); }
inline   int8x16_t vqaddq(const int8x16_t   & v0, const int8x16_t   & v1) { return vqaddq_s8 (v0, v1); }
inline  uint16x8_t vqaddq(const uint16x8_t  & v0, const uint16x8_t  & v1) { return vqaddq_u16(v0, v1); }
inline   int16x8_t vqaddq(const int16x8_t   & v0, const int16x8_t   & v1) { return vqaddq_s16(v0, v1); }
inline  uint32x4_t vqaddq(const uint32x4_t  & v0, const uint32x4_t  & v1) { return vqaddq_u32(v0, v1); }
inline   int32x4_t vqaddq(const int32x4_t   & v0, const int32x4_t   & v1) { return vqaddq_s32(v0, v1); }

////////////////////////////// vqadd ///////////////////////

inline   uint8x8_t vqadd(const uint8x8_t   & v0, const uint8x8_t   & v1) { return vqadd_u8 (v0, v1); }
inline    int8x8_t vqadd(const int8x8_t    & v0, const int8x8_t    & v1) { return vqadd_s8 (v0, v1); }
inline  uint16x4_t vqadd(const uint16x4_t  & v0, const uint16x4_t  & v1) { return vqadd_u16(v0, v1); }
inline   int16x4_t vqadd(const int16x4_t   & v0, const int16x4_t   & v1) { return vqadd_s16(v0, v1); }
inline  uint32x2_t vqadd(const uint32x2_t  & v0, const uint32x2_t  & v1) { return vqadd_u32(v0, v1); }
inline   int32x2_t vqadd(const int32x2_t   & v0, const int32x2_t   & v1) { return vqadd_s32(v0, v1); }

////////////////////////////// vsubq ///////////////////////

inline  uint8x16_t vsubq(const uint8x16_t  & v0, const uint8x16_t  & v1) { return vsubq_u8 (v0, v1); }
inline   int8x16_t vsubq(const int8x16_t   & v0, const int8x16_t   & v1) { return vsubq_s8 (v0, v1); }
inline  uint16x8_t vsubq(const uint16x8_t  & v0, const uint16x8_t  & v1) { return vsubq_u16(v0, v1); }
inline   int16x8_t vsubq(const int16x8_t   & v0, const int16x8_t   & v1) { return vsubq_s16(v0, v1); }
inline  uint32x4_t vsubq(const uint32x4_t  & v0, const uint32x4_t  & v1) { return vsubq_u32(v0, v1); }
inline   int32x4_t vsubq(const int32x4_t   & v0, const int32x4_t   & v1) { return vsubq_s32(v0, v1); }
inline float32x4_t vsubq(const float32x4_t & v0, const float32x4_t & v1) { return vsubq_f32(v0, v1); }

////////////////////////////// vsub ///////////////////////

inline   uint8x8_t vsub(const uint8x8_t   & v0, const uint8x8_t   & v1) { return vsub_u8 (v0, v1); }
inline    int8x8_t vsub(const int8x8_t    & v0, const int8x8_t    & v1) { return vsub_s8 (v0, v1); }
inline  uint16x4_t vsub(const uint16x4_t  & v0, const uint16x4_t  & v1) { return vsub_u16(v0, v1); }
inline   int16x4_t vsub(const int16x4_t   & v0, const int16x4_t   & v1) { return vsub_s16(v0, v1); }
inline  uint32x2_t vsub(const uint32x2_t  & v0, const uint32x2_t  & v1) { return vsub_u32(v0, v1); }
inline   int32x2_t vsub(const int32x2_t   & v0, const int32x2_t   & v1) { return vsub_s32(v0, v1); }
inline float32x2_t vsub(const float32x2_t & v0, const float32x2_t & v1) { return vsub_f32(v0, v1); }

////////////////////////////// vqsubq ///////////////////////

inline  uint8x16_t vqsubq(const uint8x16_t  & v0, const uint8x16_t  & v1) { return vqsubq_u8 (v0, v1); }
inline   int8x16_t vqsubq(const int8x16_t   & v0, const int8x16_t   & v1) { return vqsubq_s8 (v0, v1); }
inline  uint16x8_t vqsubq(const uint16x8_t  & v0, const uint16x8_t  & v1) { return vqsubq_u16(v0, v1); }
inline   int16x8_t vqsubq(const int16x8_t   & v0, const int16x8_t   & v1) { return vqsubq_s16(v0, v1); }
inline  uint32x4_t vqsubq(const uint32x4_t  & v0, const uint32x4_t  & v1) { return vqsubq_u32(v0, v1); }
inline   int32x4_t vqsubq(const int32x4_t   & v0, const int32x4_t   & v1) { return vqsubq_s32(v0, v1); }
inline  uint64x2_t vqsubq(const uint64x2_t  & v0, const uint64x2_t  & v1) { return vqsubq_u64(v0, v1); }
inline   int64x2_t vqsubq(const int64x2_t   & v0, const int64x2_t   & v1) { return vqsubq_s64(v0, v1); }

////////////////////////////// vqsub ///////////////////////

inline   uint8x8_t vqsub(const uint8x8_t   & v0, const uint8x8_t   & v1) { return vqsub_u8 (v0, v1); }
inline    int8x8_t vqsub(const int8x8_t    & v0, const int8x8_t    & v1) { return vqsub_s8 (v0, v1); }
inline  uint16x4_t vqsub(const uint16x4_t  & v0, const uint16x4_t  & v1) { return vqsub_u16(v0, v1); }
inline   int16x4_t vqsub(const int16x4_t   & v0, const int16x4_t   & v1) { return vqsub_s16(v0, v1); }
inline  uint32x2_t vqsub(const uint32x2_t  & v0, const uint32x2_t  & v1) { return vqsub_u32(v0, v1); }
inline   int32x2_t vqsub(const int32x2_t   & v0, const int32x2_t   & v1) { return vqsub_s32(v0, v1); }
inline  uint64x1_t vqsub(const uint64x1_t  & v0, const uint64x1_t  & v1) { return vqsub_u64(v0, v1); }
inline   int64x1_t vqsub(const int64x1_t   & v0, const int64x1_t   & v1) { return vqsub_s64(v0, v1); }

////////////////////////////// vmull ///////////////////////

inline  uint16x8_t vmull(const uint8x8_t   & v0, const uint8x8_t   & v1) { return vmull_u8 (v0, v1); }
inline   int16x8_t vmull(const int8x8_t    & v0, const int8x8_t    & v1) { return vmull_s8 (v0, v1); }
inline  uint32x4_t vmull(const uint16x4_t  & v0, const uint16x4_t  & v1) { return vmull_u16(v0, v1); }
inline   int32x4_t vmull(const int16x4_t   & v0, const int16x4_t   & v1) { return vmull_s16(v0, v1); }
inline  uint64x2_t vmull(const uint32x2_t  & v0, const uint32x2_t  & v1) { return vmull_u32(v0, v1); }
inline   int64x2_t vmull(const int32x2_t   & v0, const int32x2_t   & v1) { return vmull_s32(v0, v1); }

////////////////////////////// vrev64q ///////////////////////

inline uint8x16_t  vrev64q(const uint8x16_t  & v) { return vrev64q_u8 (v); }
inline int8x16_t   vrev64q(const int8x16_t   & v) { return vrev64q_s8 (v); }
inline uint16x8_t  vrev64q(const uint16x8_t  & v) { return vrev64q_u16(v); }
inline int16x8_t   vrev64q(const int16x8_t   & v) { return vrev64q_s16(v); }
inline uint32x4_t  vrev64q(const uint32x4_t  & v) { return vrev64q_u32(v); }
inline int32x4_t   vrev64q(const int32x4_t   & v) { return vrev64q_s32(v); }
inline float32x4_t vrev64q(const float32x4_t & v) { return vrev64q_f32(v); }

////////////////////////////// vrev64 ///////////////////////

inline uint8x8_t   vrev64(const uint8x8_t   & v) { return vrev64_u8 (v); }
inline int8x8_t    vrev64(const int8x8_t    & v) { return vrev64_s8 (v); }
inline uint16x4_t  vrev64(const uint16x4_t  & v) { return vrev64_u16(v); }
inline int16x4_t   vrev64(const int16x4_t   & v) { return vrev64_s16(v); }
inline uint32x2_t  vrev64(const uint32x2_t  & v) { return vrev64_u32(v); }
inline int32x2_t   vrev64(const int32x2_t   & v) { return vrev64_s32(v); }
inline float32x2_t vrev64(const float32x2_t & v) { return vrev64_f32(v); }

////////////////////////////// vceqq ///////////////////////

inline  uint8x16_t vceqq(const uint8x16_t  & v0, const uint8x16_t  & v1) { return vceqq_u8 (v0, v1); }
inline  uint8x16_t vceqq(const int8x16_t   & v0, const int8x16_t   & v1) { return vceqq_s8 (v0, v1); }
inline  uint16x8_t vceqq(const uint16x8_t  & v0, const uint16x8_t  & v1) { return vceqq_u16(v0, v1); }
inline  uint16x8_t vceqq(const int16x8_t   & v0, const int16x8_t   & v1) { return vceqq_s16(v0, v1); }
inline  uint32x4_t vceqq(const uint32x4_t  & v0, const uint32x4_t  & v1) { return vceqq_u32(v0, v1); }
inline  uint32x4_t vceqq(const int32x4_t   & v0, const int32x4_t   & v1) { return vceqq_s32(v0, v1); }
inline  uint32x4_t vceqq(const float32x4_t & v0, const float32x4_t & v1) { return vceqq_f32(v0, v1); }

////////////////////////////// vceq ///////////////////////

inline   uint8x8_t vceq(const uint8x8_t   & v0, const uint8x8_t   & v1) { return vceq_u8 (v0, v1); }
inline   uint8x8_t vceq(const int8x8_t    & v0, const int8x8_t    & v1) { return vceq_s8 (v0, v1); }
inline  uint16x4_t vceq(const uint16x4_t  & v0, const uint16x4_t  & v1) { return vceq_u16(v0, v1); }
inline  uint16x4_t vceq(const int16x4_t   & v0, const int16x4_t   & v1) { return vceq_s16(v0, v1); }
inline  uint32x2_t vceq(const uint32x2_t  & v0, const uint32x2_t  & v1) { return vceq_u32(v0, v1); }
inline  uint32x2_t vceq(const int32x2_t   & v0, const int32x2_t   & v1) { return vceq_s32(v0, v1); }
inline  uint32x2_t vceq(const float32x2_t & v0, const float32x2_t & v1) { return vceq_f32(v0, v1); }

////////////////////////////// vcgtq ///////////////////////

inline  uint8x16_t vcgtq(const uint8x16_t  & v0, const uint8x16_t  & v1) { return vcgtq_u8 (v0, v1); }
inline  uint8x16_t vcgtq(const int8x16_t   & v0, const int8x16_t   & v1) { return vcgtq_s8 (v0, v1); }
inline  uint16x8_t vcgtq(const uint16x8_t  & v0, const uint16x8_t  & v1) { return vcgtq_u16(v0, v1); }
inline  uint16x8_t vcgtq(const int16x8_t   & v0, const int16x8_t   & v1) { return vcgtq_s16(v0, v1); }
inline  uint32x4_t vcgtq(const uint32x4_t  & v0, const uint32x4_t  & v1) { return vcgtq_u32(v0, v1); }
inline  uint32x4_t vcgtq(const int32x4_t   & v0, const int32x4_t   & v1) { return vcgtq_s32(v0, v1); }
inline  uint32x4_t vcgtq(const float32x4_t & v0, const float32x4_t & v1) { return vcgtq_f32(v0, v1); }

////////////////////////////// vcgt ///////////////////////

inline   uint8x8_t vcgt(const uint8x8_t   & v0, const uint8x8_t   & v1) { return vcgt_u8 (v0, v1); }
inline   uint8x8_t vcgt(const int8x8_t    & v0, const int8x8_t    & v1) { return vcgt_s8 (v0, v1); }
inline  uint16x4_t vcgt(const uint16x4_t  & v0, const uint16x4_t  & v1) { return vcgt_u16(v0, v1); }
inline  uint16x4_t vcgt(const int16x4_t   & v0, const int16x4_t   & v1) { return vcgt_s16(v0, v1); }
inline  uint32x2_t vcgt(const uint32x2_t  & v0, const uint32x2_t  & v1) { return vcgt_u32(v0, v1); }
inline  uint32x2_t vcgt(const int32x2_t   & v0, const int32x2_t   & v1) { return vcgt_s32(v0, v1); }
inline  uint32x2_t vcgt(const float32x2_t & v0, const float32x2_t & v1) { return vcgt_f32(v0, v1); }

////////////////////////////// vcgeq ///////////////////////

inline  uint8x16_t vcgeq(const uint8x16_t  & v0, const uint8x16_t  & v1) { return vcgeq_u8 (v0, v1); }
inline  uint8x16_t vcgeq(const int8x16_t   & v0, const int8x16_t   & v1) { return vcgeq_s8 (v0, v1); }
inline  uint16x8_t vcgeq(const uint16x8_t  & v0, const uint16x8_t  & v1) { return vcgeq_u16(v0, v1); }
inline  uint16x8_t vcgeq(const int16x8_t   & v0, const int16x8_t   & v1) { return vcgeq_s16(v0, v1); }
inline  uint32x4_t vcgeq(const uint32x4_t  & v0, const uint32x4_t  & v1) { return vcgeq_u32(v0, v1); }
inline  uint32x4_t vcgeq(const int32x4_t   & v0, const int32x4_t   & v1) { return vcgeq_s32(v0, v1); }
inline  uint32x4_t vcgeq(const float32x4_t & v0, const float32x4_t & v1) { return vcgeq_f32(v0, v1); }

////////////////////////////// vcge ///////////////////////

inline   uint8x8_t vcge(const uint8x8_t   & v0, const uint8x8_t   & v1) { return vcge_u8 (v0, v1); }
inline   uint8x8_t vcge(const int8x8_t    & v0, const int8x8_t    & v1) { return vcge_s8 (v0, v1); }
inline  uint16x4_t vcge(const uint16x4_t  & v0, const uint16x4_t  & v1) { return vcge_u16(v0, v1); }
inline  uint16x4_t vcge(const int16x4_t   & v0, const int16x4_t   & v1) { return vcge_s16(v0, v1); }
inline  uint32x2_t vcge(const uint32x2_t  & v0, const uint32x2_t  & v1) { return vcge_u32(v0, v1); }
inline  uint32x2_t vcge(const int32x2_t   & v0, const int32x2_t   & v1) { return vcge_s32(v0, v1); }
inline  uint32x2_t vcge(const float32x2_t & v0, const float32x2_t & v1) { return vcge_f32(v0, v1); }

////////////////////////////// vandq ///////////////////////

inline  uint8x16_t vandq(const uint8x16_t  & v0, const uint8x16_t  & v1) { return vandq_u8 (v0, v1); }
inline   int8x16_t vandq(const int8x16_t   & v0, const int8x16_t   & v1) { return vandq_s8 (v0, v1); }
inline  uint16x8_t vandq(const uint16x8_t  & v0, const uint16x8_t  & v1) { return vandq_u16(v0, v1); }
inline   int16x8_t vandq(const int16x8_t   & v0, const int16x8_t   & v1) { return vandq_s16(v0, v1); }
inline  uint32x4_t vandq(const uint32x4_t  & v0, const uint32x4_t  & v1) { return vandq_u32(v0, v1); }
inline   int32x4_t vandq(const int32x4_t   & v0, const int32x4_t   & v1) { return vandq_s32(v0, v1); }

////////////////////////////// vand ///////////////////////

inline   uint8x8_t vand(const uint8x8_t   & v0, const uint8x8_t   & v1) { return vand_u8 (v0, v1); }
inline    int8x8_t vand(const int8x8_t    & v0, const int8x8_t    & v1) { return vand_s8 (v0, v1); }
inline  uint16x4_t vand(const uint16x4_t  & v0, const uint16x4_t  & v1) { return vand_u16(v0, v1); }
inline   int16x4_t vand(const int16x4_t   & v0, const int16x4_t   & v1) { return vand_s16(v0, v1); }
inline  uint32x2_t vand(const uint32x2_t  & v0, const uint32x2_t  & v1) { return vand_u32(v0, v1); }
inline   int32x2_t vand(const int32x2_t   & v0, const int32x2_t   & v1) { return vand_s32(v0, v1); }

////////////////////////////// vmovn ///////////////////////

inline uint8x8_t   vmovn(const uint16x8_t  & v) { return vmovn_u16(v); }
inline int8x8_t    vmovn(const int16x8_t   & v) { return vmovn_s16(v); }
inline uint16x4_t  vmovn(const uint32x4_t  & v) { return vmovn_u32(v); }
inline int16x4_t   vmovn(const int32x4_t   & v) { return vmovn_s32(v); }
inline uint32x2_t  vmovn(const uint64x2_t  & v) { return vmovn_u64(v); }
inline int32x2_t   vmovn(const int64x2_t   & v) { return vmovn_s64(v); }

////////////////////////////// vqmovn ///////////////////////

inline uint8x8_t   vqmovn(const uint16x8_t  & v) { return vqmovn_u16(v); }
inline int8x8_t    vqmovn(const int16x8_t   & v) { return vqmovn_s16(v); }
inline uint16x4_t  vqmovn(const uint32x4_t  & v) { return vqmovn_u32(v); }
inline int16x4_t   vqmovn(const int32x4_t   & v) { return vqmovn_s32(v); }
inline uint32x2_t  vqmovn(const uint64x2_t  & v) { return vqmovn_u64(v); }
inline int32x2_t   vqmovn(const int64x2_t   & v) { return vqmovn_s64(v); }

////////////////////////////// vmovl ///////////////////////

inline uint16x8_t  vmovl(const uint8x8_t   & v) { return  vmovl_u8(v); }
inline int16x8_t   vmovl(const int8x8_t    & v) { return  vmovl_s8(v); }
inline uint32x4_t  vmovl(const uint16x4_t  & v) { return vmovl_u16(v); }
inline int32x4_t   vmovl(const int16x4_t   & v) { return vmovl_s16(v); }

////////////////////////////// vmvnq ///////////////////////

inline uint8x16_t  vmvnq(const uint8x16_t  & v) { return vmvnq_u8 (v); }
inline int8x16_t   vmvnq(const int8x16_t   & v) { return vmvnq_s8 (v); }
inline uint16x8_t  vmvnq(const uint16x8_t  & v) { return vmvnq_u16(v); }
inline int16x8_t   vmvnq(const int16x8_t   & v) { return vmvnq_s16(v); }
inline uint32x4_t  vmvnq(const uint32x4_t  & v) { return vmvnq_u32(v); }
inline int32x4_t   vmvnq(const int32x4_t   & v) { return vmvnq_s32(v); }

////////////////////////////// vmvn ///////////////////////

inline uint8x8_t   vmvn(const uint8x8_t   & v) { return vmvn_u8 (v); }
inline int8x8_t    vmvn(const int8x8_t    & v) { return vmvn_s8 (v); }
inline uint16x4_t  vmvn(const uint16x4_t  & v) { return vmvn_u16(v); }
inline int16x4_t   vmvn(const int16x4_t   & v) { return vmvn_s16(v); }
inline uint32x2_t  vmvn(const uint32x2_t  & v) { return vmvn_u32(v); }
inline int32x2_t   vmvn(const int32x2_t   & v) { return vmvn_s32(v); }

////////////////////////////// vbicq ///////////////////////

inline  uint8x16_t vbicq(const uint8x16_t  & v0, const uint8x16_t  & v1) { return vbicq_u8 (v0, v1); }
inline   int8x16_t vbicq(const int8x16_t   & v0, const int8x16_t   & v1) { return vbicq_s8 (v0, v1); }
inline  uint16x8_t vbicq(const uint16x8_t  & v0, const uint16x8_t  & v1) { return vbicq_u16(v0, v1); }
inline   int16x8_t vbicq(const int16x8_t   & v0, const int16x8_t   & v1) { return vbicq_s16(v0, v1); }
inline  uint32x4_t vbicq(const uint32x4_t  & v0, const uint32x4_t  & v1) { return vbicq_u32(v0, v1); }
inline   int32x4_t vbicq(const int32x4_t   & v0, const int32x4_t   & v1) { return vbicq_s32(v0, v1); }
inline  uint64x2_t vbicq(const uint64x2_t  & v0, const uint64x2_t  & v1) { return vbicq_u64(v0, v1); }
inline   int64x2_t vbicq(const int64x2_t   & v0, const int64x2_t   & v1) { return vbicq_s64(v0, v1); }

////////////////////////////// vbic ///////////////////////

inline   uint8x8_t vbic(const uint8x8_t   & v0, const uint8x8_t   & v1) { return vbic_u8 (v0, v1); }
inline    int8x8_t vbic(const int8x8_t    & v0, const int8x8_t    & v1) { return vbic_s8 (v0, v1); }
inline  uint16x4_t vbic(const uint16x4_t  & v0, const uint16x4_t  & v1) { return vbic_u16(v0, v1); }
inline   int16x4_t vbic(const int16x4_t   & v0, const int16x4_t   & v1) { return vbic_s16(v0, v1); }
inline  uint32x2_t vbic(const uint32x2_t  & v0, const uint32x2_t  & v1) { return vbic_u32(v0, v1); }
inline   int32x2_t vbic(const int32x2_t   & v0, const int32x2_t   & v1) { return vbic_s32(v0, v1); }
inline  uint64x1_t vbic(const uint64x1_t  & v0, const uint64x1_t  & v1) { return vbic_u64(v0, v1); }
inline   int64x1_t vbic(const int64x1_t   & v0, const int64x1_t   & v1) { return vbic_s64(v0, v1); }

////////////////////////////// vtransform ///////////////////////

template <typename Op>
void vtransform(Size2D size,
                const typename Op::type * src0Base, ptrdiff_t src0Stride,
                const typename Op::type * src1Base, ptrdiff_t src1Stride,
                typename Op::type * dstBase, ptrdiff_t dstStride, const Op & op)
{
    typedef typename Op::type type;
    typedef typename VecTraits<type>::vec128 vec128;
    typedef typename VecTraits<type>::vec64 vec64;

    if (src0Stride == src1Stride && src0Stride == dstStride &&
        src0Stride == (ptrdiff_t)(size.width * sizeof(type)))
    {
        size.width *= size.height;
        size.height = 1;
    }

    const size_t step_base = 32 / sizeof(type);
    size_t roiw_base = size.width >= (step_base - 1) ? size.width - step_base + 1 : 0;
    const size_t step_tail = 8 / sizeof(type);
    size_t roiw_tail = size.width >= (step_tail - 1) ? size.width - step_tail + 1 : 0;

    for (size_t y = 0; y < size.height; ++y)
    {
        const type * src0 = internal::getRowPtr(src0Base, src0Stride, y);
        const type * src1 = internal::getRowPtr(src1Base, src1Stride, y);
        typename Op::type * dst = internal::getRowPtr(dstBase, dstStride, y);
        size_t x = 0;

        for( ; x < roiw_base; x += step_base )
        {
            internal::prefetch(src0 + x);
            internal::prefetch(src1 + x);

            vec128 v_src00 = vld1q(src0 + x), v_src01 = vld1q(src0 + x + 16 / sizeof(type));
            vec128 v_src10 = vld1q(src1 + x), v_src11 = vld1q(src1 + x + 16 / sizeof(type));
            vec128 v_dst;

            op(v_src00, v_src10, v_dst);
            vst1q(dst + x, v_dst);

            op(v_src01, v_src11, v_dst);
            vst1q(dst + x + 16 / sizeof(type), v_dst);
        }
        for( ; x < roiw_tail; x += step_tail )
        {
            vec64 v_src0 = vld1(src0 + x);
            vec64 v_src1 = vld1(src1 + x);
            vec64 v_dst;

            op(v_src0, v_src1, v_dst);
            vst1(dst + x, v_dst);
        }

        for (; x < size.width; ++x)
        {
            op(src0 + x, src1 + x, dst + x);
        }
    }
}

} }

#endif // CAROTENE_NEON

#endif
