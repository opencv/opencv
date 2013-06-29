// Copyright 2012 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Image transforms and color space conversion methods for lossless decoder.
//
// Authors: Vikas Arora (vikaas.arora@gmail.com)
//          Jyrki Alakuijala (jyrki@google.com)

#ifndef WEBP_DSP_LOSSLESS_H_
#define WEBP_DSP_LOSSLESS_H_

#include "../webp/types.h"
#include "../webp/decode.h"

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

//------------------------------------------------------------------------------
// Image transforms.

struct VP8LTransform;  // Defined in dec/vp8li.h.

// Performs inverse transform of data given transform information, start and end
// rows. Transform will be applied to rows [row_start, row_end[.
// The *in and *out pointers refer to source and destination data respectively
// corresponding to the intermediate row (row_start).
void VP8LInverseTransform(const struct VP8LTransform* const transform,
                          int row_start, int row_end,
                          const uint32_t* const in, uint32_t* const out);

// Similar to the static method ColorIndexInverseTransform() that is part of
// lossless.c, but used only for alpha decoding. It takes uint8_t (rather than
// uint32_t) arguments for 'src' and 'dst'.
void VP8LColorIndexInverseTransformAlpha(
    const struct VP8LTransform* const transform, int y_start, int y_end,
    const uint8_t* src, uint8_t* dst);

// Subtracts green from blue and red channels.
void VP8LSubtractGreenFromBlueAndRed(uint32_t* argb_data, int num_pixs);

void VP8LResidualImage(int width, int height, int bits,
                       uint32_t* const argb, uint32_t* const argb_scratch,
                       uint32_t* const image);

void VP8LColorSpaceTransform(int width, int height, int bits, int step,
                             uint32_t* const argb, uint32_t* image);

//------------------------------------------------------------------------------
// Color space conversion.

// Converts from BGRA to other color spaces.
void VP8LConvertFromBGRA(const uint32_t* const in_data, int num_pixels,
                         WEBP_CSP_MODE out_colorspace, uint8_t* const rgba);

//------------------------------------------------------------------------------
// Misc methods.

// Computes sampled size of 'size' when sampling using 'sampling bits'.
static WEBP_INLINE uint32_t VP8LSubSampleSize(uint32_t size,
                                              uint32_t sampling_bits) {
  return (size + (1 << sampling_bits) - 1) >> sampling_bits;
}

// Faster logarithm for integers. Small values use a look-up table.
#define LOG_LOOKUP_IDX_MAX 256
extern const float kLog2Table[LOG_LOOKUP_IDX_MAX];
extern const float kSLog2Table[LOG_LOOKUP_IDX_MAX];
extern float VP8LFastLog2Slow(int v);
extern float VP8LFastSLog2Slow(int v);
static WEBP_INLINE float VP8LFastLog2(int v) {
  return (v < LOG_LOOKUP_IDX_MAX) ? kLog2Table[v] : VP8LFastLog2Slow(v);
}
// Fast calculation of v * log2(v) for integer input.
static WEBP_INLINE float VP8LFastSLog2(int v) {
  return (v < LOG_LOOKUP_IDX_MAX) ? kSLog2Table[v] : VP8LFastSLog2Slow(v);
}


// In-place difference of each component with mod 256.
static WEBP_INLINE uint32_t VP8LSubPixels(uint32_t a, uint32_t b) {
  const uint32_t alpha_and_green =
      0x00ff00ffu + (a & 0xff00ff00u) - (b & 0xff00ff00u);
  const uint32_t red_and_blue =
      0xff00ff00u + (a & 0x00ff00ffu) - (b & 0x00ff00ffu);
  return (alpha_and_green & 0xff00ff00u) | (red_and_blue & 0x00ff00ffu);
}

void VP8LBundleColorMap(const uint8_t* const row, int width,
                        int xbits, uint32_t* const dst);

//------------------------------------------------------------------------------

#if defined(__cplusplus) || defined(c_plusplus)
}    // extern "C"
#endif

#endif  // WEBP_DSP_LOSSLESS_H_
