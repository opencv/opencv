// Copyright 2011 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Alpha-plane decompression.
//
// Author: Skal (pascal.massimino@gmail.com)

#include <stdlib.h>
#include "./vp8i.h"
#include "./vp8li.h"
#include "../utils/filters.h"
#include "../utils/quant_levels_dec.h"
#include "../webp/format_constants.h"

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

//------------------------------------------------------------------------------
// Decodes the compressed data 'data' of size 'data_size' into the 'output'.
// The 'output' buffer should be pre-allocated and must be of the same
// dimension 'height'x'width', as that of the image.
//
// Returns 1 on successfully decoding the compressed alpha and
//         0 if either:
//           error in bit-stream header (invalid compression mode or filter), or
//           error returned by appropriate compression method.

static int DecodeAlpha(const uint8_t* data, size_t data_size,
                       int width, int height, uint8_t* output) {
  WEBP_FILTER_TYPE filter;
  int pre_processing;
  int rsrv;
  int ok = 0;
  int method;
  const uint8_t* const alpha_data = data + ALPHA_HEADER_LEN;
  const size_t alpha_data_size = data_size - ALPHA_HEADER_LEN;

  assert(width > 0 && height > 0);
  assert(data != NULL && output != NULL);

  if (data_size <= ALPHA_HEADER_LEN) {
    return 0;
  }

  method = (data[0] >> 0) & 0x03;
  filter = (data[0] >> 2) & 0x03;
  pre_processing = (data[0] >> 4) & 0x03;
  rsrv = (data[0] >> 6) & 0x03;
  if (method < ALPHA_NO_COMPRESSION ||
      method > ALPHA_LOSSLESS_COMPRESSION ||
      filter >= WEBP_FILTER_LAST ||
      pre_processing > ALPHA_PREPROCESSED_LEVELS ||
      rsrv != 0) {
    return 0;
  }

  if (method == ALPHA_NO_COMPRESSION) {
    const size_t alpha_decoded_size = height * width;
    ok = (alpha_data_size >= alpha_decoded_size);
    if (ok) memcpy(output, alpha_data, alpha_decoded_size);
  } else {
    ok = VP8LDecodeAlphaImageStream(width, height, alpha_data, alpha_data_size,
                                    output);
  }

  if (ok) {
    WebPUnfilterFunc unfilter_func = WebPUnfilters[filter];
    if (unfilter_func != NULL) {
      // TODO(vikas): Implement on-the-fly decoding & filter mechanism to decode
      // and apply filter per image-row.
      unfilter_func(width, height, width, output);
    }
    if (pre_processing == ALPHA_PREPROCESSED_LEVELS) {
      ok = DequantizeLevels(output, width, height);
    }
  }

  return ok;
}

//------------------------------------------------------------------------------

const uint8_t* VP8DecompressAlphaRows(VP8Decoder* const dec,
                                      int row, int num_rows) {
  const int width = dec->pic_hdr_.width_;
  const int height = dec->pic_hdr_.height_;

  if (row < 0 || num_rows < 0 || row + num_rows > height) {
    return NULL;    // sanity check.
  }

  if (row == 0) {
    // Decode everything during the first call.
    assert(!dec->is_alpha_decoded_);
    if (!DecodeAlpha(dec->alpha_data_, (size_t)dec->alpha_data_size_,
                     width, height, dec->alpha_plane_)) {
      return NULL;  // Error.
    }
    dec->is_alpha_decoded_ = 1;
  }

  // Return a pointer to the current decoded row.
  return dec->alpha_plane_ + row * width;
}

#if defined(__cplusplus) || defined(c_plusplus)
}    // extern "C"
#endif
