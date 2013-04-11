// Copyright 2013 Google Inc. All Rights Reserved.
//
// This code is licensed under the same terms as WebM:
//  Software License Agreement:  http://www.webmproject.org/license/software/
//  Additional IP Rights Grant:  http://www.webmproject.org/license/additional/
// -----------------------------------------------------------------------------
//
// Alpha plane de-quantization utility
//
// Author:  Vikas Arora (vikasa@google.com)

#ifndef WEBP_UTILS_QUANT_LEVELS_DEC_H_
#define WEBP_UTILS_QUANT_LEVELS_DEC_H_

#include "../webp/types.h"

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

// Apply post-processing to input 'data' of size 'width'x'height' assuming
// that the source was quantized to a reduced number of levels.
// Returns false in case of error (data is NULL, invalid parameters, ...).
int DequantizeLevels(uint8_t* const data, int width, int height);

#if defined(__cplusplus) || defined(c_plusplus)
}    // extern "C"
#endif

#endif  /* WEBP_UTILS_QUANT_LEVELS_DEC_H_ */
