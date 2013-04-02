// Copyright 2013 Google Inc. All Rights Reserved.
//
// This code is licensed under the same terms as WebM:
//  Software License Agreement:  http://www.webmproject.org/license/software/
//  Additional IP Rights Grant:  http://www.webmproject.org/license/additional/
// -----------------------------------------------------------------------------
//
// TODO(skal): implement gradient smoothing.
//
// Author: Skal (pascal.massimino@gmail.com)

#include "./quant_levels_dec.h"

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

int DequantizeLevels(uint8_t* const data, int width, int height) {
  if (data == NULL || width <= 0 || height <= 0) return 0;
  (void)data;
  (void)width;
  (void)height;
  return 1;
}

#if defined(__cplusplus) || defined(c_plusplus)
}    // extern "C"
#endif
