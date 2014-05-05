// Copyright 2013 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
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
