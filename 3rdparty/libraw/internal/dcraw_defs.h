/* -*- C++ -*-
 * Copyright 2019-2021 LibRaw LLC (info@libraw.org)
 *

 LibRaw is free software; you can redistribute it and/or modify
 it under the terms of the one of two licenses as you choose:

1. GNU LESSER GENERAL PUBLIC LICENSE version 2.1
   (See file LICENSE.LGPL provided in LibRaw distribution archive for details).

2. COMMON DEVELOPMENT AND DISTRIBUTION LICENSE (CDDL) Version 1.0
   (See file LICENSE.CDDL provided in LibRaw distribution archive for details).

 */

#ifndef DCRAW_DEFS_H
#define DCRAW_DEFS_H

#include <math.h>
#define LIBRAW_LIBRARY_BUILD
#define LIBRAW_IO_REDEFINED
#include "libraw/libraw.h"
#include "libraw/libraw_types.h"
#include "internal/defines.h"
#include "internal/var_defines.h"

#define stmread(buf, maxlen, fp) stread(buf, MIN(maxlen, sizeof(buf)), fp)
#define strbuflen(buf) strnlen(buf, sizeof(buf) - 1)
#define makeIs(idx) (maker_index == idx)
#define strnXcat(buf, string)                                                  \
  strncat(buf, string, LIM(sizeof(buf) - strbuflen(buf) - 1, 0, sizeof(buf)))

// DNG was written by:
#define nonDNG    0
#define CameraDNG 1
#define AdobeDNG  2

// Makernote tag type:
#define is_0x927c 0 /* most cameras */
#define is_0xc634 2 /* Adobe DNG, Sony SR2, Pentax */

// abbreviations
#define ilm imgdata.lens.makernotes
#define icWBC imgdata.color.WB_Coeffs
#define icWBCCTC imgdata.color.WBCT_Coeffs
#define imCanon imgdata.makernotes.canon
#define imFuji imgdata.makernotes.fuji
#define imHassy imgdata.makernotes.hasselblad
#define imKodak imgdata.makernotes.kodak
#define imNikon imgdata.makernotes.nikon
#define imOly imgdata.makernotes.olympus
#define imPana imgdata.makernotes.panasonic
#define imPentax imgdata.makernotes.pentax
#define imPhaseOne imgdata.makernotes.phaseone
#define imRicoh imgdata.makernotes.ricoh
#define imSamsung imgdata.makernotes.samsung
#define imSony imgdata.makernotes.sony
#define imCommon imgdata.makernotes.common


#define ph1_bits(n) ph1_bithuff(n, 0)
#define ph1_huff(h) ph1_bithuff(*h, h + 1)
#define getbits(n)  getbithuff(n, 0)
#define gethuff(h)  getbithuff(*h, h + 1)

#endif
