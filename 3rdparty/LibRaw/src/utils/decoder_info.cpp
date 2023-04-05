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
#include "../../internal/libraw_cxx_defs.h"

const char *LibRaw::unpack_function_name()
{
  libraw_decoder_info_t decoder_info;
  get_decoder_info(&decoder_info);
  return decoder_info.decoder_name;
}

int LibRaw::get_decoder_info(libraw_decoder_info_t *d_info)
{
  if (!d_info)
    return LIBRAW_UNSPECIFIED_ERROR;
  d_info->decoder_name = 0;
  d_info->decoder_flags = 0;
  if (!load_raw)
    return LIBRAW_OUT_OF_ORDER_CALL;

  // dcraw.c names order
  if (load_raw == &LibRaw::android_tight_load_raw)
  {
    d_info->decoder_name = "android_tight_load_raw()";
    d_info->decoder_flags = LIBRAW_DECODER_FIXEDMAXC;
  }
  else if (load_raw == &LibRaw::android_loose_load_raw)
  {
    d_info->decoder_name = "android_loose_load_raw()";
    d_info->decoder_flags = LIBRAW_DECODER_FIXEDMAXC;
  }
  else if (load_raw == &LibRaw::vc5_dng_load_raw_placeholder)
  {
      d_info->decoder_name = "vc5_dng_load_raw_placeholder()";
#ifndef USE_GPRSDK
    d_info->decoder_flags = LIBRAW_DECODER_UNSUPPORTED_FORMAT;
#endif
  }
  else if (load_raw == &LibRaw::canon_600_load_raw)
  {
    d_info->decoder_name = "canon_600_load_raw()";
    d_info->decoder_flags = LIBRAW_DECODER_FIXEDMAXC;
  }
  else if (load_raw == &LibRaw::fuji_compressed_load_raw)
  {
    d_info->decoder_name = "fuji_compressed_load_raw()";
  }
  else if (load_raw == &LibRaw::fuji_14bit_load_raw)
  {
    d_info->decoder_name = "fuji_14bit_load_raw()";
  }
  else if (load_raw == &LibRaw::canon_load_raw)
  {
    d_info->decoder_name = "canon_load_raw()";
  }
  else if (load_raw == &LibRaw::lossless_jpeg_load_raw)
  {
    d_info->decoder_name = "lossless_jpeg_load_raw()";
    d_info->decoder_flags =
        LIBRAW_DECODER_HASCURVE | LIBRAW_DECODER_TRYRAWSPEED | LIBRAW_DECODER_TRYRAWSPEED3;
  }
  else if (load_raw == &LibRaw::canon_sraw_load_raw)
  {
    d_info->decoder_name = "canon_sraw_load_raw()";
    d_info->decoder_flags = LIBRAW_DECODER_TRYRAWSPEED3;
  }
  else if (load_raw == &LibRaw::crxLoadRaw)
  {
    d_info->decoder_name = "crxLoadRaw()";
  }
  else if (load_raw == &LibRaw::lossless_dng_load_raw)
  {
    d_info->decoder_name = "lossless_dng_load_raw()";
    d_info->decoder_flags = LIBRAW_DECODER_HASCURVE |
                            LIBRAW_DECODER_TRYRAWSPEED | LIBRAW_DECODER_TRYRAWSPEED3 |
                            LIBRAW_DECODER_ADOBECOPYPIXEL;
  }
  else if (load_raw == &LibRaw::packed_dng_load_raw)
  {
    d_info->decoder_name = "packed_dng_load_raw()";
    d_info->decoder_flags = LIBRAW_DECODER_HASCURVE |
                            LIBRAW_DECODER_TRYRAWSPEED | LIBRAW_DECODER_TRYRAWSPEED3 |
                            LIBRAW_DECODER_ADOBECOPYPIXEL;
  }
  else if (load_raw == &LibRaw::pentax_load_raw)
  {
    d_info->decoder_name = "pentax_load_raw()";
    d_info->decoder_flags = LIBRAW_DECODER_TRYRAWSPEED | LIBRAW_DECODER_TRYRAWSPEED3;
  }
  else if (load_raw == &LibRaw::nikon_load_raw)
  {
    d_info->decoder_name = "nikon_load_raw()";
    d_info->decoder_flags =
        LIBRAW_DECODER_TRYRAWSPEED | LIBRAW_DECODER_HASCURVE | LIBRAW_DECODER_TRYRAWSPEED3;
  }
  else if (load_raw == &LibRaw::nikon_coolscan_load_raw)
  {
    d_info->decoder_name = "nikon_coolscan_load_raw()";
    d_info->decoder_flags = LIBRAW_DECODER_FIXEDMAXC;
  }
  else if (load_raw == &LibRaw::nikon_he_load_raw_placeholder)
  {
    d_info->decoder_name = "nikon_he_load_raw_placeholder()";
    d_info->decoder_flags = LIBRAW_DECODER_UNSUPPORTED_FORMAT;
  }
  else if (load_raw == &LibRaw::nikon_load_sraw)
  {
    d_info->decoder_name = "nikon_load_sraw()";
    d_info->decoder_flags = LIBRAW_DECODER_HASCURVE | LIBRAW_DECODER_FIXEDMAXC;
  }
  else if (load_raw == &LibRaw::nikon_yuv_load_raw)
  {
    d_info->decoder_name = "nikon_load_yuv_load_raw()";
    d_info->decoder_flags = LIBRAW_DECODER_HASCURVE | LIBRAW_DECODER_FIXEDMAXC;
  }
  else if (load_raw == &LibRaw::rollei_load_raw)
  {
    // UNTESTED
    d_info->decoder_name = "rollei_load_raw()";
    d_info->decoder_flags = LIBRAW_DECODER_FIXEDMAXC;
  }
  else if (load_raw == &LibRaw::phase_one_load_raw)
  {
    d_info->decoder_name = "phase_one_load_raw()";
  }
  else if (load_raw == &LibRaw::phase_one_load_raw_c)
  {
    d_info->decoder_name = "phase_one_load_raw_c()";
    d_info->decoder_flags = LIBRAW_DECODER_TRYRAWSPEED3; /* FIXME: need to make sure correction not applied*/
  }
  else if (load_raw == &LibRaw::phase_one_load_raw_s)
  {
    d_info->decoder_name = "phase_one_load_raw_s()";
  }
  else if (load_raw == &LibRaw::hasselblad_load_raw)
  {
    d_info->decoder_name = "hasselblad_load_raw()";
    d_info->decoder_flags = LIBRAW_DECODER_TRYRAWSPEED3; /* FIXME: need to make sure correction not applied*/
  }
  else if (load_raw == &LibRaw::leaf_hdr_load_raw)
  {
    d_info->decoder_name = "leaf_hdr_load_raw()";
  }
  else if (load_raw == &LibRaw::unpacked_load_raw)
  {
    d_info->decoder_name = "unpacked_load_raw()";
	d_info->decoder_flags = LIBRAW_DECODER_FLATDATA;
  }
  else if (load_raw == &LibRaw::unpacked_load_raw_reversed)
  {
    d_info->decoder_name = "unpacked_load_raw_reversed()";
    d_info->decoder_flags = LIBRAW_DECODER_FIXEDMAXC;
  }
  else if (load_raw == &LibRaw::sinar_4shot_load_raw)
  {
    // UNTESTED
    d_info->decoder_name = "sinar_4shot_load_raw()";
    d_info->decoder_flags = LIBRAW_DECODER_SINAR4SHOT;
  }
  else if (load_raw == &LibRaw::imacon_full_load_raw)
  {
    d_info->decoder_name = "imacon_full_load_raw()";
  }
  else if (load_raw == &LibRaw::hasselblad_full_load_raw)
  {
    d_info->decoder_name = "hasselblad_full_load_raw()";
  }
  else if (load_raw == &LibRaw::packed_load_raw)
  {
    d_info->decoder_name = "packed_load_raw()";
    d_info->decoder_flags = LIBRAW_DECODER_TRYRAWSPEED | LIBRAW_DECODER_TRYRAWSPEED3;
  }
  else if (load_raw == &LibRaw::broadcom_load_raw)
  {
    // UNTESTED
    d_info->decoder_name = "broadcom_load_raw()";
    d_info->decoder_flags = LIBRAW_DECODER_FIXEDMAXC;
  }
  else if (load_raw == &LibRaw::nokia_load_raw)
  {
    // UNTESTED
    d_info->decoder_name = "nokia_load_raw()";
    d_info->decoder_flags = LIBRAW_DECODER_FIXEDMAXC;
  }
#ifdef LIBRAW_OLD_VIDEO_SUPPORT
  else if (load_raw == &LibRaw::canon_rmf_load_raw)
  {
    // UNTESTED
    d_info->decoder_name = "canon_rmf_load_raw()";
  }
#endif
  else if (load_raw == &LibRaw::panasonic_load_raw)
  {
    d_info->decoder_name = "panasonic_load_raw()";
    d_info->decoder_flags = LIBRAW_DECODER_TRYRAWSPEED | LIBRAW_DECODER_TRYRAWSPEED;
  }
  else if (load_raw == &LibRaw::panasonicC6_load_raw)
  {
    d_info->decoder_name = "panasonicC6_load_raw()";
    /* FIXME: No rawspeed3:  not sure it handles 12-bit data too */
  }
  else if (load_raw == &LibRaw::panasonicC7_load_raw)
  {
    d_info->decoder_name = "panasonicC7_load_raw()";
  }
  else if (load_raw == &LibRaw::olympus_load_raw)
  {
    d_info->decoder_name = "olympus_load_raw()";
    d_info->decoder_flags = LIBRAW_DECODER_TRYRAWSPEED | LIBRAW_DECODER_TRYRAWSPEED3;;
  }
  else if (load_raw == &LibRaw::minolta_rd175_load_raw)
  {
    // UNTESTED
    d_info->decoder_name = "minolta_rd175_load_raw()";
  }
  else if (load_raw == &LibRaw::quicktake_100_load_raw)
  {
    // UNTESTED
    d_info->decoder_name = "quicktake_100_load_raw()";
  }
  else if (load_raw == &LibRaw::kodak_radc_load_raw)
  {
    d_info->decoder_name = "kodak_radc_load_raw()";
  }
  else if (load_raw == &LibRaw::kodak_jpeg_load_raw)
  {
    // UNTESTED + RBAYER
    d_info->decoder_name = "kodak_jpeg_load_raw()";
  }
  else if (load_raw == &LibRaw::lossy_dng_load_raw)
  {
    // Check rbayer
    d_info->decoder_name = "lossy_dng_load_raw()";
    d_info->decoder_flags =
        LIBRAW_DECODER_TRYRAWSPEED | LIBRAW_DECODER_HASCURVE;
  }
  else if (load_raw == &LibRaw::kodak_dc120_load_raw)
  {
    d_info->decoder_name = "kodak_dc120_load_raw()";
  }
  else if (load_raw == &LibRaw::eight_bit_load_raw)
  {
    d_info->decoder_name = "eight_bit_load_raw()";
    d_info->decoder_flags = LIBRAW_DECODER_HASCURVE | LIBRAW_DECODER_FIXEDMAXC;
  }
  else if (load_raw == &LibRaw::kodak_c330_load_raw)
  {
    d_info->decoder_name = "kodak_yrgb_load_raw()";
    d_info->decoder_flags = LIBRAW_DECODER_HASCURVE | LIBRAW_DECODER_FIXEDMAXC;
  }
  else if (load_raw == &LibRaw::kodak_c603_load_raw)
  {
    d_info->decoder_name = "kodak_yrgb_load_raw()";
    d_info->decoder_flags = LIBRAW_DECODER_HASCURVE | LIBRAW_DECODER_FIXEDMAXC;
  }
  else if (load_raw == &LibRaw::kodak_262_load_raw)
  {
    d_info->decoder_name = "kodak_262_load_raw()"; // UNTESTED!
    d_info->decoder_flags = LIBRAW_DECODER_HASCURVE | LIBRAW_DECODER_FIXEDMAXC;
  }
  else if (load_raw == &LibRaw::kodak_65000_load_raw)
  {
    d_info->decoder_name = "kodak_65000_load_raw()";
    d_info->decoder_flags = LIBRAW_DECODER_HASCURVE;
  }
  else if (load_raw == &LibRaw::kodak_ycbcr_load_raw)
  {
    // UNTESTED
    d_info->decoder_name = "kodak_ycbcr_load_raw()";
    d_info->decoder_flags = LIBRAW_DECODER_HASCURVE | LIBRAW_DECODER_FIXEDMAXC;
  }
  else if (load_raw == &LibRaw::kodak_rgb_load_raw)
  {
    // UNTESTED
    d_info->decoder_name = "kodak_rgb_load_raw()";
    d_info->decoder_flags = LIBRAW_DECODER_FIXEDMAXC;
  }
  else if (load_raw == &LibRaw::sony_load_raw)
  {
    d_info->decoder_name = "sony_load_raw()";
  }
  else if (load_raw == &LibRaw::sony_ljpeg_load_raw)
  {
    d_info->decoder_name = "sony_ljpeg_load_raw()";
  }
  else if (load_raw == &LibRaw::sony_arw_load_raw)
  {
    d_info->decoder_name = "sony_arw_load_raw()";
    d_info->decoder_flags = LIBRAW_DECODER_TRYRAWSPEED | LIBRAW_DECODER_TRYRAWSPEED3;
  }
  else if (load_raw == &LibRaw::sony_arw2_load_raw)
  {
    d_info->decoder_name = "sony_arw2_load_raw()";
    d_info->decoder_flags = LIBRAW_DECODER_HASCURVE |
                            LIBRAW_DECODER_TRYRAWSPEED | LIBRAW_DECODER_TRYRAWSPEED3 |
                            LIBRAW_DECODER_SONYARW2;
  }
  else if (load_raw == &LibRaw::sony_arq_load_raw)
  {
    d_info->decoder_name = "sony_arq_load_raw()";
    d_info->decoder_flags = LIBRAW_DECODER_LEGACY_WITH_MARGINS | LIBRAW_DECODER_FLATDATA | LIBRAW_DECODER_FLAT_BG2_SWAPPED;
  }
  else if (load_raw == &LibRaw::samsung_load_raw)
  {
    d_info->decoder_name = "samsung_load_raw()";
    d_info->decoder_flags = LIBRAW_DECODER_TRYRAWSPEED | LIBRAW_DECODER_TRYRAWSPEED3;
  }
  else if (load_raw == &LibRaw::samsung2_load_raw)
  {
    d_info->decoder_name = "samsung2_load_raw()";
  }
  else if (load_raw == &LibRaw::samsung3_load_raw)
  {
    d_info->decoder_name = "samsung3_load_raw()";
  }
  else if (load_raw == &LibRaw::smal_v6_load_raw)
  {
    // UNTESTED
    d_info->decoder_name = "smal_v6_load_raw()";
    d_info->decoder_flags = LIBRAW_DECODER_FIXEDMAXC;
  }
  else if (load_raw == &LibRaw::smal_v9_load_raw)
  {
    // UNTESTED
    d_info->decoder_name = "smal_v9_load_raw()";
    d_info->decoder_flags = LIBRAW_DECODER_FIXEDMAXC;
  }
#ifdef LIBRAW_OLD_VIDEO_SUPPORT
  else if (load_raw == &LibRaw::redcine_load_raw)
  {
    d_info->decoder_name = "redcine_load_raw()";
    d_info->decoder_flags = LIBRAW_DECODER_HASCURVE;
  }
#endif
  else if (load_raw == &LibRaw::x3f_load_raw)
  {
    d_info->decoder_name = "x3f_load_raw()";
    d_info->decoder_flags = LIBRAW_DECODER_OWNALLOC | LIBRAW_DECODER_FIXEDMAXC |
                            LIBRAW_DECODER_LEGACY_WITH_MARGINS;
  }
  else if (load_raw == &LibRaw::pentax_4shot_load_raw)
  {
    d_info->decoder_name = "pentax_4shot_load_raw()";
    d_info->decoder_flags = LIBRAW_DECODER_OWNALLOC;
  }
  else if (load_raw == &LibRaw::deflate_dng_load_raw)
  {
    d_info->decoder_name = "deflate_dng_load_raw()";
    d_info->decoder_flags = LIBRAW_DECODER_OWNALLOC;
  }
  else if (load_raw == &LibRaw::uncompressed_fp_dng_load_raw)
  {
    d_info->decoder_name = "uncompressed_fp_dng_load_raw()";
    d_info->decoder_flags = LIBRAW_DECODER_OWNALLOC;
  }
  else if (load_raw == &LibRaw::nikon_load_striped_packed_raw)
  {
    d_info->decoder_name = "nikon_load_striped_packed_raw()";
  }
  else if (load_raw == &LibRaw::nikon_load_padded_packed_raw)
  {
    d_info->decoder_name = "nikon_load_padded_packed_raw()";
  }
  else if (load_raw == &LibRaw::nikon_14bit_load_raw)
  {
    d_info->decoder_name = "nikon_14bit_load_raw()";
  }
  /* -- added 07/02/18 -- */
  else if (load_raw == &LibRaw::unpacked_load_raw_fuji_f700s20)
  {
    d_info->decoder_name = "unpacked_load_raw_fuji_f700s20()";
  }
  else if (load_raw == &LibRaw::unpacked_load_raw_FujiDBP)
  {
    d_info->decoder_name = "unpacked_load_raw_FujiDBP()";
  }
#ifdef USE_6BY9RPI
  else if (load_raw == &LibRaw::rpi_load_raw8)
  {
	d_info->decoder_name = "rpi_load_raw8";
  }
  else if (load_raw == &LibRaw::rpi_load_raw12)
  {
	d_info->decoder_name = "rpi_load_raw12";
  }
  else if (load_raw == &LibRaw::rpi_load_raw14)
  {
	d_info->decoder_name = "rpi_load_raw14";
  }
  else if (load_raw == &LibRaw::rpi_load_raw16)
  {
	d_info->decoder_name = "rpi_load_raw16";
  }
#endif
  else
  {
    d_info->decoder_name = "Unknown unpack function";
    d_info->decoder_flags = LIBRAW_DECODER_NOTSET;
  }
  return LIBRAW_SUCCESS;
}
