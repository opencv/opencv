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

int LibRaw::dcraw_process(void)
{
  int quality, i;

  int iterations = -1, dcb_enhance = 1, noiserd = 0;
  float preser = 0;
  float expos = 1.0;

  CHECK_ORDER_LOW(LIBRAW_PROGRESS_LOAD_RAW);
  //    CHECK_ORDER_HIGH(LIBRAW_PROGRESS_PRE_INTERPOLATE);

  try
  {

    int no_crop = 1;

    if (~O.cropbox[2] && ~O.cropbox[3])
      no_crop = 0;

    libraw_decoder_info_t di;
    get_decoder_info(&di);

    bool is_bayer = (imgdata.idata.filters || P1.colors == 1);
    int subtract_inline =
        !O.bad_pixels && !O.dark_frame && is_bayer && !IO.zero_is_bad;

    int rc = raw2image_ex(subtract_inline); // allocate imgdata.image and copy data!
	if (rc != LIBRAW_SUCCESS)
		return rc;

    // Adjust sizes

    int save_4color = O.four_color_rgb;

    if (IO.zero_is_bad)
    {
      remove_zeroes();
      SET_PROC_FLAG(LIBRAW_PROGRESS_REMOVE_ZEROES);
    }

    if (O.bad_pixels && no_crop)
    {
      bad_pixels(O.bad_pixels);
      SET_PROC_FLAG(LIBRAW_PROGRESS_BAD_PIXELS);
    }

    if (O.dark_frame && no_crop)
    {
      subtract(O.dark_frame);
      SET_PROC_FLAG(LIBRAW_PROGRESS_DARK_FRAME);
    }
    /* pre subtract black callback: check for it above to disable subtract
     * inline */

    if (callbacks.pre_subtractblack_cb)
      (callbacks.pre_subtractblack_cb)(this);

    quality = 2 + !IO.fuji_width;

    if (O.user_qual >= 0)
      quality = O.user_qual;

    if (!subtract_inline || !C.data_maximum)
    {
      adjust_bl();
      subtract_black_internal();
    }

    if (!(di.decoder_flags & LIBRAW_DECODER_FIXEDMAXC))
      adjust_maximum();

    if (O.user_sat > 0)
      C.maximum = O.user_sat;

    if (P1.is_foveon)
    {
      if (load_raw == &LibRaw::x3f_load_raw)
      {
        // Filter out zeroes
        for (int q = 0; q < S.height * S.width; q++)
        {
          for (int c = 0; c < 4; c++)
            if ((short)imgdata.image[q][c] < 0)
              imgdata.image[q][c] = 0;
        }
      }
      SET_PROC_FLAG(LIBRAW_PROGRESS_FOVEON_INTERPOLATE);
    }

    if (O.green_matching && !O.half_size)
    {
      green_matching();
    }

    if (callbacks.pre_scalecolors_cb)
      (callbacks.pre_scalecolors_cb)(this);

    if (!O.no_auto_scale)
    {
      scale_colors();
      SET_PROC_FLAG(LIBRAW_PROGRESS_SCALE_COLORS);
    }

    if (callbacks.pre_preinterpolate_cb)
      (callbacks.pre_preinterpolate_cb)(this);

    pre_interpolate();

    SET_PROC_FLAG(LIBRAW_PROGRESS_PRE_INTERPOLATE);

    if (O.dcb_iterations >= 0)
      iterations = O.dcb_iterations;
    if (O.dcb_enhance_fl >= 0)
      dcb_enhance = O.dcb_enhance_fl;
    if (O.fbdd_noiserd >= 0)
      noiserd = O.fbdd_noiserd;

    /* pre-exposure correction callback */

    if (O.exp_correc > 0)
    {
      expos = O.exp_shift;
      preser = O.exp_preser;
      exp_bef(expos, preser);
    }

    if (callbacks.pre_interpolate_cb)
      (callbacks.pre_interpolate_cb)(this);

    /* post-exposure correction fallback */
    if (P1.filters && !O.no_interpolation)
    {
      if (noiserd > 0 && P1.colors == 3 && P1.filters > 1000)
        fbdd(noiserd);

      if (P1.filters > 1000 && callbacks.interpolate_bayer_cb)
        (callbacks.interpolate_bayer_cb)(this);
      else if (P1.filters == 9 && callbacks.interpolate_xtrans_cb)
        (callbacks.interpolate_xtrans_cb)(this);
      else if (quality == 0)
        lin_interpolate();
      else if (quality == 1 || P1.colors > 3)
        vng_interpolate();
      else if (quality == 2 && P1.filters > 1000)
        ppg_interpolate();
      else if (P1.filters == LIBRAW_XTRANS)
      {
        // Fuji X-Trans
        xtrans_interpolate(quality > 2 ? 3 : 1);
      }
      else if (quality == 3)
        ahd_interpolate(); // really don't need it here due to fallback op
      else if (quality == 4)
        dcb(iterations, dcb_enhance);

      else if (quality == 11)
        dht_interpolate();
      else if (quality == 12)
        aahd_interpolate();
      // fallback to AHD
      else
      {
        ahd_interpolate();
        imgdata.process_warnings |= LIBRAW_WARN_FALLBACK_TO_AHD;
      }

      SET_PROC_FLAG(LIBRAW_PROGRESS_INTERPOLATE);
    }
    if (IO.mix_green)
    {
      for (P1.colors = 3, i = 0; i < S.height * S.width; i++)
        imgdata.image[i][1] = (imgdata.image[i][1] + imgdata.image[i][3]) >> 1;
      SET_PROC_FLAG(LIBRAW_PROGRESS_MIX_GREEN);
    }

    if (callbacks.post_interpolate_cb)
      (callbacks.post_interpolate_cb)(this);
    else if (!P1.is_foveon && P1.colors == 3 && O.med_passes > 0)
    {
      median_filter();
      SET_PROC_FLAG(LIBRAW_PROGRESS_MEDIAN_FILTER);
    }

    if (O.highlight == 2)
    {
      blend_highlights();
      SET_PROC_FLAG(LIBRAW_PROGRESS_HIGHLIGHTS);
    }

    if (O.highlight > 2)
    {
      recover_highlights();
      SET_PROC_FLAG(LIBRAW_PROGRESS_HIGHLIGHTS);
    }

    if (O.use_fuji_rotate)
    {
      fuji_rotate();
      SET_PROC_FLAG(LIBRAW_PROGRESS_FUJI_ROTATE);
    }

    if (!libraw_internal_data.output_data.histogram)
    {
      libraw_internal_data.output_data.histogram =
          (int(*)[LIBRAW_HISTOGRAM_SIZE])malloc(
              sizeof(*libraw_internal_data.output_data.histogram) * 4);
    }
#ifndef NO_LCMS
    if (O.camera_profile)
    {
      apply_profile(O.camera_profile, O.output_profile);
      SET_PROC_FLAG(LIBRAW_PROGRESS_APPLY_PROFILE);
    }
#endif

    if (callbacks.pre_converttorgb_cb)
      (callbacks.pre_converttorgb_cb)(this);

    convert_to_rgb();
    SET_PROC_FLAG(LIBRAW_PROGRESS_CONVERT_RGB);

    if (callbacks.post_converttorgb_cb)
      (callbacks.post_converttorgb_cb)(this);

    if (O.use_fuji_rotate)
    {
      stretch();
      SET_PROC_FLAG(LIBRAW_PROGRESS_STRETCH);
    }
    O.four_color_rgb = save_4color; // also, restore

    return 0;
  }
  catch (const std::bad_alloc&)
  {
      recycle();
      return LIBRAW_UNSUFFICIENT_MEMORY;
  }
  catch (const LibRaw_exceptions& err)
  {
    EXCEPTION_HANDLER(err);
  }
}
