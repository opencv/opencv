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
#ifdef USE_RAWSPEED3
#include "rawspeed3_capi.h"
#endif

static void cleargps(libraw_gps_info_t *q)
{
  for (int i = 0; i < 3; i++)
    q->latitude[i] = q->longitude[i] = q->gpstimestamp[i] = 0.f;
  q->altitude = 0.f;
  q->altref = q->latref = q->longref = q->gpsstatus = q->gpsparsed = 0;
}

LibRaw::LibRaw(unsigned int flags) : memmgr(1024)
{
  double aber[4] = {1, 1, 1, 1};
  double gamm[6] = {0.45, 4.5, 0, 0, 0, 0};
  unsigned greybox[4] = {0, 0, UINT_MAX, UINT_MAX};
  unsigned cropbox[4] = {0, 0, UINT_MAX, UINT_MAX};
  ZERO(imgdata);

  cleargps(&imgdata.other.parsed_gps);
  ZERO(libraw_internal_data);
  ZERO(callbacks);

  _rawspeed_camerameta = _rawspeed_decoder = NULL;
  _rawspeed3_handle = NULL;
  dnghost = NULL;
  dngnegative = NULL;
  dngimage = NULL;
  _x3f_data = NULL;

#ifdef USE_RAWSPEED
  CameraMetaDataLR *camerameta =
      make_camera_metadata(); // May be NULL in case of exception in
                              // make_camera_metadata()
  _rawspeed_camerameta = static_cast<void *>(camerameta);
#endif
  callbacks.data_cb = (flags & LIBRAW_OPTIONS_NO_DATAERR_CALLBACK)
                          ? NULL
                          : &default_data_callback;
  callbacks.exif_cb = NULL; // no default callback
  callbacks.pre_identify_cb = NULL;
  callbacks.post_identify_cb = NULL;
  callbacks.pre_subtractblack_cb = callbacks.pre_scalecolors_cb =
      callbacks.pre_preinterpolate_cb = callbacks.pre_interpolate_cb =
          callbacks.interpolate_bayer_cb = callbacks.interpolate_xtrans_cb =
              callbacks.post_interpolate_cb = callbacks.pre_converttorgb_cb =
                  callbacks.post_converttorgb_cb = NULL;

  memmove(&imgdata.params.aber, &aber, sizeof(aber));
  memmove(&imgdata.params.gamm, &gamm, sizeof(gamm));
  memmove(&imgdata.params.greybox, &greybox, sizeof(greybox));
  memmove(&imgdata.params.cropbox, &cropbox, sizeof(cropbox));

  imgdata.params.bright = 1;
  imgdata.params.use_camera_matrix = 1;
  imgdata.params.user_flip = -1;
  imgdata.params.user_black = -1;
  imgdata.params.user_cblack[0] = imgdata.params.user_cblack[1] =
      imgdata.params.user_cblack[2] = imgdata.params.user_cblack[3] = -1000001;
  imgdata.params.user_sat = -1;
  imgdata.params.user_qual = -1;
  imgdata.params.output_color = 1;
  imgdata.params.output_bps = 8;
  imgdata.params.use_fuji_rotate = 1;
  imgdata.params.exp_shift = 1.0;
  imgdata.params.auto_bright_thr = LIBRAW_DEFAULT_AUTO_BRIGHTNESS_THRESHOLD;
  imgdata.params.adjust_maximum_thr = LIBRAW_DEFAULT_ADJUST_MAXIMUM_THRESHOLD;
  imgdata.rawparams.use_rawspeed = 1;
  imgdata.rawparams.use_dngsdk = LIBRAW_DNG_DEFAULT;
  imgdata.params.no_auto_scale = 0;
  imgdata.params.no_interpolation = 0;
  imgdata.rawparams.specials = 0; /* was inverted : LIBRAW_PROCESSING_DP2Q_INTERPOLATERG |      LIBRAW_PROCESSING_DP2Q_INTERPOLATEAF; */
  imgdata.rawparams.options = LIBRAW_RAWOPTIONS_CONVERTFLOAT_TO_INT;
  imgdata.rawparams.sony_arw2_posterization_thr = 0;
  imgdata.rawparams.max_raw_memory_mb = LIBRAW_MAX_ALLOC_MB_DEFAULT;
  imgdata.params.green_matching = 0;
  imgdata.rawparams.custom_camera_strings = 0;
  imgdata.rawparams.coolscan_nef_gamma = 1.0f;
  imgdata.parent_class = this;
  imgdata.progress_flags = 0;
  imgdata.color.dng_levels.baseline_exposure = -999.f;
  imgdata.color.dng_levels.LinearResponseLimit = 1.0f;
  MN.hasselblad.nIFD_CM[0] =
    MN.hasselblad.nIFD_CM[1] = -1;
  MN.kodak.ISOCalibrationGain = 1.0f;
  _exitflag = 0;
  tls = new LibRaw_TLS;
  tls->init();
}

LibRaw::~LibRaw()
{
  recycle();
  delete tls;
#ifdef USE_RAWSPEED3
  if (_rawspeed3_handle)
      rawspeed3_close(_rawspeed3_handle);
  _rawspeed3_handle = NULL;
#endif

#ifdef USE_RAWSPEED
  if (_rawspeed_camerameta)
  {
    CameraMetaDataLR *cmeta =
        static_cast<CameraMetaDataLR *>(_rawspeed_camerameta);
    delete cmeta;
    _rawspeed_camerameta = NULL;
  }
#endif
}

void x3f_clear(void *);

void LibRaw::recycle()
{
  recycle_datastream();
#define FREE(a)                                                                \
  do                                                                           \
  {                                                                            \
    if (a)                                                                     \
    {                                                                          \
      free(a);                                                                 \
      a = NULL;                                                                \
    }                                                                          \
  } while (0)

  FREE(imgdata.image);

  // explicit cleanup of afdata allocations; entire array is zeroed below
  for (int i = 0; i < LIBRAW_AFDATA_MAXCOUNT; i++)
      FREE(MN.common.afdata[i].AFInfoData);

  FREE(imgdata.thumbnail.thumb);
  FREE(libraw_internal_data.internal_data.meta_data);
  FREE(libraw_internal_data.output_data.histogram);
  FREE(libraw_internal_data.output_data.oprof);
  FREE(imgdata.color.profile);
  FREE(imgdata.rawdata.ph1_cblack);
  FREE(imgdata.rawdata.ph1_rblack);
  FREE(imgdata.rawdata.raw_alloc);
  FREE(imgdata.idata.xmpdata);

  parseCR3_Free();

#undef FREE

  ZERO(imgdata.sizes);
  imgdata.sizes.raw_inset_crops[0].cleft = imgdata.sizes.raw_inset_crops[1].cleft = 0xffff;
  imgdata.sizes.raw_inset_crops[0].ctop  = imgdata.sizes.raw_inset_crops[1].ctop = 0xffff;

  ZERO(imgdata.idata);
  ZERO(imgdata.color);
  ZERO(imgdata.lens);
  ZERO(imgdata.other);
  ZERO(imgdata.rawdata);
  ZERO(imgdata.shootinginfo);
  ZERO(imgdata.thumbnail);
  ZERO(imgdata.thumbs_list);
  ZERO(MN);
  cleargps(&imgdata.other.parsed_gps);
  ZERO(libraw_internal_data);

  imgdata.lens.makernotes.FocalUnits = 1;
  imgdata.lens.makernotes.LensID = LIBRAW_LENS_NOT_SET;
  imgdata.shootinginfo.DriveMode = -1;
  imgdata.shootinginfo.FocusMode = -1;
  imgdata.shootinginfo.MeteringMode = -1;
  imgdata.shootinginfo.AFPoint = -1;
  imgdata.shootinginfo.ExposureMode = -1;
  imgdata.shootinginfo.ExposureProgram = -1;
  imgdata.shootinginfo.ImageStabilization = -1;

  imgdata.color.dng_levels.baseline_exposure = -999.f;
  imgdata.color.dng_levels.LinearResponseLimit = 1.f;
  imgdata.color.dng_color[0].illuminant =
      imgdata.color.dng_color[1].illuminant = LIBRAW_WBI_None;
  for (int i = 0; i < 4; i++) imgdata.color.dng_levels.analogbalance[i] = 1.0f;

  MN.canon.DefaultCropAbsolute.l = -1;
  MN.canon.DefaultCropAbsolute.t = -1;
  MN.canon.AutoLightingOptimizer =  3; // 'off' value

  MN.fuji.WB_Preset = 0xffff;
  MN.fuji.ExpoMidPointShift = -999.f;
  MN.fuji.DynamicRange = 0xffff;
  MN.fuji.FilmMode = 0xffff;
  MN.fuji.DynamicRangeSetting = 0xffff;
  MN.fuji.DevelopmentDynamicRange = 0xffff;
  MN.fuji.AutoDynamicRange = 0xffff;
  MN.fuji.DRangePriority = 0xffff;
  MN.fuji.FocusMode = 0xffff;
  MN.fuji.AFMode = 0xffff;
  MN.fuji.FocusPixel[0] = MN.fuji.FocusPixel[1] = 0xffff;
  MN.fuji.FocusSettings = 0xffffffff;
  MN.fuji.AF_C_Settings = 0xffffffff;
  MN.fuji.FocusWarning = 0xffff;
  for (int i = 0; i < 3; i++) MN.fuji.ImageStabilization[i] = 0xffff;
  MN.fuji.DriveMode = -1;
  MN.fuji.ImageCount = -1;
  MN.fuji.AutoBracketing = -1;
  MN.fuji.SequenceNumber = -1;
  MN.fuji.SeriesLength = -1;
  MN.fuji.PixelShiftOffset[0] = MN.fuji.PixelShiftOffset[1] = -999.f;

  MN.hasselblad.nIFD_CM[0] = MN.hasselblad.nIFD_CM[1] = -1;

  MN.kodak.BlackLevelTop = 0xffff;
  MN.kodak.BlackLevelBottom = 0xffff;
  MN.kodak.ISOCalibrationGain = 1.0f;

  MN.nikon.SensorHighSpeedCrop.cleft = 0xffff;
  MN.nikon.SensorHighSpeedCrop.ctop = 0xffff;

  MN.olympus.FocusMode[0] = 0xffff;
  MN.olympus.AutoFocus    = 0xffff;
  MN.olympus.AFPoint      = 0xffff;
  MN.olympus.AFResult     = 0xffff;
  MN.olympus.AFFineTune   = 0xff;
  for (int i = 0; i < 3; i++) {
    MN.olympus.AFFineTuneAdj[i] = -32768;
    MN.olympus.SpecialMode[i] = 0xffffffff;
  }
  MN.olympus.ZoomStepCount = 0xffff;
  MN.olympus.FocusStepCount = 0xffff;
  MN.olympus.FocusStepInfinity = 0xffff;
  MN.olympus.FocusStepNear = 0xffff;
  MN.olympus.FocusDistance = -999.0;
  for (int i = 0; i < 4; i++) MN.olympus.AspectFrame[i] = 0xffff;
  MN.olympus.StackedImage[0] = 0xffffffff;

  MN.panasonic.LensManufacturer = 0xffffffff;

  MN.pentax.FocusMode[0] =
    MN.pentax.FocusMode[1] = 0xffff;
  MN.pentax.AFPointSelected[1] = 0xffff;
  MN.pentax.AFPointSelected_Area = 0xffff;
  MN.pentax.AFPointsInFocus = 0xffffffff;
  MN.pentax.AFPointMode = 0xff;

  MN.ricoh.AFStatus = 0xffff;
  MN.ricoh.AFAreaMode = 0xffff;
  MN.ricoh.WideAdapter = 0xffff;
  MN.ricoh.CropMode = 0xffff;
  MN.ricoh.NDFilter = 0xffff;
  MN.ricoh.AutoBracketing = 0xffff;
  MN.ricoh.MacroMode = 0xffff;
  MN.ricoh.FlashMode = 0xffff;
  MN.ricoh.FlashExposureComp = -999.0;
  MN.ricoh.ManualFlashOutput = -999.0;

  MN.samsung.ColorSpace[0] = MN.samsung.ColorSpace[1] = -1;

  MN.sony.CameraType = LIBRAW_SONY_CameraType_UNKNOWN;
  MN.sony.group2010 = 0;
  MN.sony.real_iso_offset = 0xffff;
  MN.sony.ImageCount3_offset = 0xffff;
  MN.sony.MeteringMode_offset = 0xffff;
  MN.sony.ExposureProgram_offset = 0xffff;
  MN.sony.ReleaseMode2_offset = 0xffff;
  MN.sony.ElectronicFrontCurtainShutter = 0xffffffff;
  MN.sony.MinoltaCamID = 0xffffffff;
  MN.sony.RAWFileType = 0xffff;
  MN.sony.AFAreaModeSetting = 0xff;
  MN.sony.AFAreaMode = 0xffff;
  MN.sony.FlexibleSpotPosition[0] =
      MN.sony.FlexibleSpotPosition[1] = 0xffff;
  MN.sony.AFPointSelected = MN.sony.AFPointSelected_0x201e = 0xff;
  MN.sony.AFTracking = 0xff;
  MN.sony.FocusPosition = 0xffff;
  MN.sony.LongExposureNoiseReduction = 0xffffffff;
  MN.sony.Quality = 0xffffffff;
  MN.sony.HighISONoiseReduction = 0xffff;
  MN.sony.SonyRawFileType = 0xffff;
  MN.sony.RawSizeType = 0xffff;
  MN.sony.AFMicroAdjValue = 0x7f;
  MN.sony.AFMicroAdjOn = -1;
  MN.sony.AFMicroAdjRegisteredLenses = 0xff;

  _exitflag = 0;
#ifdef USE_RAWSPEED
  if (_rawspeed_decoder)
  {
    RawSpeed::RawDecoder *d =
        static_cast<RawSpeed::RawDecoder *>(_rawspeed_decoder);
    delete d;
  }
  _rawspeed_decoder = 0;
#endif
#ifdef USE_RAWSPEED3
  if (_rawspeed3_handle)
      rawspeed3_release(_rawspeed3_handle);
#endif
#ifdef USE_DNGSDK
  if (dngnegative)
  {
    dng_negative *ng = (dng_negative *)dngnegative;
    delete ng;
    dngnegative = 0;
  }
  if(dngimage)
  {
      dng_image *dimage = (dng_image*)dngimage;
      delete dimage;
      dngimage = 0;
  }
#endif
#ifdef USE_X3FTOOLS
  if (_x3f_data)
  {
    x3f_clear(_x3f_data);
    _x3f_data = 0;
  }
#endif
  memmgr.cleanup();

  imgdata.thumbnail.tformat = LIBRAW_THUMBNAIL_UNKNOWN;
  libraw_internal_data.unpacker_data.thumb_format = LIBRAW_INTERNAL_THUMBNAIL_UNKNOWN;
  imgdata.progress_flags = 0;

  load_raw =  0;

  tls->init();
}
