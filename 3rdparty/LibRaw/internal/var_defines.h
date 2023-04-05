/* -*- C++ -*-
 * File: var_defines.h
 * Copyright 2008-2021 LibRaw LLC (info@libraw.org)
 * Created: Sat Mar  8, 2008
 *
 * LibRaw redefinitions of dcraw internal variables

LibRaw is free software; you can redistribute it and/or modify
it under the terms of the one of two licenses as you choose:

1. GNU LESSER GENERAL PUBLIC LICENSE version 2.1
   (See file LICENSE.LGPL provided in LibRaw distribution archive for details).

2. COMMON DEVELOPMENT AND DISTRIBUTION LICENSE (CDDL) Version 1.0
   (See file LICENSE.CDDL provided in LibRaw distribution archive for details).

 */

#ifndef VAR_DEFINES_H
#define VAR_DEFINES_H


// imgdata.idata
#define make              (imgdata.idata.make)
#define model             (imgdata.idata.model)
#define software          (imgdata.idata.software)
#define is_raw            (imgdata.idata.raw_count)
#define dng_version       (imgdata.idata.dng_version)
#define is_foveon         (imgdata.idata.is_foveon)
#define colors            (imgdata.idata.colors)
#define cdesc             (imgdata.idata.cdesc)
#define filters           (imgdata.idata.filters)
#define xtrans            (imgdata.idata.xtrans)
#define xtrans_abs        (imgdata.idata.xtrans_abs)
#define xmpdata           (imgdata.idata.xmpdata)
#define xmplen            (imgdata.idata.xmplen)
//imgdata image
#define image             (imgdata.image)
#define raw_image         (imgdata.rawdata.raw_image)
#define color_image       (imgdata.rawdata.color_image)
#define normalized_make   (imgdata.idata.normalized_make)
#define normalized_model  (imgdata.idata.normalized_model)
#define maker_index       (imgdata.idata.maker_index)

// imgdata.sizes
#define raw_height        (imgdata.sizes.raw_height)
#define raw_width         (imgdata.sizes.raw_width)
#define raw_pitch         (imgdata.sizes.raw_pitch)
#define height            (imgdata.sizes.height)
#define width             (imgdata.sizes.width)
#define top_margin        (imgdata.sizes.top_margin)
#define left_margin       (imgdata.sizes.left_margin)
#define bottom_margin     (imgdata.sizes.bottom_margin)
#define right_margin      (imgdata.sizes.right_margin)
#define iheight           (imgdata.sizes.iheight)
#define iwidth            (imgdata.sizes.iwidth)
#define pixel_aspect      (imgdata.sizes.pixel_aspect)
#define flip              (imgdata.sizes.flip)
#define mask              (imgdata.sizes.mask)
#define raw_stride        (libraw_internal_data.unpacker_data.raw_stride)

//imgdata.color
#define white             (imgdata.color.white)
#define cam_mul           (imgdata.color.cam_mul)
#define pre_mul           (imgdata.color.pre_mul)
#define cmatrix           (imgdata.color.cmatrix)
#define rgb_cam           (imgdata.color.rgb_cam)
#ifndef SRC_USES_CURVE
#define curve             (imgdata.color.curve)
#endif
#ifndef SRC_USES_BLACK
#define black             (imgdata.color.black)
#define cblack            (imgdata.color.cblack)
#endif
#define maximum           (imgdata.color.maximum)
#define channel_maximum   (imgdata.color.channel_maximum)
#define profile_length    (imgdata.color.profile_length)
#define color_flags       (imgdata.color.color_flags)
#define ph1               (imgdata.color.phase_one_data)
#define flash_used        (imgdata.color.flash_used)
#define canon_ev          (imgdata.color.canon_ev)
#define model2            (imgdata.color.model2)

//imgdata.thumbnail
#define thumb_width       (imgdata.thumbnail.twidth)
#define thumb_height      (imgdata.thumbnail.theight)
#define thumb_length      (imgdata.thumbnail.tlength)


//imgdata.others
#define iso_speed         (imgdata.other.iso_speed)
#define shutter           (imgdata.other.shutter)
#define aperture          (imgdata.other.aperture)
#define focal_len         (imgdata.other.focal_len)
#define timestamp         (imgdata.other.timestamp)
#define shot_order        (imgdata.other.shot_order)
#define gpsdata           (imgdata.other.gpsdata)
#define desc              (imgdata.other.desc)
#define artist            (imgdata.other.artist)

#define FujiCropMode      (imgdata.makernotes.fuji.CropMode)

//imgdata.output
#define greybox           (imgdata.params.greybox)
#define cropbox           (imgdata.params.cropbox)
#define aber              (imgdata.params.aber)
#define gamm              (imgdata.params.gamm)
#define user_mul          (imgdata.params.user_mul)
#define shot_select       (imgdata.rawparams.shot_select)
#define bright            (imgdata.params.bright)
#define threshold         (imgdata.params.threshold)
#define half_size         (imgdata.params.half_size)
#define four_color_rgb    (imgdata.params.four_color_rgb)
#define highlight         (imgdata.params.highlight)
#define use_auto_wb       (imgdata.params.use_auto_wb)
#define use_camera_wb     (imgdata.params.use_camera_wb)
#define use_camera_matrix (imgdata.params.use_camera_matrix)
#define output_color      (imgdata.params.output_color)
#define output_bps        (imgdata.params.output_bps)
#define gamma_16bit       (imgdata.params.gamma_16bit)
#define output_tiff       (imgdata.params.output_tiff)
#define med_passes        (imgdata.params.med_passes)
#define no_auto_bright    (imgdata.params.no_auto_bright)
#define auto_bright_thr   (imgdata.params.auto_bright_thr)
#define use_fuji_rotate   (imgdata.params.use_fuji_rotate)
#define filtering_mode    (imgdata.params.filtering_mode)

// DCB
#define dcb_iterations    (imgdata.params.iterations)
#define dcb_enhance_fl    (imgdata.params.dcb_enhance)
#define fbdd_noiserd      (imgdata.params.fbdd_noiserd)

//libraw_internal_data.internal_data
#define meta_data         (libraw_internal_data.internal_data.meta_data)
#define ifp               libraw_internal_data.internal_data.input
#define ifname            ((char*)libraw_internal_data.internal_data.input->fname())
#define ofp               libraw_internal_data.internal_data.output
#define profile_offset    (libraw_internal_data.internal_data.profile_offset)
#define thumb_offset      (libraw_internal_data.internal_data.toffset)
#define pana_black        (libraw_internal_data.internal_data.pana_black)

//libraw_internal_data.internal_output_params
#define mix_green         (libraw_internal_data.internal_output_params.mix_green)
#define raw_color         (libraw_internal_data.internal_output_params.raw_color)
#define use_gamma         (libraw_internal_data.internal_output_params.use_gamma)
#define zero_is_bad       (libraw_internal_data.internal_output_params.zero_is_bad)
#ifndef SRC_USES_SHRINK
#define shrink            (libraw_internal_data.internal_output_params.shrink)
#endif
#define fuji_width        (libraw_internal_data.internal_output_params.fuji_width)
#define thumb_format	  (libraw_internal_data.unpacker_data.thumb_format)

//libraw_internal_data.output_data
#define histogram         (libraw_internal_data.output_data.histogram)
#define oprof             (libraw_internal_data.output_data.oprof)

//libraw_internal_data.identify_data
#define exif_cfa          (libraw_internal_data.identify_data.olympus_exif_cfa)
#define unique_id         (libraw_internal_data.identify_data.unique_id)
#define OlyID             (libraw_internal_data.identify_data.OlyID)
#define tiff_nifds        (libraw_internal_data.identify_data.tiff_nifds)
#define tiff_flip         (libraw_internal_data.identify_data.tiff_flip)
#define metadata_blocks   (libraw_internal_data.identify_data.metadata_blocks)

//libraw_internal_data.unpacker_data
#define order             (libraw_internal_data.unpacker_data.order)
#define data_error        (libraw_internal_data.unpacker_data.data_error)
#define cr2_slice         (libraw_internal_data.unpacker_data.cr2_slice)
#define sraw_mul          (libraw_internal_data.unpacker_data.sraw_mul)
#define kodak_cbpp        (libraw_internal_data.unpacker_data.kodak_cbpp)
#define strip_offset      (libraw_internal_data.unpacker_data.strip_offset)
#define data_offset       (libraw_internal_data.unpacker_data.data_offset)
#define data_size         (libraw_internal_data.unpacker_data.data_size)
#define meta_offset       (libraw_internal_data.unpacker_data.meta_offset)
#define meta_length       (libraw_internal_data.unpacker_data.meta_length)
#define thumb_misc        (libraw_internal_data.unpacker_data.thumb_misc)
#define fuji_layout       (libraw_internal_data.unpacker_data.fuji_layout)
#define tiff_samples      (libraw_internal_data.unpacker_data.tiff_samples)
#define tiff_bps          (libraw_internal_data.unpacker_data.tiff_bps)
#define tiff_compress     (libraw_internal_data.unpacker_data.tiff_compress)
#define tiff_sampleformat (libraw_internal_data.unpacker_data.tiff_sampleformat)
#define zero_after_ff     (libraw_internal_data.unpacker_data.zero_after_ff)
#define tile_width        (libraw_internal_data.unpacker_data.tile_width)
#define tile_length       (libraw_internal_data.unpacker_data.tile_length)
#define load_flags        (libraw_internal_data.unpacker_data.load_flags)
#define pana_encoding     (libraw_internal_data.unpacker_data.pana_encoding)
#define pana_bpp          (libraw_internal_data.unpacker_data.pana_bpp)
#define CM_found          (libraw_internal_data.unpacker_data.CM_found)

#define is_NikonTransfer  (libraw_internal_data.unpacker_data.is_NikonTransfer)
#define is_Olympus        (libraw_internal_data.unpacker_data.is_Olympus)
#define OlympusDNG_SubDirOffsetValid (libraw_internal_data.unpacker_data.OlympusDNG_SubDirOffsetValid)
#define is_Sony           (libraw_internal_data.unpacker_data.is_Sony)
#define is_PentaxRicohMakernotes    (libraw_internal_data.unpacker_data.is_PentaxRicohMakernotes)
#define is_pana_raw       (libraw_internal_data.unpacker_data.is_pana_raw)


#ifdef LIBRAW_IO_REDEFINED
#define fread(ptr,size,n,stream)   stream->read(ptr,size,n)
#define fseek(stream,o,w)          stream->seek(o,w)
#define fseeko(stream,o,w)         stream->seek(o,w)
#define ftell(stream)              stream->tell()
#define ftello(stream)             stream->tell()
#define feof(stream)               stream->eof()
#ifdef getc
#undef getc
#endif
#define getc(stream)               stream->get_char()
#define fgetc(stream)              stream->get_char()
#define fgetcb(stream)             stream->get_char_buf()
#define fgets(str,n,stream)        stream->gets(str,n)
#define fscanf(stream,fmt,ptr)     stream->scanf_one(fmt,ptr)
#endif

#endif
