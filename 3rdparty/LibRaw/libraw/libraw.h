/* -*- C++ -*-
 * File: libraw.h
 * Copyright 2008-2021 LibRaw LLC (info@libraw.org)
 * Created: Sat Mar  8, 2008
 *
 * LibRaw C++ interface
 *

LibRaw is free software; you can redistribute it and/or modify
it under the terms of the one of two licenses as you choose:

1. GNU LESSER GENERAL PUBLIC LICENSE version 2.1
   (See file LICENSE.LGPL provided in LibRaw distribution archive for details).

2. COMMON DEVELOPMENT AND DISTRIBUTION LICENSE (CDDL) Version 1.0
   (See file LICENSE.CDDL provided in LibRaw distribution archive for details).

*/

#ifndef _LIBRAW_CLASS_H
#define _LIBRAW_CLASS_H

#ifdef __linux__
#define _FILE_OFFSET_BITS 64
#endif

// Enable use old cinema cameras if USE_OLD_VIDEOCAMS defined
#ifdef USE_OLD_VIDEOCAMS
#define LIBRAW_OLD_VIDEO_SUPPORT
#endif

#ifndef LIBRAW_USE_DEPRECATED_IOSTREAMS_DATASTREAM
#define LIBRAW_NO_IOSTREAMS_DATASTREAM
#endif

#ifndef LIBRAW_NO_IOSTREAMS_DATASTREAM
/* maximum file size to use LibRaw_file_datastream (fully buffered) I/O */
#define LIBRAW_USE_STREAMS_DATASTREAM_MAXSIZE (250 * 1024L * 1024L)
#endif

#include <limits.h>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* better WIN32 defines */

/* better WIN32 defines */

#if defined(WIN32) || defined(_WIN32)

/* Win32 API */
#  ifndef LIBRAW_WIN32_CALLS
#   define LIBRAW_WIN32_CALLS
#  endif

/* DLLs: Microsoft or Intel compiler */
# if defined(_MSC_VER) || defined(__INTEL_COMPILER)
# ifndef LIBRAW_WIN32_DLLDEFS
#  define LIBRAW_WIN32_DLLDEFS
# endif
#endif

/* wchar_t* API for std::filebuf */
# if (defined(_MSC_VER)  && (_MSC_VER > 1310)) || (defined(__INTEL_COMPILER) && (__INTEL_COMPILER >= 910))
#  ifndef LIBRAW_WIN32_UNICODEPATHS
#   define LIBRAW_WIN32_UNICODEPATHS
#  endif
# elif _GLIBCXX_HAVE__WFOPEN && _GLIBCXX_USE_WCHAR_T
#  ifndef LIBRAW_WIN32_UNICODEPATHS
#    define LIBRAW_WIN32_UNICODEPATHS
#  endif
# elif defined(_LIBCPP_HAS_OPEN_WITH_WCHAR)
#  ifndef LIBRAW_WIN32_UNICODEPATHS
#    define LIBRAW_WIN32_UNICODEPATHS
#  endif
# endif

#endif

#include "libraw_datastream.h"
#include "libraw_types.h"
#include "libraw_const.h"
#include "libraw_internal.h"
#include "libraw_alloc.h"

#ifdef __cplusplus
extern "C"
{
#endif
  DllDef const char *libraw_strerror(int errorcode);
  DllDef const char *libraw_strprogress(enum LibRaw_progress);
  /* LibRaw C API */
  DllDef libraw_data_t *libraw_init(unsigned int flags);
  DllDef int libraw_open_file(libraw_data_t *, const char *);
#ifndef LIBRAW_NO_IOSTREAMS_DATASTREAM
  DllDef int libraw_open_file_ex(libraw_data_t *, const char *,
                                 INT64 max_buff_sz);
#endif
#if defined(_WIN32) || defined(WIN32)
  DllDef int libraw_open_wfile(libraw_data_t *, const wchar_t *);
#ifndef LIBRAW_NO_IOSTREAMS_DATASTREAM
  DllDef int libraw_open_wfile_ex(libraw_data_t *, const wchar_t *,
                                  INT64 max_buff_sz);
#endif
#endif

  DllDef int libraw_open_buffer(libraw_data_t *, const void *buffer, size_t size);
  DllDef int libraw_open_bayer(libraw_data_t *lr, unsigned char *data,
                               unsigned datalen, ushort _raw_width,
                               ushort _raw_height, ushort _left_margin,
                               ushort _top_margin, ushort _right_margin,
                               ushort _bottom_margin, unsigned char procflags,
                               unsigned char bayer_battern,
                               unsigned unused_bits, unsigned otherflags,
                               unsigned black_level);
  DllDef int libraw_unpack(libraw_data_t *);
  DllDef int libraw_unpack_thumb(libraw_data_t *);
  DllDef int libraw_unpack_thumb_ex(libraw_data_t *,int);
  DllDef void libraw_recycle_datastream(libraw_data_t *);
  DllDef void libraw_recycle(libraw_data_t *);
  DllDef void libraw_close(libraw_data_t *);
  DllDef void libraw_subtract_black(libraw_data_t *);
  DllDef int libraw_raw2image(libraw_data_t *);
  DllDef void libraw_free_image(libraw_data_t *);
  /* version helpers */
  DllDef const char *libraw_version();
  DllDef int libraw_versionNumber();
  /* Camera list */
  DllDef const char **libraw_cameraList();
  DllDef int libraw_cameraCount();

  /* helpers */
  DllDef void libraw_set_exifparser_handler(libraw_data_t *,
                                            exif_parser_callback cb,
                                            void *datap);
  DllDef void libraw_set_dataerror_handler(libraw_data_t *, data_callback func,
                                           void *datap);
  DllDef void libraw_set_progress_handler(libraw_data_t *, progress_callback cb,
                                          void *datap);
  DllDef const char *libraw_unpack_function_name(libraw_data_t *lr);
  DllDef int libraw_get_decoder_info(libraw_data_t *lr,
                                     libraw_decoder_info_t *d);
  DllDef int libraw_COLOR(libraw_data_t *, int row, int col);
  DllDef unsigned libraw_capabilities();

  /* DCRAW compatibility */
  DllDef int libraw_adjust_sizes_info_only(libraw_data_t *);
  DllDef int libraw_dcraw_ppm_tiff_writer(libraw_data_t *lr,
                                          const char *filename);
  DllDef int libraw_dcraw_thumb_writer(libraw_data_t *lr, const char *fname);
  DllDef int libraw_dcraw_process(libraw_data_t *lr);
  DllDef libraw_processed_image_t *
  libraw_dcraw_make_mem_image(libraw_data_t *lr, int *errc);
  DllDef libraw_processed_image_t *
  libraw_dcraw_make_mem_thumb(libraw_data_t *lr, int *errc);
  DllDef void libraw_dcraw_clear_mem(libraw_processed_image_t *);
  /* getters/setters used by 3DLut Creator */
  DllDef void libraw_set_demosaic(libraw_data_t *lr, int value);
  DllDef void libraw_set_output_color(libraw_data_t *lr, int value);
  DllDef void libraw_set_adjust_maximum_thr(libraw_data_t *lr, float value);
  DllDef void libraw_set_user_mul(libraw_data_t *lr, int index, float val);
  DllDef void libraw_set_output_bps(libraw_data_t *lr, int value);
  DllDef void libraw_set_gamma(libraw_data_t *lr, int index, float value);
  DllDef void libraw_set_no_auto_bright(libraw_data_t *lr, int value);
  DllDef void libraw_set_bright(libraw_data_t *lr, float value);
  DllDef void libraw_set_highlight(libraw_data_t *lr, int value);
  DllDef void libraw_set_fbdd_noiserd(libraw_data_t *lr, int value);
  DllDef int libraw_get_raw_height(libraw_data_t *lr);
  DllDef int libraw_get_raw_width(libraw_data_t *lr);
  DllDef int libraw_get_iheight(libraw_data_t *lr);
  DllDef int libraw_get_iwidth(libraw_data_t *lr);
  DllDef float libraw_get_cam_mul(libraw_data_t *lr, int index);
  DllDef float libraw_get_pre_mul(libraw_data_t *lr, int index);
  DllDef float libraw_get_rgb_cam(libraw_data_t *lr, int index1, int index2);
  DllDef int libraw_get_color_maximum(libraw_data_t *lr);
  DllDef void libraw_set_output_tif(libraw_data_t *lr, int value);
  DllDef libraw_iparams_t *libraw_get_iparams(libraw_data_t *lr);
  DllDef libraw_lensinfo_t *libraw_get_lensinfo(libraw_data_t *lr);
  DllDef libraw_imgother_t *libraw_get_imgother(libraw_data_t *lr);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

class DllDef LibRaw
{
public:
  libraw_data_t imgdata;

  LibRaw(unsigned int flags = LIBRAW_OPTIONS_NONE);
  libraw_output_params_t *output_params_ptr() { return &imgdata.params; }
#ifndef LIBRAW_NO_IOSTREAMS_DATASTREAM
  int open_file(const char *fname,
                INT64 max_buffered_sz = LIBRAW_USE_STREAMS_DATASTREAM_MAXSIZE);
#if defined(_WIN32) || defined(WIN32)
  int open_file(const wchar_t *fname,
                INT64 max_buffered_sz = LIBRAW_USE_STREAMS_DATASTREAM_MAXSIZE);
#endif
#else
  int open_file(const char *fname);
#if defined(_WIN32) || defined(WIN32)
  int open_file(const wchar_t *fname);
#endif

#endif
  int open_buffer(const void *buffer, size_t size);
  virtual int open_datastream(LibRaw_abstract_datastream *);
  virtual int open_bayer(const unsigned char *data, unsigned datalen,
                         ushort _raw_width, ushort _raw_height,
                         ushort _left_margin, ushort _top_margin,
                         ushort _right_margin, ushort _bottom_margin,
                         unsigned char procflags, unsigned char bayer_pattern,
                         unsigned unused_bits, unsigned otherflags,
                         unsigned black_level);
  int error_count() { return libraw_internal_data.unpacker_data.data_error; }
  void recycle_datastream();
  int unpack(void);
  int unpack_thumb(void);
  int unpack_thumb_ex(int);
  int thumbOK(INT64 maxsz = -1);
  int adjust_sizes_info_only(void);
  int subtract_black();
  int subtract_black_internal();
  int raw2image();
  int raw2image_ex(int do_subtract_black);
  void raw2image_start();
  void free_image();
  int adjust_maximum();
  int adjust_to_raw_inset_crop(unsigned mask, float maxcrop = 0.55f); 
  void set_exifparser_handler(exif_parser_callback cb, void *data)
  {
    callbacks.exifparser_data = data;
    callbacks.exif_cb = cb;
  }
  void set_dataerror_handler(data_callback func, void *data)
  {
    callbacks.datacb_data = data;
    callbacks.data_cb = func;
  }
  void set_progress_handler(progress_callback pcb, void *data)
  {
    callbacks.progresscb_data = data;
    callbacks.progress_cb = pcb;
  }

  static const char* cameramakeridx2maker(unsigned maker);
  int setMakeFromIndex(unsigned index);

  void convertFloatToInt(float dmin = 4096.f, float dmax = 32767.f,
                         float dtarget = 16383.f);
  /* helpers */
  static unsigned capabilities();
  static const char *version();
  static int versionNumber();
  static const char **cameraList();
  static int cameraCount();
  static const char *strprogress(enum LibRaw_progress);
  static const char *strerror(int p);
  /* dcraw emulation */
  int dcraw_ppm_tiff_writer(const char *filename);
  int dcraw_thumb_writer(const char *fname);
  int dcraw_process(void);
  /* information calls */
  int is_fuji_rotated()
  {
    return libraw_internal_data.internal_output_params.fuji_width;
  }
  int is_sraw();
  int sraw_midpoint();
  int is_nikon_sraw();
  int is_coolscan_nef();
  int is_jpeg_thumb();
  int is_floating_point();
  int have_fpdata();
  /* memory writers */
  virtual libraw_processed_image_t *dcraw_make_mem_image(int *errcode = NULL);
  virtual libraw_processed_image_t *dcraw_make_mem_thumb(int *errcode = NULL);
  static void dcraw_clear_mem(libraw_processed_image_t *);

  /* Additional calls for make_mem_image */
  void get_mem_image_format(int *width, int *height, int *colors,
                            int *bps) const;
  int copy_mem_image(void *scan0, int stride, int bgr);

  /* free all internal data structures */
  void recycle();
  virtual ~LibRaw(void);

  int COLOR(int row, int col)
  {
    if (!imgdata.idata.filters)
      return 6; /* Special value 0+1+2+3 */
    if (imgdata.idata.filters < 1000)
      return fcol(row, col);
    return libraw_internal_data.internal_output_params.fuji_width
               ? FCF(row, col)
               : FC(row, col);
  }

  int FC(int row, int col)
  {
    return (imgdata.idata.filters >> (((row << 1 & 14) | (col & 1)) << 1) & 3);
  }
  int fcol(int row, int col);

  const char *unpack_function_name();
  virtual int get_decoder_info(libraw_decoder_info_t *d_info);
  libraw_internal_data_t *get_internal_data_pointer()
  {
    return &libraw_internal_data;
  }

  static float powf_lim(float a, float b, float limup)
  {
    return (b > limup || b < -limup) ? 0.f : powf(a, b);
  }
  static float libraw_powf64l(float a, float b) { return powf_lim(a, b, 64.f); }

  static unsigned sgetn(int n, uchar *s)
  {
    unsigned result = 0;
    while (n-- > 0)
      result = (result << 8) | (*s++);
    return result;
  }

  /* Phase one correction/subtractBL calls */
  /* Returns libraw error code */

  int phase_one_subtract_black(ushort *src, ushort *dest);
  int phase_one_correct();

  int set_rawspeed_camerafile(char *filename);
  virtual void setCancelFlag();
  virtual void clearCancelFlag();
  virtual int adobe_coeff(unsigned, const char *, int internal_only = 0);

  void set_dng_host(void *);

protected:
  static void *memmem(char *haystack, size_t haystacklen, char *needle,
                      size_t needlelen);
  static char *strcasestr(char *h, const char *n);
  static size_t strnlen(const char *s, size_t n);

#ifdef LIBRAW_NO_IOSTREAMS_DATASTREAM
  int libraw_openfile_tail(LibRaw_abstract_datastream *stream);
#endif

  int is_curve_linear();
  void checkCancel();
  void cam_xyz_coeff(float _rgb_cam[3][4], double cam_xyz[4][3]);
  void phase_one_allocate_tempbuffer();
  void phase_one_free_tempbuffer();
  virtual int is_phaseone_compressed();
  virtual int is_canon_600();
  /* Hotspots */
  virtual void copy_fuji_uncropped(unsigned short cblack[4],
                                   unsigned short *dmaxp);
  virtual void copy_bayer(unsigned short cblack[4], unsigned short *dmaxp);
  virtual void fuji_rotate();
  virtual void convert_to_rgb_loop(float out_cam[3][4]);
  virtual void lin_interpolate_loop(int *code, int size);
  virtual void scale_colors_loop(float scale_mul[4]);

  /* Fujifilm compressed decoder public interface (to make parallel decoder) */
  virtual void
  fuji_decode_loop(struct fuji_compressed_params *common_info, int count,
                   INT64 *offsets, unsigned *sizes, uchar *q_bases);
  void fuji_decode_strip(struct fuji_compressed_params *info_common,
                         int cur_block, INT64 raw_offset, unsigned size, uchar *q_bases);
  /* CR3 decoder public interface to make parallel decoder */
  virtual void crxLoadDecodeLoop(void *, int);
  int crxDecodePlane(void *, uint32_t planeNumber);
  virtual void crxLoadFinalizeLoopE3(void *, int);
  void crxConvertPlaneLineDf(void *, int);

  int FCF(int row, int col)
  {
    int rr, cc;
    if (libraw_internal_data.unpacker_data.fuji_layout)
    {
      rr = libraw_internal_data.internal_output_params.fuji_width - 1 - col +
           (row >> 1);
      cc = col + ((row + 1) >> 1);
    }
    else
    {
      rr = libraw_internal_data.internal_output_params.fuji_width - 1 + row -
           (col >> 1);
      cc = row + ((col + 1) >> 1);
    }
    return FC(rr, cc);
  }

  void adjust_bl();
  void *malloc(size_t t);
  void *calloc(size_t n, size_t t);
  void *realloc(void *p, size_t s);
  void free(void *p);
  void derror();

  LibRaw_TLS *tls;
  libraw_internal_data_t libraw_internal_data;
  decode first_decode[2048], *second_decode, *free_decode;
  tiff_ifd_t tiff_ifd[LIBRAW_IFD_MAXCOUNT];
  libraw_memmgr memmgr;
  libraw_callbacks_t callbacks;

  //void (LibRaw::*write_thumb)();
  void (LibRaw::*write_fun)();
  void (LibRaw::*load_raw)();
  //void (LibRaw::*thumb_load_raw)();
  void (LibRaw::*pentax_component_load_raw)();

  void kodak_thumb_loader();
  void write_thumb_ppm_tiff(FILE *);
#ifdef USE_X3FTOOLS
  void x3f_thumb_loader();
  INT64 x3f_thumb_size();
#endif

  int own_filtering_supported() { return 0; }
  void identify();
  void initdata();
  unsigned parse_custom_cameras(unsigned limit, libraw_custom_camera_t table[],
                                char **list);
  void write_ppm_tiff();
  void convert_to_rgb();
  void remove_zeroes();
  void crop_masked_pixels();
#ifndef NO_LCMS
  void apply_profile(const char *, const char *);
#endif
  void pre_interpolate();
  void border_interpolate(int border);
  void lin_interpolate();
  void vng_interpolate();
  void ppg_interpolate();
  void cielab(ushort rgb[3], short lab[3]);
  void xtrans_interpolate(int);
  void ahd_interpolate();
  void dht_interpolate();
  void aahd_interpolate();

  void dcb(int iterations, int dcb_enhance);
  void fbdd(int noiserd);
  void exp_bef(float expos, float preser);

  void bad_pixels(const char *);
  void subtract(const char *);
  void hat_transform(float *temp, float *base, int st, int size, int sc);
  void wavelet_denoise();
  void scale_colors();
  void median_filter();
  void blend_highlights();
  void recover_highlights();
  void green_matching();

  void stretch();

  void jpeg_thumb_writer(FILE *tfp, char *thumb, int thumb_length);
#if 0
  void jpeg_thumb();
  void ppm_thumb();
  void ppm16_thumb();
  void layer_thumb();
  void rollei_thumb();
#endif
  void kodak_thumb_load_raw();

  unsigned get4();

  int flip_index(int row, int col);
  void gamma_curve(double pwr, double ts, int mode, int imax);
  void cubic_spline(const int *x_, const int *y_, const int len);

  /* RawSpeed data */
  void *_rawspeed_camerameta;
  void *_rawspeed_decoder;
  void *_rawspeed3_handle;
  void fix_after_rawspeed(int bl);
  int try_rawspeed(); /* returns LIBRAW_SUCCESS on success */
  /* Fast cancel flag */
  long _exitflag;

  /* DNG SDK data */
  void *dnghost;
  void *dngnegative;
  void *dngimage;
  int valid_for_dngsdk();
  int try_dngsdk();
  /* X3F data */
  void *_x3f_data; /* keep it even if USE_X3FTOOLS is not defined to do not change sizeof(LibRaw)*/

  int raw_was_read()
  {
    return imgdata.rawdata.raw_image || imgdata.rawdata.color4_image ||
           imgdata.rawdata.color3_image || imgdata.rawdata.float_image ||
           imgdata.rawdata.float3_image || imgdata.rawdata.float4_image;
  }

#ifdef LIBRAW_LIBRARY_BUILD
#include "internal/libraw_internal_funcs.h"
#endif
};

#ifdef LIBRAW_LIBRARY_BUILD
ushort libraw_sget2_static(short _order, uchar *s);
unsigned libraw_sget4_static(short _order, uchar *s);
int libraw_tagtype_dataunit_bytes(int tagtype);
double  libraw_sgetreal_static(short _order, int type, uchar *s);
float   libraw_int_to_float (int i);
#endif


#ifdef LIBRAW_LIBRARY_BUILD
#define RUN_CALLBACK(stage, iter, expect)                                      \
  if (callbacks.progress_cb)                                                   \
  {                                                                            \
    int rr = (*callbacks.progress_cb)(callbacks.progresscb_data, stage, iter,  \
                                      expect);                                 \
    if (rr != 0)                                                               \
      throw LIBRAW_EXCEPTION_CANCELLED_BY_CALLBACK;                            \
  }
#endif

#endif /* __cplusplus */

#endif /* _LIBRAW_CLASS_H */
