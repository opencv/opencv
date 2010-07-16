
/* pngpriv.h - private declarations for use inside libpng
 *
 * libpng version 1.4.3 - June 26, 2010
 * For conditions of distribution and use, see copyright notice in png.h
 * Copyright (c) 1998-2010 Glenn Randers-Pehrson
 * (Version 0.96 Copyright (c) 1996, 1997 Andreas Dilger)
 * (Version 0.88 Copyright (c) 1995, 1996 Guy Eric Schalnat, Group 42, Inc.)
 *
 * This code is released under the libpng license.
 * For conditions of distribution and use, see the disclaimer
 * and license in png.h
 */

/* The symbols declared in this file (including the functions declared
 * as PNG_EXTERN) are PRIVATE.  They are not part of the libpng public
 * interface, and are not recommended for use by regular applications.
 * Some of them may become public in the future; others may stay private,
 * change in an incompatible way, or even disappear.
 * Although the libpng users are not forbidden to include this header,
 * they should be well aware of the issues that may arise from doing so.
 */

#ifndef PNGPRIV_H
#define PNGPRIV_H

#ifndef PNG_VERSION_INFO_ONLY

#include <stdlib.h>

#if defined _MSC_VER && _MSC_VER >= 1400
#pragma warning(disable: 4267)
#endif

/* The functions exported by PNG_EXTERN are internal functions, which
 * aren't usually used outside the library (as far as I know), so it is
 * debatable if they should be exported at all.  In the future, when it
 * is possible to have run-time registry of chunk-handling functions,
 * some of these will be made available again.
#define PNG_EXTERN extern
 */
#define PNG_EXTERN

/* Other defines specific to compilers can go here.  Try to keep
 * them inside an appropriate ifdef/endif pair for portability.
 */

#ifdef PNG_FLOATING_POINT_SUPPORTED
#  ifdef MACOS
     /* We need to check that <math.h> hasn't already been included earlier
      * as it seems it doesn't agree with <fp.h>, yet we should really use
      * <fp.h> if possible.
      */
#    if !defined(__MATH_H__) && !defined(__MATH_H) && !defined(__cmath__)
#      include <fp.h>
#    endif
#  else
#    include <math.h>
#  endif
#  if defined(_AMIGA) && defined(__SASC) && defined(_M68881)
     /* Amiga SAS/C: We must include builtin FPU functions when compiling using
      * MATH=68881
      */
#    include <m68881.h>
#  endif
#endif

/* Codewarrior on NT has linking problems without this. */
#if (defined(__MWERKS__) && defined(WIN32)) || defined(__STDC__)
#  define PNG_ALWAYS_EXTERN
#endif

/* This provides the non-ANSI (far) memory allocation routines. */
#if defined(__TURBOC__) && defined(__MSDOS__)
#  include <mem.h>
#  include <alloc.h>
#endif

#if defined(WIN32) || defined(_Windows) || defined(_WINDOWS) || \
    defined(_WIN32) || defined(__WIN32__)
#  include <windows.h>  /* defines _WINDOWS_ macro */
/* I have no idea why is this necessary... */
#  ifdef _MSC_VER
#    include <malloc.h>
#  endif
#endif

/* Various modes of operation.  Note that after an init, mode is set to
 * zero automatically when the structure is created.
 */
#define PNG_HAVE_IHDR               0x01
#define PNG_HAVE_PLTE               0x02
#define PNG_HAVE_IDAT               0x04
#define PNG_AFTER_IDAT              0x08 /* Have complete zlib datastream */
#define PNG_HAVE_IEND               0x10
#define PNG_HAVE_gAMA               0x20
#define PNG_HAVE_cHRM               0x40
#define PNG_HAVE_sRGB               0x80
#define PNG_HAVE_CHUNK_HEADER      0x100
#define PNG_WROTE_tIME             0x200
#define PNG_WROTE_INFO_BEFORE_PLTE 0x400
#define PNG_BACKGROUND_IS_GRAY     0x800
#define PNG_HAVE_PNG_SIGNATURE    0x1000
#define PNG_HAVE_CHUNK_AFTER_IDAT 0x2000 /* Have another chunk after IDAT */

/* Flags for the transformations the PNG library does on the image data */
#define PNG_BGR                 0x0001
#define PNG_INTERLACE           0x0002
#define PNG_PACK                0x0004
#define PNG_SHIFT               0x0008
#define PNG_SWAP_BYTES          0x0010
#define PNG_INVERT_MONO         0x0020
#define PNG_QUANTIZE            0x0040 /* formerly PNG_DITHER */
#define PNG_BACKGROUND          0x0080
#define PNG_BACKGROUND_EXPAND   0x0100
                          /*    0x0200 unused */
#define PNG_16_TO_8             0x0400
#define PNG_RGBA                0x0800
#define PNG_EXPAND              0x1000
#define PNG_GAMMA               0x2000
#define PNG_GRAY_TO_RGB         0x4000
#define PNG_FILLER              0x8000L
#define PNG_PACKSWAP           0x10000L
#define PNG_SWAP_ALPHA         0x20000L
#define PNG_STRIP_ALPHA        0x40000L
#define PNG_INVERT_ALPHA       0x80000L
#define PNG_USER_TRANSFORM    0x100000L
#define PNG_RGB_TO_GRAY_ERR   0x200000L
#define PNG_RGB_TO_GRAY_WARN  0x400000L
#define PNG_RGB_TO_GRAY       0x600000L  /* two bits, RGB_TO_GRAY_ERR|WARN */
                       /*     0x800000L     Unused */
#define PNG_ADD_ALPHA         0x1000000L  /* Added to libpng-1.2.7 */
#define PNG_EXPAND_tRNS       0x2000000L  /* Added to libpng-1.2.9 */
                       /*   0x4000000L  unused */
                       /*   0x8000000L  unused */
                       /*  0x10000000L  unused */
                       /*  0x20000000L  unused */
                       /*  0x40000000L  unused */

/* Flags for png_create_struct */
#define PNG_STRUCT_PNG   0x0001
#define PNG_STRUCT_INFO  0x0002

/* Scaling factor for filter heuristic weighting calculations */
#define PNG_WEIGHT_SHIFT 8
#define PNG_WEIGHT_FACTOR (1<<(PNG_WEIGHT_SHIFT))
#define PNG_COST_SHIFT 3
#define PNG_COST_FACTOR (1<<(PNG_COST_SHIFT))

/* Flags for the png_ptr->flags rather than declaring a byte for each one */
#define PNG_FLAG_ZLIB_CUSTOM_STRATEGY     0x0001
#define PNG_FLAG_ZLIB_CUSTOM_LEVEL        0x0002
#define PNG_FLAG_ZLIB_CUSTOM_MEM_LEVEL    0x0004
#define PNG_FLAG_ZLIB_CUSTOM_WINDOW_BITS  0x0008
#define PNG_FLAG_ZLIB_CUSTOM_METHOD       0x0010
#define PNG_FLAG_ZLIB_FINISHED            0x0020
#define PNG_FLAG_ROW_INIT                 0x0040
#define PNG_FLAG_FILLER_AFTER             0x0080
#define PNG_FLAG_CRC_ANCILLARY_USE        0x0100
#define PNG_FLAG_CRC_ANCILLARY_NOWARN     0x0200
#define PNG_FLAG_CRC_CRITICAL_USE         0x0400
#define PNG_FLAG_CRC_CRITICAL_IGNORE      0x0800
                                /*        0x1000  unused */
                                /*        0x2000  unused */
                                /*        0x4000  unused */
#define PNG_FLAG_KEEP_UNKNOWN_CHUNKS      0x8000L
#define PNG_FLAG_KEEP_UNSAFE_CHUNKS       0x10000L
#define PNG_FLAG_LIBRARY_MISMATCH         0x20000L
#define PNG_FLAG_STRIP_ERROR_NUMBERS      0x40000L
#define PNG_FLAG_STRIP_ERROR_TEXT         0x80000L
#define PNG_FLAG_MALLOC_NULL_MEM_OK       0x100000L
#define PNG_FLAG_ADD_ALPHA                0x200000L  /* Added to libpng-1.2.8 */
#define PNG_FLAG_STRIP_ALPHA              0x400000L  /* Added to libpng-1.2.8 */
#define PNG_FLAG_BENIGN_ERRORS_WARN       0x800000L  /* Added to libpng-1.4.0 */
                                  /*     0x1000000L  unused */
                                  /*     0x2000000L  unused */
                                  /*     0x4000000L  unused */
                                  /*     0x8000000L  unused */
                                  /*    0x10000000L  unused */
                                  /*    0x20000000L  unused */
                                  /*    0x40000000L  unused */

#define PNG_FLAG_CRC_ANCILLARY_MASK (PNG_FLAG_CRC_ANCILLARY_USE | \
                                     PNG_FLAG_CRC_ANCILLARY_NOWARN)

#define PNG_FLAG_CRC_CRITICAL_MASK  (PNG_FLAG_CRC_CRITICAL_USE | \
                                     PNG_FLAG_CRC_CRITICAL_IGNORE)

#define PNG_FLAG_CRC_MASK           (PNG_FLAG_CRC_ANCILLARY_MASK | \
                                     PNG_FLAG_CRC_CRITICAL_MASK)

/* Save typing and make code easier to understand */

#define PNG_COLOR_DIST(c1, c2) (abs((int)((c1).red) - (int)((c2).red)) + \
   abs((int)((c1).green) - (int)((c2).green)) + \
   abs((int)((c1).blue) - (int)((c2).blue)))

/* Added to libpng-1.2.6 JB */
#define PNG_ROWBYTES(pixel_bits, width) \
    ((pixel_bits) >= 8 ? \
    ((png_size_t)(width) * (((png_size_t)(pixel_bits)) >> 3)) : \
    (( ((png_size_t)(width) * ((png_size_t)(pixel_bits))) + 7) >> 3) )

/* PNG_OUT_OF_RANGE returns true if value is outside the range
 * ideal-delta..ideal+delta.  Each argument is evaluated twice.
 * "ideal" and "delta" should be constants, normally simple
 * integers, "value" a variable. Added to libpng-1.2.6 JB
 */
#define PNG_OUT_OF_RANGE(value, ideal, delta) \
        ( (value) < (ideal)-(delta) || (value) > (ideal)+(delta) )

/* Constant strings for known chunk types.  If you need to add a chunk,
 * define the name here, and add an invocation of the macro wherever it's
 * needed.
 */
#define PNG_IHDR PNG_CONST png_byte png_IHDR[5] = { 73,  72,  68,  82, '\0'}
#define PNG_IDAT PNG_CONST png_byte png_IDAT[5] = { 73,  68,  65,  84, '\0'}
#define PNG_IEND PNG_CONST png_byte png_IEND[5] = { 73,  69,  78,  68, '\0'}
#define PNG_PLTE PNG_CONST png_byte png_PLTE[5] = { 80,  76,  84,  69, '\0'}
#define PNG_bKGD PNG_CONST png_byte png_bKGD[5] = { 98,  75,  71,  68, '\0'}
#define PNG_cHRM PNG_CONST png_byte png_cHRM[5] = { 99,  72,  82,  77, '\0'}
#define PNG_gAMA PNG_CONST png_byte png_gAMA[5] = {103,  65,  77,  65, '\0'}
#define PNG_hIST PNG_CONST png_byte png_hIST[5] = {104,  73,  83,  84, '\0'}
#define PNG_iCCP PNG_CONST png_byte png_iCCP[5] = {105,  67,  67,  80, '\0'}
#define PNG_iTXt PNG_CONST png_byte png_iTXt[5] = {105,  84,  88, 116, '\0'}
#define PNG_oFFs PNG_CONST png_byte png_oFFs[5] = {111,  70,  70, 115, '\0'}
#define PNG_pCAL PNG_CONST png_byte png_pCAL[5] = {112,  67,  65,  76, '\0'}
#define PNG_sCAL PNG_CONST png_byte png_sCAL[5] = {115,  67,  65,  76, '\0'}
#define PNG_pHYs PNG_CONST png_byte png_pHYs[5] = {112,  72,  89, 115, '\0'}
#define PNG_sBIT PNG_CONST png_byte png_sBIT[5] = {115,  66,  73,  84, '\0'}
#define PNG_sPLT PNG_CONST png_byte png_sPLT[5] = {115,  80,  76,  84, '\0'}
#define PNG_sRGB PNG_CONST png_byte png_sRGB[5] = {115,  82,  71,  66, '\0'}
#define PNG_sTER PNG_CONST png_byte png_sTER[5] = {115,  84,  69,  82, '\0'}
#define PNG_tEXt PNG_CONST png_byte png_tEXt[5] = {116,  69,  88, 116, '\0'}
#define PNG_tIME PNG_CONST png_byte png_tIME[5] = {116,  73,  77,  69, '\0'}
#define PNG_tRNS PNG_CONST png_byte png_tRNS[5] = {116,  82,  78,  83, '\0'}
#define PNG_zTXt PNG_CONST png_byte png_zTXt[5] = {122,  84,  88, 116, '\0'}


/* Inhibit C++ name-mangling for libpng functions but not for system calls. */
#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/* These functions are used internally in the code.  They generally
 * shouldn't be used unless you are writing code to add or replace some
 * functionality in libpng.  More information about most functions can
 * be found in the files where the functions are located.
 */

/* Allocate memory for an internal libpng struct */
PNG_EXTERN png_voidp png_create_struct PNGARG((int type));

/* Free memory from internal libpng struct */
PNG_EXTERN void png_destroy_struct PNGARG((png_voidp struct_ptr));

PNG_EXTERN png_voidp png_create_struct_2 PNGARG((int type, png_malloc_ptr
  malloc_fn, png_voidp mem_ptr));
PNG_EXTERN void png_destroy_struct_2 PNGARG((png_voidp struct_ptr,
   png_free_ptr free_fn, png_voidp mem_ptr));

/* Free any memory that info_ptr points to and reset struct. */
PNG_EXTERN void png_info_destroy PNGARG((png_structp png_ptr,
   png_infop info_ptr));

/* Function to allocate memory for zlib.  PNGAPI is disallowed. */
PNG_EXTERN voidpf png_zalloc PNGARG((voidpf png_ptr, uInt items, uInt size));

/* Function to free memory for zlib.  PNGAPI is disallowed. */
PNG_EXTERN void png_zfree PNGARG((voidpf png_ptr, voidpf ptr));

/* Next four functions are used internally as callbacks.  PNGAPI is required
 * but not PNG_EXPORT.  PNGAPI added at libpng version 1.2.3. */

PNG_EXTERN void PNGAPI png_default_read_data PNGARG((png_structp png_ptr,
   png_bytep data, png_size_t length));

#ifdef PNG_PROGRESSIVE_READ_SUPPORTED
PNG_EXTERN void PNGAPI png_push_fill_buffer PNGARG((png_structp png_ptr,
   png_bytep buffer, png_size_t length));
#endif

PNG_EXTERN void PNGAPI png_default_write_data PNGARG((png_structp png_ptr,
   png_bytep data, png_size_t length));

#ifdef PNG_WRITE_FLUSH_SUPPORTED
#ifdef PNG_STDIO_SUPPORTED
PNG_EXTERN void PNGAPI png_default_flush PNGARG((png_structp png_ptr));
#endif
#endif

/* Reset the CRC variable */
PNG_EXTERN void png_reset_crc PNGARG((png_structp png_ptr));

/* Write the "data" buffer to whatever output you are using */
PNG_EXTERN void png_write_data PNGARG((png_structp png_ptr, png_bytep data,
   png_size_t length));

/* Read the chunk header (length + type name) */
PNG_EXTERN png_uint_32 png_read_chunk_header PNGARG((png_structp png_ptr));

/* Read data from whatever input you are using into the "data" buffer */
PNG_EXTERN void png_read_data PNGARG((png_structp png_ptr, png_bytep data,
   png_size_t length));

/* Read bytes into buf, and update png_ptr->crc */
PNG_EXTERN void png_crc_read PNGARG((png_structp png_ptr, png_bytep buf,
   png_size_t length));

/* Decompress data in a chunk that uses compression */
#if defined(PNG_zTXt_SUPPORTED) || defined(PNG_iTXt_SUPPORTED) || \
    defined(PNG_iCCP_SUPPORTED) || defined(PNG_sPLT_SUPPORTED)
PNG_EXTERN void png_decompress_chunk PNGARG((png_structp png_ptr,
   int comp_type, png_size_t chunklength, png_size_t prefix_length,
   png_size_t *data_length));
#endif

/* Read "skip" bytes, read the file crc, and (optionally) verify png_ptr->crc */
PNG_EXTERN int png_crc_finish PNGARG((png_structp png_ptr, png_uint_32 skip));

/* Read the CRC from the file and compare it to the libpng calculated CRC */
PNG_EXTERN int png_crc_error PNGARG((png_structp png_ptr));

/* Calculate the CRC over a section of data.  Note that we are only
 * passing a maximum of 64K on systems that have this as a memory limit,
 * since this is the maximum buffer size we can specify.
 */
PNG_EXTERN void png_calculate_crc PNGARG((png_structp png_ptr, png_bytep ptr,
   png_size_t length));

#ifdef PNG_WRITE_FLUSH_SUPPORTED
PNG_EXTERN void png_flush PNGARG((png_structp png_ptr));
#endif

/* Write various chunks */

/* Write the IHDR chunk, and update the png_struct with the necessary
 * information.
 */
PNG_EXTERN void png_write_IHDR PNGARG((png_structp png_ptr, png_uint_32 width,
   png_uint_32 height,
   int bit_depth, int color_type, int compression_method, int filter_method,
   int interlace_method));

PNG_EXTERN void png_write_PLTE PNGARG((png_structp png_ptr, png_colorp palette,
   png_uint_32 num_pal));

PNG_EXTERN void png_write_IDAT PNGARG((png_structp png_ptr, png_bytep data,
   png_size_t length));

PNG_EXTERN void png_write_IEND PNGARG((png_structp png_ptr));

#ifdef PNG_WRITE_gAMA_SUPPORTED
#ifdef PNG_FLOATING_POINT_SUPPORTED
PNG_EXTERN void png_write_gAMA PNGARG((png_structp png_ptr, double file_gamma));
#endif
#ifdef PNG_FIXED_POINT_SUPPORTED
PNG_EXTERN void png_write_gAMA_fixed PNGARG((png_structp png_ptr,
    png_fixed_point file_gamma));
#endif
#endif

#ifdef PNG_WRITE_sBIT_SUPPORTED
PNG_EXTERN void png_write_sBIT PNGARG((png_structp png_ptr, png_color_8p sbit,
   int color_type));
#endif

#ifdef PNG_WRITE_cHRM_SUPPORTED
#ifdef PNG_FLOATING_POINT_SUPPORTED
PNG_EXTERN void png_write_cHRM PNGARG((png_structp png_ptr,
   double white_x, double white_y,
   double red_x, double red_y, double green_x, double green_y,
   double blue_x, double blue_y));
#endif
PNG_EXTERN void png_write_cHRM_fixed PNGARG((png_structp png_ptr,
   png_fixed_point int_white_x, png_fixed_point int_white_y,
   png_fixed_point int_red_x, png_fixed_point int_red_y, png_fixed_point
   int_green_x, png_fixed_point int_green_y, png_fixed_point int_blue_x,
   png_fixed_point int_blue_y));
#endif

#ifdef PNG_WRITE_sRGB_SUPPORTED
PNG_EXTERN void png_write_sRGB PNGARG((png_structp png_ptr,
   int intent));
#endif

#ifdef PNG_WRITE_iCCP_SUPPORTED
PNG_EXTERN void png_write_iCCP PNGARG((png_structp png_ptr,
   png_charp name, int compression_type,
   png_charp profile, int proflen));
   /* Note to maintainer: profile should be png_bytep */
#endif

#ifdef PNG_WRITE_sPLT_SUPPORTED
PNG_EXTERN void png_write_sPLT PNGARG((png_structp png_ptr,
   png_sPLT_tp palette));
#endif

#ifdef PNG_WRITE_tRNS_SUPPORTED
PNG_EXTERN void png_write_tRNS PNGARG((png_structp png_ptr, png_bytep trans,
   png_color_16p values, int number, int color_type));
#endif

#ifdef PNG_WRITE_bKGD_SUPPORTED
PNG_EXTERN void png_write_bKGD PNGARG((png_structp png_ptr,
   png_color_16p values, int color_type));
#endif

#ifdef PNG_WRITE_hIST_SUPPORTED
PNG_EXTERN void png_write_hIST PNGARG((png_structp png_ptr, png_uint_16p hist,
   int num_hist));
#endif

#if defined(PNG_WRITE_TEXT_SUPPORTED) || defined(PNG_WRITE_pCAL_SUPPORTED) || \
    defined(PNG_WRITE_iCCP_SUPPORTED) || defined(PNG_WRITE_sPLT_SUPPORTED)
PNG_EXTERN png_size_t png_check_keyword PNGARG((png_structp png_ptr,
   png_charp key, png_charpp new_key));
#endif

#ifdef PNG_WRITE_tEXt_SUPPORTED
PNG_EXTERN void png_write_tEXt PNGARG((png_structp png_ptr, png_charp key,
   png_charp text, png_size_t text_len));
#endif

#ifdef PNG_WRITE_zTXt_SUPPORTED
PNG_EXTERN void png_write_zTXt PNGARG((png_structp png_ptr, png_charp key,
   png_charp text, png_size_t text_len, int compression));
#endif

#ifdef PNG_WRITE_iTXt_SUPPORTED
PNG_EXTERN void png_write_iTXt PNGARG((png_structp png_ptr,
   int compression, png_charp key, png_charp lang, png_charp lang_key,
   png_charp text));
#endif

#ifdef PNG_TEXT_SUPPORTED  /* Added at version 1.0.14 and 1.2.4 */
PNG_EXTERN int png_set_text_2 PNGARG((png_structp png_ptr,
   png_infop info_ptr, png_textp text_ptr, int num_text));
#endif

#ifdef PNG_WRITE_oFFs_SUPPORTED
PNG_EXTERN void png_write_oFFs PNGARG((png_structp png_ptr,
   png_int_32 x_offset, png_int_32 y_offset, int unit_type));
#endif

#ifdef PNG_WRITE_pCAL_SUPPORTED
PNG_EXTERN void png_write_pCAL PNGARG((png_structp png_ptr, png_charp purpose,
   png_int_32 X0, png_int_32 X1, int type, int nparams,
   png_charp units, png_charpp params));
#endif

#ifdef PNG_WRITE_pHYs_SUPPORTED
PNG_EXTERN void png_write_pHYs PNGARG((png_structp png_ptr,
   png_uint_32 x_pixels_per_unit, png_uint_32 y_pixels_per_unit,
   int unit_type));
#endif

#ifdef PNG_WRITE_tIME_SUPPORTED
PNG_EXTERN void png_write_tIME PNGARG((png_structp png_ptr,
   png_timep mod_time));
#endif

#ifdef PNG_WRITE_sCAL_SUPPORTED
#if defined(PNG_FLOATING_POINT_SUPPORTED) && defined(PNG_STDIO_SUPPORTED)
PNG_EXTERN void png_write_sCAL PNGARG((png_structp png_ptr,
   int unit, double width, double height));
#else
#ifdef PNG_FIXED_POINT_SUPPORTED
PNG_EXTERN void png_write_sCAL_s PNGARG((png_structp png_ptr,
   int unit, png_charp width, png_charp height));
#endif
#endif
#endif

/* Called when finished processing a row of data */
PNG_EXTERN void png_write_finish_row PNGARG((png_structp png_ptr));

/* Internal use only.   Called before first row of data */
PNG_EXTERN void png_write_start_row PNGARG((png_structp png_ptr));

#ifdef PNG_READ_GAMMA_SUPPORTED
PNG_EXTERN void png_build_gamma_table PNGARG((png_structp png_ptr,
   png_byte bit_depth));
#endif

/* Combine a row of data, dealing with alpha, etc. if requested */
PNG_EXTERN void png_combine_row PNGARG((png_structp png_ptr, png_bytep row,
   int mask));

#ifdef PNG_READ_INTERLACING_SUPPORTED
/* Expand an interlaced row */
/* OLD pre-1.0.9 interface:
PNG_EXTERN void png_do_read_interlace PNGARG((png_row_infop row_info,
   png_bytep row, int pass, png_uint_32 transformations));
 */
PNG_EXTERN void png_do_read_interlace PNGARG((png_structp png_ptr));
#endif

/* GRR TO DO (2.0 or whenever):  simplify other internal calling interfaces */

#ifdef PNG_WRITE_INTERLACING_SUPPORTED
/* Grab pixels out of a row for an interlaced pass */
PNG_EXTERN void png_do_write_interlace PNGARG((png_row_infop row_info,
   png_bytep row, int pass));
#endif

/* Unfilter a row */
PNG_EXTERN void png_read_filter_row PNGARG((png_structp png_ptr,
   png_row_infop row_info, png_bytep row, png_bytep prev_row, int filter));

/* Choose the best filter to use and filter the row data */
PNG_EXTERN void png_write_find_filter PNGARG((png_structp png_ptr,
   png_row_infop row_info));

/* Write out the filtered row. */
PNG_EXTERN void png_write_filtered_row PNGARG((png_structp png_ptr,
   png_bytep filtered_row));
/* Finish a row while reading, dealing with interlacing passes, etc. */
PNG_EXTERN void png_read_finish_row PNGARG((png_structp png_ptr));

/* Initialize the row buffers, etc. */
PNG_EXTERN void png_read_start_row PNGARG((png_structp png_ptr));
/* Optional call to update the users info structure */
PNG_EXTERN void png_read_transform_info PNGARG((png_structp png_ptr,
   png_infop info_ptr));

/* These are the functions that do the transformations */
#ifdef PNG_READ_FILLER_SUPPORTED
PNG_EXTERN void png_do_read_filler PNGARG((png_row_infop row_info,
   png_bytep row, png_uint_32 filler, png_uint_32 flags));
#endif

#ifdef PNG_READ_SWAP_ALPHA_SUPPORTED
PNG_EXTERN void png_do_read_swap_alpha PNGARG((png_row_infop row_info,
   png_bytep row));
#endif

#ifdef PNG_WRITE_SWAP_ALPHA_SUPPORTED
PNG_EXTERN void png_do_write_swap_alpha PNGARG((png_row_infop row_info,
   png_bytep row));
#endif

#ifdef PNG_READ_INVERT_ALPHA_SUPPORTED
PNG_EXTERN void png_do_read_invert_alpha PNGARG((png_row_infop row_info,
   png_bytep row));
#endif

#ifdef PNG_WRITE_INVERT_ALPHA_SUPPORTED
PNG_EXTERN void png_do_write_invert_alpha PNGARG((png_row_infop row_info,
   png_bytep row));
#endif

#if defined(PNG_WRITE_FILLER_SUPPORTED) || \
    defined(PNG_READ_STRIP_ALPHA_SUPPORTED)
PNG_EXTERN void png_do_strip_filler PNGARG((png_row_infop row_info,
   png_bytep row, png_uint_32 flags));
#endif

#if defined(PNG_READ_SWAP_SUPPORTED) || defined(PNG_WRITE_SWAP_SUPPORTED)
PNG_EXTERN void png_do_swap PNGARG((png_row_infop row_info, png_bytep row));
#endif

#if defined(PNG_READ_PACKSWAP_SUPPORTED) || \
    defined(PNG_WRITE_PACKSWAP_SUPPORTED)
PNG_EXTERN void png_do_packswap PNGARG((png_row_infop row_info, png_bytep row));
#endif

#ifdef PNG_READ_RGB_TO_GRAY_SUPPORTED
PNG_EXTERN int png_do_rgb_to_gray PNGARG((png_structp png_ptr, png_row_infop
   row_info, png_bytep row));
#endif

#ifdef PNG_READ_GRAY_TO_RGB_SUPPORTED
PNG_EXTERN void png_do_gray_to_rgb PNGARG((png_row_infop row_info,
   png_bytep row));
#endif

#ifdef PNG_READ_PACK_SUPPORTED
PNG_EXTERN void png_do_unpack PNGARG((png_row_infop row_info, png_bytep row));
#endif

#ifdef PNG_READ_SHIFT_SUPPORTED
PNG_EXTERN void png_do_unshift PNGARG((png_row_infop row_info, png_bytep row,
   png_color_8p sig_bits));
#endif

#if defined(PNG_READ_INVERT_SUPPORTED) || defined(PNG_WRITE_INVERT_SUPPORTED)
PNG_EXTERN void png_do_invert PNGARG((png_row_infop row_info, png_bytep row));
#endif

#ifdef PNG_READ_16_TO_8_SUPPORTED
PNG_EXTERN void png_do_chop PNGARG((png_row_infop row_info, png_bytep row));
#endif

#ifdef PNG_READ_QUANTIZE_SUPPORTED
PNG_EXTERN void png_do_quantize PNGARG((png_row_infop row_info,
   png_bytep row, png_bytep palette_lookup, png_bytep quantize_lookup));

#  ifdef PNG_CORRECT_PALETTE_SUPPORTED
PNG_EXTERN void png_correct_palette PNGARG((png_structp png_ptr,
   png_colorp palette, int num_palette));
#  endif
#endif

#if defined(PNG_READ_BGR_SUPPORTED) || defined(PNG_WRITE_BGR_SUPPORTED)
PNG_EXTERN void png_do_bgr PNGARG((png_row_infop row_info, png_bytep row));
#endif

#ifdef PNG_WRITE_PACK_SUPPORTED
PNG_EXTERN void png_do_pack PNGARG((png_row_infop row_info,
   png_bytep row, png_uint_32 bit_depth));
#endif

#ifdef PNG_WRITE_SHIFT_SUPPORTED
PNG_EXTERN void png_do_shift PNGARG((png_row_infop row_info, png_bytep row,
   png_color_8p bit_depth));
#endif

#ifdef PNG_READ_BACKGROUND_SUPPORTED
#ifdef PNG_READ_GAMMA_SUPPORTED
PNG_EXTERN void png_do_background PNGARG((png_row_infop row_info, png_bytep row,
   png_color_16p trans_color, png_color_16p background,
   png_color_16p background_1,
   png_bytep gamma_table, png_bytep gamma_from_1, png_bytep gamma_to_1,
   png_uint_16pp gamma_16, png_uint_16pp gamma_16_from_1,
   png_uint_16pp gamma_16_to_1, int gamma_shift));
#else
PNG_EXTERN void png_do_background PNGARG((png_row_infop row_info, png_bytep row,
   png_color_16p trans_color, png_color_16p background));
#endif
#endif

#ifdef PNG_READ_GAMMA_SUPPORTED
PNG_EXTERN void png_do_gamma PNGARG((png_row_infop row_info, png_bytep row,
   png_bytep gamma_table, png_uint_16pp gamma_16_table,
   int gamma_shift));
#endif

#ifdef PNG_READ_EXPAND_SUPPORTED
PNG_EXTERN void png_do_expand_palette PNGARG((png_row_infop row_info,
   png_bytep row, png_colorp palette, png_bytep trans, int num_trans));
PNG_EXTERN void png_do_expand PNGARG((png_row_infop row_info,
   png_bytep row, png_color_16p trans_value));
#endif

/* The following decodes the appropriate chunks, and does error correction,
 * then calls the appropriate callback for the chunk if it is valid.
 */

/* Decode the IHDR chunk */
PNG_EXTERN void png_handle_IHDR PNGARG((png_structp png_ptr, png_infop info_ptr,
   png_uint_32 length));
PNG_EXTERN void png_handle_PLTE PNGARG((png_structp png_ptr, png_infop info_ptr,
   png_uint_32 length));
PNG_EXTERN void png_handle_IEND PNGARG((png_structp png_ptr, png_infop info_ptr,
   png_uint_32 length));

#ifdef PNG_READ_bKGD_SUPPORTED
PNG_EXTERN void png_handle_bKGD PNGARG((png_structp png_ptr, png_infop info_ptr,
   png_uint_32 length));
#endif

#ifdef PNG_READ_cHRM_SUPPORTED
PNG_EXTERN void png_handle_cHRM PNGARG((png_structp png_ptr, png_infop info_ptr,
   png_uint_32 length));
#endif

#ifdef PNG_READ_gAMA_SUPPORTED
PNG_EXTERN void png_handle_gAMA PNGARG((png_structp png_ptr, png_infop info_ptr,
   png_uint_32 length));
#endif

#ifdef PNG_READ_hIST_SUPPORTED
PNG_EXTERN void png_handle_hIST PNGARG((png_structp png_ptr, png_infop info_ptr,
   png_uint_32 length));
#endif

#ifdef PNG_READ_iCCP_SUPPORTED
extern void png_handle_iCCP PNGARG((png_structp png_ptr, png_infop info_ptr,
   png_uint_32 length));
#endif /* PNG_READ_iCCP_SUPPORTED */

#ifdef PNG_READ_iTXt_SUPPORTED
PNG_EXTERN void png_handle_iTXt PNGARG((png_structp png_ptr, png_infop info_ptr,
   png_uint_32 length));
#endif

#ifdef PNG_READ_oFFs_SUPPORTED
PNG_EXTERN void png_handle_oFFs PNGARG((png_structp png_ptr, png_infop info_ptr,
   png_uint_32 length));
#endif

#ifdef PNG_READ_pCAL_SUPPORTED
PNG_EXTERN void png_handle_pCAL PNGARG((png_structp png_ptr, png_infop info_ptr,
   png_uint_32 length));
#endif

#ifdef PNG_READ_pHYs_SUPPORTED
PNG_EXTERN void png_handle_pHYs PNGARG((png_structp png_ptr, png_infop info_ptr,
   png_uint_32 length));
#endif

#ifdef PNG_READ_sBIT_SUPPORTED
PNG_EXTERN void png_handle_sBIT PNGARG((png_structp png_ptr, png_infop info_ptr,
   png_uint_32 length));
#endif

#ifdef PNG_READ_sCAL_SUPPORTED
PNG_EXTERN void png_handle_sCAL PNGARG((png_structp png_ptr, png_infop info_ptr,
   png_uint_32 length));
#endif

#ifdef PNG_READ_sPLT_SUPPORTED
extern void png_handle_sPLT PNGARG((png_structp png_ptr, png_infop info_ptr,
   png_uint_32 length));
#endif /* PNG_READ_sPLT_SUPPORTED */

#ifdef PNG_READ_sRGB_SUPPORTED
PNG_EXTERN void png_handle_sRGB PNGARG((png_structp png_ptr, png_infop info_ptr,
   png_uint_32 length));
#endif

#ifdef PNG_READ_tEXt_SUPPORTED
PNG_EXTERN void png_handle_tEXt PNGARG((png_structp png_ptr, png_infop info_ptr,
   png_uint_32 length));
#endif

#ifdef PNG_READ_tIME_SUPPORTED
PNG_EXTERN void png_handle_tIME PNGARG((png_structp png_ptr, png_infop info_ptr,
   png_uint_32 length));
#endif

#ifdef PNG_READ_tRNS_SUPPORTED
PNG_EXTERN void png_handle_tRNS PNGARG((png_structp png_ptr, png_infop info_ptr,
   png_uint_32 length));
#endif

#ifdef PNG_READ_zTXt_SUPPORTED
PNG_EXTERN void png_handle_zTXt PNGARG((png_structp png_ptr, png_infop info_ptr,
   png_uint_32 length));
#endif

PNG_EXTERN void png_handle_unknown PNGARG((png_structp png_ptr,
   png_infop info_ptr, png_uint_32 length));

PNG_EXTERN void png_check_chunk_name PNGARG((png_structp png_ptr,
   png_bytep chunk_name));

/* Handle the transformations for reading and writing */
PNG_EXTERN void png_do_read_transformations PNGARG((png_structp png_ptr));
PNG_EXTERN void png_do_write_transformations PNGARG((png_structp png_ptr));

PNG_EXTERN void png_init_read_transformations PNGARG((png_structp png_ptr));

#ifdef PNG_PROGRESSIVE_READ_SUPPORTED
PNG_EXTERN void png_push_read_chunk PNGARG((png_structp png_ptr,
   png_infop info_ptr));
PNG_EXTERN void png_push_read_sig PNGARG((png_structp png_ptr,
   png_infop info_ptr));
PNG_EXTERN void png_push_check_crc PNGARG((png_structp png_ptr));
PNG_EXTERN void png_push_crc_skip PNGARG((png_structp png_ptr,
   png_uint_32 length));
PNG_EXTERN void png_push_crc_finish PNGARG((png_structp png_ptr));
PNG_EXTERN void png_push_save_buffer PNGARG((png_structp png_ptr));
PNG_EXTERN void png_push_restore_buffer PNGARG((png_structp png_ptr,
   png_bytep buffer, png_size_t buffer_length));
PNG_EXTERN void png_push_read_IDAT PNGARG((png_structp png_ptr));
PNG_EXTERN void png_process_IDAT_data PNGARG((png_structp png_ptr,
   png_bytep buffer, png_size_t buffer_length));
PNG_EXTERN void png_push_process_row PNGARG((png_structp png_ptr));
PNG_EXTERN void png_push_handle_unknown PNGARG((png_structp png_ptr,
   png_infop info_ptr, png_uint_32 length));
PNG_EXTERN void png_push_have_info PNGARG((png_structp png_ptr,
   png_infop info_ptr));
PNG_EXTERN void png_push_have_end PNGARG((png_structp png_ptr,
   png_infop info_ptr));
PNG_EXTERN void png_push_have_row PNGARG((png_structp png_ptr, png_bytep row));
PNG_EXTERN void png_push_read_end PNGARG((png_structp png_ptr,
   png_infop info_ptr));
PNG_EXTERN void png_process_some_data PNGARG((png_structp png_ptr,
   png_infop info_ptr));
PNG_EXTERN void png_read_push_finish_row PNGARG((png_structp png_ptr));
#ifdef PNG_READ_tEXt_SUPPORTED
PNG_EXTERN void png_push_handle_tEXt PNGARG((png_structp png_ptr,
   png_infop info_ptr, png_uint_32 length));
PNG_EXTERN void png_push_read_tEXt PNGARG((png_structp png_ptr,
   png_infop info_ptr));
#endif
#ifdef PNG_READ_zTXt_SUPPORTED
PNG_EXTERN void png_push_handle_zTXt PNGARG((png_structp png_ptr,
   png_infop info_ptr, png_uint_32 length));
PNG_EXTERN void png_push_read_zTXt PNGARG((png_structp png_ptr,
   png_infop info_ptr));
#endif
#ifdef PNG_READ_iTXt_SUPPORTED
PNG_EXTERN void png_push_handle_iTXt PNGARG((png_structp png_ptr,
   png_infop info_ptr, png_uint_32 length));
PNG_EXTERN void png_push_read_iTXt PNGARG((png_structp png_ptr,
   png_infop info_ptr));
#endif

#endif /* PNG_PROGRESSIVE_READ_SUPPORTED */

#ifdef PNG_MNG_FEATURES_SUPPORTED
PNG_EXTERN void png_do_read_intrapixel PNGARG((png_row_infop row_info,
   png_bytep row));
PNG_EXTERN void png_do_write_intrapixel PNGARG((png_row_infop row_info,
   png_bytep row));
#endif

/* Added at libpng version 1.4.0 */
#ifdef PNG_cHRM_SUPPORTED
PNG_EXTERN int png_check_cHRM_fixed PNGARG((png_structp png_ptr,
   png_fixed_point int_white_x, png_fixed_point int_white_y,
   png_fixed_point int_red_x, png_fixed_point int_red_y, png_fixed_point
   int_green_x, png_fixed_point int_green_y, png_fixed_point int_blue_x,
   png_fixed_point int_blue_y));
#endif

#ifdef PNG_cHRM_SUPPORTED
#ifdef PNG_CHECK_cHRM_SUPPORTED
/* Added at libpng version 1.2.34 and 1.4.0 */
PNG_EXTERN void png_64bit_product PNGARG((long v1, long v2,
   unsigned long *hi_product, unsigned long *lo_product));
#endif
#endif

/* Added at libpng version 1.4.0 */
PNG_EXTERN void png_check_IHDR PNGARG((png_structp png_ptr,
   png_uint_32 width, png_uint_32 height, int bit_depth,
   int color_type, int interlace_type, int compression_type,
   int filter_type));

/* Free all memory used by the read (old method - NOT DLL EXPORTED) */
extern void png_read_destroy PNGARG((png_structp png_ptr, png_infop info_ptr,
   png_infop end_info_ptr));

/* Free any memory used in png_ptr struct (old method - NOT DLL EXPORTED) */
extern void png_write_destroy PNGARG((png_structp png_ptr));

#ifdef USE_FAR_KEYWORD  /* memory model conversion function */
extern void *png_far_to_near PNGARG((png_structp png_ptr,png_voidp ptr,
   int check));
#endif /* USE_FAR_KEYWORD */

/* Define PNG_DEBUG at compile time for debugging information.  Higher
 * numbers for PNG_DEBUG mean more debugging information.  This has
 * only been added since version 0.95 so it is not implemented throughout
 * libpng yet, but more support will be added as needed.
 */
#ifdef PNG_DEBUG
#if (PNG_DEBUG > 0)
#if !defined(PNG_DEBUG_FILE) && defined(_MSC_VER)
#include <crtdbg.h>
#if (PNG_DEBUG > 1)
#ifndef _DEBUG
#  define _DEBUG
#endif
#ifndef png_debug
#define png_debug(l,m)  _RPT0(_CRT_WARN,m PNG_STRING_NEWLINE)
#endif
#ifndef png_debug1
#define png_debug1(l,m,p1)  _RPT1(_CRT_WARN,m PNG_STRING_NEWLINE,p1)
#endif
#ifndef png_debug2
#define png_debug2(l,m,p1,p2) _RPT2(_CRT_WARN,m PNG_STRING_NEWLINE,p1,p2)
#endif
#endif
#else /* PNG_DEBUG_FILE || !_MSC_VER */
#ifndef PNG_DEBUG_FILE
#define PNG_DEBUG_FILE stderr
#endif /* PNG_DEBUG_FILE */

#if (PNG_DEBUG > 1)
/* Note: ["%s"m PNG_STRING_NEWLINE] probably does not work on
 * non-ISO compilers
 */
#  ifdef __STDC__
#    ifndef png_debug
#      define png_debug(l,m) \
       { \
       int num_tabs=l; \
       fprintf(PNG_DEBUG_FILE,"%s"m PNG_STRING_NEWLINE,(num_tabs==1 ? "\t" : \
         (num_tabs==2 ? "\t\t":(num_tabs>2 ? "\t\t\t":"")))); \
       }
#    endif
#    ifndef png_debug1
#      define png_debug1(l,m,p1) \
       { \
       int num_tabs=l; \
       fprintf(PNG_DEBUG_FILE,"%s"m PNG_STRING_NEWLINE,(num_tabs==1 ? "\t" : \
         (num_tabs==2 ? "\t\t":(num_tabs>2 ? "\t\t\t":""))),p1); \
       }
#    endif
#    ifndef png_debug2
#      define png_debug2(l,m,p1,p2) \
       { \
       int num_tabs=l; \
       fprintf(PNG_DEBUG_FILE,"%s"m PNG_STRING_NEWLINE,(num_tabs==1 ? "\t" : \
         (num_tabs==2 ? "\t\t":(num_tabs>2 ? "\t\t\t":""))),p1,p2); \
       }
#    endif
#  else /* __STDC __ */
#    ifndef png_debug
#      define png_debug(l,m) \
       { \
       int num_tabs=l; \
       char format[256]; \
       snprintf(format,256,"%s%s%s",(num_tabs==1 ? "\t" : \
         (num_tabs==2 ? "\t\t":(num_tabs>2 ? "\t\t\t":""))), \
         m,PNG_STRING_NEWLINE); \
       fprintf(PNG_DEBUG_FILE,format); \
       }
#    endif
#    ifndef png_debug1
#      define png_debug1(l,m,p1) \
       { \
       int num_tabs=l; \
       char format[256]; \
       snprintf(format,256,"%s%s%s",(num_tabs==1 ? "\t" : \
         (num_tabs==2 ? "\t\t":(num_tabs>2 ? "\t\t\t":""))), \
         m,PNG_STRING_NEWLINE); \
       fprintf(PNG_DEBUG_FILE,format,p1); \
       }
#    endif
#    ifndef png_debug2
#      define png_debug2(l,m,p1,p2) \
       { \
       int num_tabs=l; \
       char format[256]; \
       snprintf(format,256,"%s%s%s",(num_tabs==1 ? "\t" : \
         (num_tabs==2 ? "\t\t":(num_tabs>2 ? "\t\t\t":""))), \
         m,PNG_STRING_NEWLINE); \
       fprintf(PNG_DEBUG_FILE,format,p1,p2); \
       }
#    endif
#  endif /* __STDC __ */
#endif /* (PNG_DEBUG > 1) */

#endif /* _MSC_VER */
#endif /* (PNG_DEBUG > 0) */
#endif /* PNG_DEBUG */
#ifndef png_debug
#define png_debug(l, m)
#endif
#ifndef png_debug1
#define png_debug1(l, m, p1)
#endif
#ifndef png_debug2
#define png_debug2(l, m, p1, p2)
#endif

/* Maintainer: Put new private prototypes here ^ and in libpngpf.3 */

#ifdef __cplusplus
}
#endif

#endif /* PNG_VERSION_INFO_ONLY */
#endif /* PNGPRIV_H */
