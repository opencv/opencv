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

/* Library for accessing X3F Files
----------------------------------------------------------------
BSD-style License
----------------------------------------------------------------

* Copyright (c) 2010, Roland Karlsson (roland@proxel.se)
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*     * Redistributions of source code must retain the above copyright
*       notice, this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in the
*       documentation and/or other materials provided with the distribution.
*     * Neither the name of the organization nor the
*       names of its contributors may be used to endorse or promote products
*       derived from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY ROLAND KARLSSON ''AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL ROLAND KARLSSON BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#ifndef X3F_TOOLS_H
#define X3F_TOOLS_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include "../libraw/libraw_datastream.h"

/* From X3F_IO.H */

#define SIZE_UNIQUE_IDENTIFIER 16
#define SIZE_WHITE_BALANCE 32
#define SIZE_COLOR_MODE 32
#define NUM_EXT_DATA_2_1 32
#define NUM_EXT_DATA_3_0 64
#define NUM_EXT_DATA NUM_EXT_DATA_3_0

#define X3F_VERSION(MAJ, MIN) (uint32_t)(((MAJ) << 16) + MIN)
#define X3F_VERSION_2_0 X3F_VERSION(2, 0)
#define X3F_VERSION_2_1 X3F_VERSION(2, 1)
#define X3F_VERSION_2_2 X3F_VERSION(2, 2)
#define X3F_VERSION_2_3 X3F_VERSION(2, 3)
#define X3F_VERSION_3_0 X3F_VERSION(3, 0)
#define X3F_VERSION_4_0 X3F_VERSION(4, 0)

/* Main file identifier */
#define X3F_FOVb (uint32_t)(0x62564f46)
/* Directory identifier */
#define X3F_SECd (uint32_t)(0x64434553)
/* Property section identifiers */
#define X3F_PROP (uint32_t)(0x504f5250)
#define X3F_SECp (uint32_t)(0x70434553)
/* Image section identifiers */
#define X3F_IMAG (uint32_t)(0x46414d49)
#define X3F_IMA2 (uint32_t)(0x32414d49)
#define X3F_SECi (uint32_t)(0x69434553)
/* CAMF section identifiers */
#define X3F_CAMF (uint32_t)(0x464d4143)
#define X3F_SECc (uint32_t)(0x63434553)
/* CAMF entry identifiers */
#define X3F_CMbP (uint32_t)(0x50624d43)
#define X3F_CMbT (uint32_t)(0x54624d43)
#define X3F_CMbM (uint32_t)(0x4d624d43)
#define X3F_CMb (uint32_t)(0x00624d43)
/* SDQ section identifiers ? - TODO */
#define X3F_SPPA (uint32_t)(0x41505053)
#define X3F_SECs (uint32_t)(0x73434553)

#define X3F_IMAGE_THUMB_PLAIN (uint32_t)(0x00020003)
#define X3F_IMAGE_THUMB_HUFFMAN (uint32_t)(0x0002000b)
#define X3F_IMAGE_THUMB_JPEG (uint32_t)(0x00020012)
#define X3F_IMAGE_THUMB_SDQ (uint32_t)(0x00020019) /* SDQ ? - TODO */

#define X3F_IMAGE_RAW_HUFFMAN_X530 (uint32_t)(0x00030005)
#define X3F_IMAGE_RAW_HUFFMAN_10BIT (uint32_t)(0x00030006)
#define X3F_IMAGE_RAW_TRUE (uint32_t)(0x0003001e)
#define X3F_IMAGE_RAW_MERRILL (uint32_t)(0x0001001e)
#define X3F_IMAGE_RAW_QUATTRO (uint32_t)(0x00010023)
#define X3F_IMAGE_RAW_SDQ (uint32_t)(0x00010025)
#define X3F_IMAGE_RAW_SDQH (uint32_t)(0x00010027)
#define X3F_IMAGE_RAW_SDQH2 (uint32_t)(0x00010029)

#define X3F_IMAGE_HEADER_SIZE 28
#define X3F_CAMF_HEADER_SIZE 28
#define X3F_PROPERTY_LIST_HEADER_SIZE 24

typedef uint16_t utf16_t;

typedef int bool_t;

typedef enum x3f_extended_types_e
{
  X3F_EXT_TYPE_NONE = 0,
  X3F_EXT_TYPE_EXPOSURE_ADJUST = 1,
  X3F_EXT_TYPE_CONTRAST_ADJUST = 2,
  X3F_EXT_TYPE_SHADOW_ADJUST = 3,
  X3F_EXT_TYPE_HIGHLIGHT_ADJUST = 4,
  X3F_EXT_TYPE_SATURATION_ADJUST = 5,
  X3F_EXT_TYPE_SHARPNESS_ADJUST = 6,
  X3F_EXT_TYPE_RED_ADJUST = 7,
  X3F_EXT_TYPE_GREEN_ADJUST = 8,
  X3F_EXT_TYPE_BLUE_ADJUST = 9,
  X3F_EXT_TYPE_FILL_LIGHT_ADJUST = 10
} x3f_extended_types_t;

typedef struct x3f_property_s
{
  /* Read from file */
  uint32_t name_offset;
  uint32_t value_offset;

  /* Computed */
  utf16_t *name;  /* 0x0000 terminated UTF 16 */
  utf16_t *value; /* 0x0000 terminated UTF 16 */
} x3f_property_t;

typedef struct x3f_property_table_s
{
  uint32_t size;
  x3f_property_t *element;
} x3f_property_table_t;

typedef struct x3f_property_list_s
{
  /* 2.0 Fields */
  uint32_t num_properties;
  uint32_t character_format;
  uint32_t reserved;
  uint32_t total_length;

  x3f_property_table_t property_table;

  void *data;

  uint32_t data_size;

} x3f_property_list_t;

typedef struct x3f_table8_s
{
  uint32_t size;
  uint8_t *element;
} x3f_table8_t;

typedef struct x3f_table16_s
{
  uint32_t size;
  uint16_t *element;
} x3f_table16_t;

typedef struct x3f_table32_s
{
  uint32_t size;
  uint32_t *element;
} x3f_table32_t;

typedef struct
{
  uint8_t *data; /* Pointer to actual image data */
  void *buf;     /* Pointer to allocated buffer for free() */
  uint32_t rows;
  uint32_t columns;
  uint32_t channels;
  uint32_t row_stride;
} x3f_area8_t;

typedef struct
{
  uint16_t *data; /* Pointer to actual image data */
  void *buf;      /* Pointer to allocated buffer for free() */
  uint32_t rows;
  uint32_t columns;
  uint32_t channels;
  uint32_t row_stride;
} x3f_area16_t;

#define UNDEFINED_LEAF 0xffffffff

typedef struct x3f_huffnode_s
{
  struct x3f_huffnode_s *branch[2];
  uint32_t leaf;
} x3f_huffnode_t;

typedef struct x3f_hufftree_s
{
  uint32_t free_node_index; /* Free node index in huffman tree array */
  uint32_t total_node_index;
  x3f_huffnode_t *nodes;    /* Coding tree */
} x3f_hufftree_t;

typedef struct x3f_true_huffman_element_s
{
  uint8_t code_size;
  uint8_t code;
} x3f_true_huffman_element_t;

typedef struct x3f_true_huffman_s
{
  uint32_t size;
  x3f_true_huffman_element_t *element;
} x3f_true_huffman_t;

/* 0=bottom, 1=middle, 2=top */
#define TRUE_PLANES 3

typedef struct x3f_true_s
{
  uint16_t seed[TRUE_PLANES]; /* Always 512,512,512 */
  uint16_t unknown;           /* Always 0 */
  x3f_true_huffman_t table;   /* Huffman table - zero
                     terminated. size is the number of
                     leaves plus 1.*/

  x3f_table32_t plane_size;            /* Size of the 3 planes */
  uint8_t *plane_address[TRUE_PLANES]; /* computed offset to the planes */
  x3f_hufftree_t tree;                 /* Coding tree */
  x3f_area16_t x3rgb16;                /* 3x16 bit X3-RGB data */
} x3f_true_t;

typedef struct x3f_quattro_s
{
  struct
  {
    uint16_t columns;
    uint16_t rows;
  } plane[TRUE_PLANES];
  uint32_t unknown;

  bool_t quattro_layout;
  x3f_area16_t top16; /* Container for the bigger top layer */
} x3f_quattro_t;

typedef struct x3f_huffman_s
{
  x3f_table16_t mapping;     /* Value Mapping = X3F lossy compression */
  x3f_table32_t table;       /* Coding Table */
  x3f_hufftree_t tree;       /* Coding tree */
  x3f_table32_t row_offsets; /* Row offsets */
  x3f_area8_t rgb8;          /* 3x8 bit RGB data */
  x3f_area16_t x3rgb16;      /* 3x16 bit X3-RGB data */
} x3f_huffman_t;

typedef struct x3f_image_data_s
{
  /* 2.0 Fields */
  /* ------------------------------------------------------------------ */
  /* Known combinations of type and format are:
     1-6, 2-3, 2-11, 2-18, 3-6 */
  uint32_t type;        /* 1 = RAW X3 (SD1)
                           2 = thumbnail or maybe just RGB
                           3 = RAW X3 */
  uint32_t format;      /* 3 = 3x8 bit pixmap
                           6 = 3x10 bit huffman with map table
                           11 = 3x8 bit huffman
                           18 = JPEG */
  uint32_t type_format; /* type<<16 + format */
  /* ------------------------------------------------------------------ */

  uint32_t columns;    /* width / row size in pixels */
  uint32_t rows;       /* height */
  uint32_t row_stride; /* row size in bytes */

  /* NULL if not used */
  x3f_huffman_t *huffman; /* Huffman help data */
  x3f_true_t *tru;        /* TRUE help data */
  x3f_quattro_t *quattro; /* Quattro help data */

  void *data; /* Take from file if NULL. Otherwise,
                 this is the actual data bytes in
                 the file. */
  uint32_t data_size;

} x3f_image_data_t;

typedef struct camf_dim_entry_s
{
  uint32_t size;
  uint32_t name_offset;
  uint32_t n; /* 0,1,2,3... */
  char *name;
} camf_dim_entry_t;

typedef enum
{
  M_FLOAT,
  M_INT,
  M_UINT
} matrix_type_t;

typedef struct camf_entry_s
{
  /* pointer into decoded data */
  void *entry;

  /* entry header */
  uint32_t id;
  uint32_t version;
  uint32_t entry_size;
  uint32_t name_offset;
  uint32_t value_offset;

  /* computed values */
  char *name_address;
  void *value_address;
  uint32_t name_size;
  uint32_t value_size;

  /* extracted values for explicit CAMF entry types*/
  uint32_t text_size;
  char *text;

  uint32_t property_num;
  char **property_name;
  uint8_t **property_value;

  uint32_t matrix_dim;
  camf_dim_entry_t *matrix_dim_entry;

  /* Offset, pointer and size and type of raw data */
  uint32_t matrix_type;
  uint32_t matrix_data_off;
  void *matrix_data;
  uint32_t matrix_element_size;

  /* Pointer and type of copied data */
  matrix_type_t matrix_decoded_type;
  void *matrix_decoded;

  /* Help data to try to estimate element size */
  uint32_t matrix_elements;
  uint32_t matrix_used_space;
  double matrix_estimated_element_size;

} camf_entry_t;

typedef struct camf_entry_table_s
{
  uint32_t size;
  camf_entry_t *element;
} camf_entry_table_t;

typedef struct x3f_camf_typeN_s
{
  uint32_t val0;
  uint32_t val1;
  uint32_t val2;
  uint32_t val3;
} x3f_camf_typeN_t;

typedef struct x3f_camf_type2_s
{
  uint32_t reserved;
  uint32_t infotype;
  uint32_t infotype_version;
  uint32_t crypt_key;
} x3f_camf_type2_t;

typedef struct x3f_camf_type4_s
{
  uint32_t decoded_data_size;
  uint32_t decode_bias;
  uint32_t block_size;
  uint32_t block_count;
} x3f_camf_type4_t;

typedef struct x3f_camf_type5_s
{
  uint32_t decoded_data_size;
  uint32_t decode_bias;
  uint32_t unknown2;
  uint32_t unknown3;
} x3f_camf_type5_t;

typedef struct x3f_camf_s
{

  /* Header info */
  uint32_t type;
  union {
    x3f_camf_typeN_t tN;
    x3f_camf_type2_t t2;
    x3f_camf_type4_t t4;
    x3f_camf_type5_t t5;
  };

  /* The encrypted raw data */
  void *data;
  uint32_t data_size;

  /* Help data for type 4 Huffman compression */
  x3f_true_huffman_t table;
  x3f_hufftree_t tree;
  uint8_t *decoding_start;
  uint32_t decoding_size;

  /* The decrypted data */
  void *decoded_data;
  uint32_t decoded_data_size;

  /* Pointers into the decrypted data */
  camf_entry_table_t entry_table;
} x3f_camf_t;

typedef struct x3f_directory_entry_header_s
{
  uint32_t identifier; /* Should be ´SECp´, "SECi", ... */
  uint32_t version;    /* 0x00020001 is version 2.1  */
  union {
    x3f_property_list_t property_list;
    x3f_image_data_t image_data;
    x3f_camf_t camf;
  } data_subsection;
} x3f_directory_entry_header_t;

typedef struct x3f_directory_entry_s
{
  struct
  {
    uint32_t offset;
    uint32_t size;
  } input, output;

  uint32_t type;

  x3f_directory_entry_header_t header;
} x3f_directory_entry_t;

typedef struct x3f_directory_section_s
{
  uint32_t identifier; /* Should be ´SECd´ */
  uint32_t version;    /* 0x00020001 is version 2.1  */

  /* 2.0 Fields */
  uint32_t num_directory_entries;
  x3f_directory_entry_t *directory_entry;
} x3f_directory_section_t;

typedef struct x3f_header_s
{
  /* 2.0 Fields */
  uint32_t identifier; /* Should be ´FOVb´ */
  uint32_t version;    /* 0x00020001 means 2.1 */
  uint8_t unique_identifier[SIZE_UNIQUE_IDENTIFIER];
  uint32_t mark_bits;
  uint32_t columns;  /* Columns and rows ... */
  uint32_t rows;     /* ... before rotation */
  uint32_t rotation; /* 0, 90, 180, 270 */

  char white_balance[SIZE_WHITE_BALANCE]; /* Introduced in 2.1 */
  char color_mode[SIZE_COLOR_MODE];       /* Introduced in 2.3 */

  /* Introduced in 2.1 and extended from 32 to 64 in 3.0 */
  uint8_t extended_types[NUM_EXT_DATA]; /* x3f_extended_types_t */
  float extended_data[NUM_EXT_DATA];    /* 32 bits, but do type differ? */
} x3f_header_t;

typedef struct x3f_info_s
{
  char *error;
  struct
  {
    LibRaw_abstract_datastream *file; /* Use if more data is needed */
  } input, output;
} x3f_info_t;

typedef struct x3f_s
{
  x3f_info_t info;
  x3f_header_t header;
  x3f_directory_section_t directory_section;
} x3f_t;

typedef enum x3f_return_e
{
  X3F_OK = 0,
  X3F_ARGUMENT_ERROR = 1,
  X3F_INFILE_ERROR = 2,
  X3F_OUTFILE_ERROR = 3,
  X3F_INTERNAL_ERROR = 4
} x3f_return_t;

x3f_return_t x3f_delete(x3f_t *x3f);

/* Hacky external flags                                                 */
/* --------------------------------------------------------------------- */

extern int legacy_offset;
extern bool_t auto_legacy_offset;

/* --------------------------------------------------------------------- */
/* Huffman Decode Macros                                                 */
/* --------------------------------------------------------------------- */

#define HUF_TREE_MAX_LENGTH 27
#define HUF_TREE_MAX_NODES(_leaves) ((HUF_TREE_MAX_LENGTH + 1) * (_leaves))
#define HUF_TREE_GET_LENGTH(_v) (((_v) >> 27) & 0x1f)
#define HUF_TREE_GET_CODE(_v) ((_v)&0x07ffffff)

x3f_t *x3f_new_from_file(LibRaw_abstract_datastream *infile);
x3f_return_t x3f_delete(x3f_t *x3f);
x3f_directory_entry_t *x3f_get_raw(x3f_t *x3f);
x3f_directory_entry_t *x3f_get_thumb_plain(x3f_t *x3f);
x3f_return_t x3f_load_data(x3f_t *x3f, x3f_directory_entry_t *DE);
x3f_directory_entry_t *x3f_get_thumb_huffman(x3f_t *x3f);
x3f_directory_entry_t *x3f_get_thumb_jpeg(x3f_t *x3f);
x3f_directory_entry_t *x3f_get_camf(x3f_t *x3f);
x3f_directory_entry_t *x3f_get_prop(x3f_t *x3f);
/* extern */ int64_t x3f_load_data_size(x3f_t *x3f, x3f_directory_entry_t *DE);

#endif
