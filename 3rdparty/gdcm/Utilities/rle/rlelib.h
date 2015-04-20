/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef RLELIB_H
#define RLELIB_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define RLE_LIB_VERSION 0.0.1

typedef struct
{
  uint32_t num_segments;
  uint32_t offset[15];
  void (*print_header) (void);
} rle_header;

typedef struct
{
  rle_header * header;
} rle_compressed_frame;

typedef struct
{
  rle_header * header;
} rle_decompressed_frame;

typedef struct
{
  int (*fill_input_buffer) (rle_decompressed_frame*);
} source_mgr;

typedef struct
{
  int output_scanline;
  int output_height;
  /*int bits_allocated; // 8 or 16, when 16 need to do padded composite*/
  int row;
  int col;
  FILE *stream;
  int current_segment;
  unsigned long current_pos;
  rle_header *header;
} rle_decompress_struct;


void rle_stdio_src(rle_decompress_struct *cinfo, FILE *infile, int *dims);
int rle_start_decompress(rle_decompress_struct *cinfo);
void rle_create_decompress(rle_decompress_struct *cinfo);
int rle_read_scanlines(rle_decompress_struct *cinfo, char *buffer, int f);
int rle_finish_decompress(rle_decompress_struct *cinfo);
void rle_destroy_decompress(rle_decompress_struct *cinfo);

#endif /* RLELIB_H */
