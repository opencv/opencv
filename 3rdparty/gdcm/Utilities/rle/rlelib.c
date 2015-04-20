/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "rlelib.h"

void rle_stdio_src(rle_decompress_struct *cinfo, FILE *infile, int *dims)
{
  int i;
  /*is.read((char*)(&Header), sizeof(uint32_t)*16);*/
  size_t len = fread(cinfo->header, sizeof(uint32_t), 16, infile);
  cinfo->row = dims[0];
  cinfo->col = dims[1];
  if( cinfo->header->num_segments > 16 || cinfo->header->num_segments < 1 )
    {
    /* Need to throw something here*/
      assert(0);
    }
  if( cinfo->header->offset[0] != 64 )
    {
    /* Need to throw something here*/
      assert(0);
    }
  for(i=1; i < cinfo->header->num_segments; ++i)
    {
    if( cinfo->header->offset[i-1] > cinfo->header->offset[i] )
      {
      /* Need to throw something here*/
      assert(0);
      }
    }
  for(i=cinfo->header->num_segments; i < 16; ++i)
    {
    if( cinfo->header->offset[i] != 0 )
      {
      /* Need to throw something here*/
      /*assert(0);*/
      fprintf(stderr, "Impossible : %d for offset # %d\n", cinfo->header->offset[i], i );
      }
    }
  cinfo->stream = infile;

  cinfo->output_height = dims[1];
}

/*
 * G.3.2 The RLE decoder
 * Pseudo code for the RLE decoder is shown below:
 * Loop until the number of output bytes equals the uncompressed segment size
 * Read the next source byte into n
 * If n> =0 and n <= 127 then
 * output the next n+1 bytes literally
 * Elseif n <= - 1 and n >= -127 then
 * output the next byte -n+1 times
 * Elseif n = - 128 then
 * output nothing
 * Endif
 * Endloop
 */

int rle_start_decompress(rle_decompress_struct *cinfo)
{
  fseek(cinfo->stream, cinfo->header->offset[0], SEEK_SET );
  cinfo->current_pos = 0;
  return 1;
}

void rle_create_decompress(rle_decompress_struct *cinfo)
{
  int i;
  cinfo->output_scanline = 0;
  cinfo->output_height = 0;
  cinfo->current_segment = 0;
  cinfo->header = (rle_header*)malloc(sizeof(rle_header));
  cinfo->header->num_segments = 0;
  for(i = 0; i < 16; ++i)
    cinfo->header->offset[i] = 0;
}

int rle_read_scanlines(rle_decompress_struct *cinfo, char *buffer, int f)
{
  signed char byte;
  signed char nextbyte;
  unsigned long length = cinfo->row * cinfo->col;
  /* read too much ! */
  printf("%d vs %d \n", cinfo->current_pos, length ) ;
  printf("scan %d \n", cinfo->output_scanline ) ;
  if( cinfo->current_segment > cinfo->header->num_segments )
    {
    return 0;
    }

  unsigned long noutbytes = 0;
  char * p = buffer;
  int c;
  while( noutbytes < cinfo->row )
    {
    size_t s1 = fread(&byte, 1, 1, cinfo->stream);
    if( byte >= 0 )
      {
      fread(p, (int)byte+1, 1, cinfo->stream);
      p+=(int)byte+1;
      noutbytes += (int)byte+1;
      }
    else if( byte < 0 && byte > -128 )
      {
      size_t s2 = fread(&nextbyte, 1, 1, cinfo->stream);
      for( c = 0; c < (int)-byte + 1; ++c )
        {
        *p++ = nextbyte;
        noutbytes++;
        }
      }
    else
      {
      assert( byte == -128 );
      }
    }
  assert( p - buffer == cinfo->row );
  assert( noutbytes == cinfo->row );
  cinfo->current_pos += cinfo->row;
  long pos = ftell(cinfo->stream);
  /*printf("pos: %d\n",pos);*/
  cinfo->output_scanline++;

  assert( cinfo->current_pos <= length );
  if( cinfo->current_pos >= length )
    {
   /* if( cinfo->current_segment > cinfo->header->num_segments )
      {
      return 0;
      }
    else*/
      {
      cinfo->current_segment++;
      cinfo->output_scanline = 0;
      }
    }

  return 1;
}

int rle_finish_decompress(rle_decompress_struct *cinfo)
{
  return 1;
}

void rle_destroy_decompress(rle_decompress_struct *cinfo)
{
  cinfo->output_scanline = 0; /* why not*/
  cinfo->output_height = 0; /* why not*/
  free(cinfo->header);
}
