/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include <stdio.h>
#include <stdlib.h> // abort

#include "rlelib.h"

void std_print_header(rle_compressed_frame * frame)
{
  rle_header *header = frame->header;
  unsigned long ns = header->num_segments;
  printf("%lu\n", ns);
}


int write_RLE_file(const char *filename)
{
  assert(0);
  return 0;
}


int fill_input_buffer(rle_decompressed_frame * frame)
{
  return 1;
}

int read_RLE_file(const char *filename)
{
  assert(0);
  return 1;
}

int main(int argc, char *argv[])
{
  rle_decompress_struct cinfo;
  FILE * infile;

  const char *filename;
  if( argc < 2 )
    {
    return 1;
    }
  filename = argv[1];

  if ((infile = fopen(filename, "rb")) == NULL) {
    fprintf(stderr, "can't open %s\n", filename);
    return 0;
  }

  rle_create_decompress(&cinfo);

  int i;
  int dims[2] = { 1760,1760 };
  int bpp = 16;
  rle_stdio_src(&cinfo, infile, dims);

  printf("num segment: %d\n", cinfo.header->num_segments );
  printf("offsets table:\n");
  for(i = 0; i < 16; ++i)
    printf("offset: %d\n", cinfo.header->offset[i] );

  (void) rle_start_decompress(&cinfo);

  char *buffer = (char*)malloc( dims[0] * (bpp / 8) );
  while( cinfo.current_segment < cinfo.header->num_segments ) {
    while (cinfo.output_scanline < cinfo.output_height) {
      (void) rle_read_scanlines(&cinfo, buffer, 1);
      /*put_scanline_someplace(buffer[0], row_stride);*/
    }
  }
  free(buffer);

  (void) rle_finish_decompress(&cinfo);

  rle_destroy_decompress(&cinfo);

  fclose(infile);

  return 0;
}
