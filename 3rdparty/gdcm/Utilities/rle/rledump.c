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

int main(int argc, char *argv[])
{
  rle_decompress_struct cinfo;
  FILE * infile;
  int i;

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

  /* Dimensions: (1760,1760,1)*/
  int dims[2] = { 1760,1760 };
  rle_stdio_src(&cinfo, infile, dims);

  /*rle_header *h = cinfo.header;*/
  printf("num segment: %d\n", cinfo.header->num_segments );
  printf("offsets table:\n");
  for(i = 0; i < 16; ++i)
    printf("offset: %d\n", cinfo.header->offset[i] );

  /* Simply dump the file info:*/

  rle_destroy_decompress(&cinfo);

  fclose(infile);

  return 0;
}
