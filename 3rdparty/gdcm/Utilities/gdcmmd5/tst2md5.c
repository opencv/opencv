/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "md5.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

/*
 * Compute the md5 sum of a tst file
 * tst file format:
 * gdcm (4 bytes)
 * sizeX (4 bytes)
 * sizeY (4 bytes)
 * sizeZ (4 bytes)
 * bytePerScalar (2 bytes)
 * numComponents (2 bytes)
 * image (size = sizeX*sizeY*sizeZ*bytePerScalar/8*numComponents
 */
#define MAGIC_LEN 4
int main(int argc, char *argv[])
{
  const char *filename;
  const char magic[] = "gdcm";
  char buffer[MAGIC_LEN+1];
  int di;
  md5_state_t state;
  md5_byte_t digest[16];
  unsigned int size_x, size_y, size_z;
  unsigned short byte_per_scalar, num_comp;
  FILE *file;
  size_t s, len;
  void *image;

  if( argc < 2 )
    {
    return 1;
    }
  filename = argv[1];
  file = fopen(filename, "rb");
  s = fread(buffer, 1, MAGIC_LEN, file);
  /* end with 0 */
  buffer[MAGIC_LEN] = '\0';
  assert( s == MAGIC_LEN );
  assert( strcmp(magic, buffer) == 0 );

  /* Size X */
  s = fread (&size_x, 1, 4, file);
  assert( s == 4 );
  /* Size Y */
  s = fread (&size_y, 1, 4, file);
  assert( s == 4 );
  /* Size Z */
  s = fread (&size_z, 1, 4, file);
  assert( s == 4 );
  /* Byte Per Scalar */
  s = fread (&byte_per_scalar, 1, 2, file);
  assert( s == 2 );
  assert( !(byte_per_scalar%8) );
  /* Number of Components */
  s = fread (&num_comp, 1, 2, file);
  assert( s == 2 );
  /* Display header */
  printf( "/* %s %d %d %d %d %d */\n", buffer, size_x, size_y, size_z,
    byte_per_scalar, num_comp );

  /* Compute len of image */
  len = size_x*size_y*size_z* (byte_per_scalar/8)*num_comp;
  /* allocate */
  image = malloc(len);
  /* read image */
  s = fread(image, 1, len, file);
  assert( s == len );

  /* compute md5 */
  md5_init(&state);
  md5_append(&state, (const md5_byte_t *)image, len);
  md5_finish(&state, digest);
  /*printf("MD5 (\"%s\") = ", test[i]); */
  printf( "{ \"" );
  for (di = 0; di < 16; ++di)
  {
    printf("%02x", digest[di]);
  }
  printf("\" , \"%s\" },\n", filename);

  free(image);
  fclose(file);

  return 0;
}

