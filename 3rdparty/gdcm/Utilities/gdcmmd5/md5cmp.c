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
#include <stdlib.h>
#include <string.h>
#include <assert.h>

void process_file(const char *filename, md5_byte_t *digest)
{
  int di;
  size_t file_size, read;
  void *buffer;
  md5_state_t state;
  FILE *file = fopen(filename, "rb");

  /* go to the end */
  /*int*/ fseek(file, 0, SEEK_END);
  file_size = ftell(file);
  /*int*/ fseek(file, 0, SEEK_SET);
  buffer = malloc(file_size);
  read = fread(buffer, 1, file_size, file);
  assert( read == file_size );

  md5_init(&state);
  md5_append(&state, (const md5_byte_t *)buffer, file_size);
  md5_finish(&state, digest);
  /*printf("MD5 (\"%s\") = ", test[i]); */
  for (di = 0; di < 16; ++di)
  {
    printf("%02x", digest[di]);
  }
  printf("\t%s\n", filename);
  free(buffer);
  fclose(file);
}

int main(int argc, char *argv[])
{
  md5_byte_t digest1[16];
  md5_byte_t digest2[16];
  if( argc < 3 )
  {
    return 1;
  }

  /* Do file1 */
  process_file(argv[1], digest1);

  /* Do file2 */
  process_file(argv[2], digest2);

  return memcmp(digest1, digest2, 16);
}

