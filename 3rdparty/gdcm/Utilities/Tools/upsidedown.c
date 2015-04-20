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
#include <assert.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
  const unsigned int x = 420;
  const unsigned int y = 608;
  const unsigned int bit = 1;
  char *buffer = malloc(x*y*bit);
  int i = 1;
  size_t len;
  const char *filename, *outfilename;
  FILE *in, *out;
  if( argc < 3 )
    return 1;
  filename = argv[1];
  outfilename = argv[2];
  in = fopen(filename, "rb" );
  out = fopen(outfilename, "wb" );
  len = fread(buffer,1,bit*x*y,in);
  assert( len == x*y*bit );

  for(i = y; i > 0; --i)
    {
    fwrite(buffer+bit*x*(i-1),1,bit*x,out);
    }
  fclose(in);
  fclose(out);
  free(buffer);
  return 0;
}
