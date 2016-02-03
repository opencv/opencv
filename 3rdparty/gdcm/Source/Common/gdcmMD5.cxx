/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmMD5.h"
#include "gdcmSystem.h"

#ifdef GDCM_USE_SYSTEM_OPENSSL
#include <openssl/md5.h>
#elif defined(GDCM_BUILD_TESTING)
#include "gdcm_md5.h"
#endif

#include <string.h>//memcmp
#include <stdlib.h> // malloc
#include <stdio.h> // fopen

/*
 */
namespace gdcm
{

class MD5Internals
{
public:
};

MD5::MD5()
{
  Internals = new MD5Internals;
}

MD5::~MD5()
{
  delete Internals;
}

bool MD5::Compute(const char *buffer, unsigned long buf_len, char digest_str[33])
{
  if( !buffer || !buf_len )
    {
    return false;
    }
#ifdef GDCM_USE_SYSTEM_OPENSSL
  unsigned char digest[16];
  MD5_CTX ctx;
  MD5_Init(&ctx);
  MD5_Update(&ctx, buffer, buf_len);
  MD5_Final(digest, &ctx);
#elif defined(GDCM_BUILD_TESTING)
  md5_byte_t digest[16];
  md5_state_t state;
  md5_init(&state);
  md5_append(&state, (const md5_byte_t *)buffer, (int)buf_len);
  md5_finish(&state, digest);
#else
  unsigned char digest[16] = {};
  return false;
#endif
  for (int di = 0; di < 16; ++di)
    {
    sprintf(digest_str+2*di, "%02x", digest[di]);
    }
  digest_str[2*16] = '\0';
  return true;
}

#ifdef GDCM_USE_SYSTEM_OPENSSL
static bool process_file(const char *filename, unsigned char *digest)
{
  if( !filename || !digest ) return false;

  FILE *file = fopen(filename, "rb");
  if(!file)
    {
    return false;
    }

  size_t file_size = System::FileSize(filename);
  void *buffer = malloc(file_size);
  if(!buffer)
    {
    fclose(file);
    return false;
    }
  size_t read = fread(buffer, 1, file_size, file);
  if( read != file_size ) return false;

  MD5_CTX ctx;
  MD5_Init(&ctx);
  MD5_Update(&ctx, buffer, file_size);
  MD5_Final(digest, &ctx);

  /*printf("MD5 (\"%s\") = ", test[i]); */
  /*for (int di = 0; di < 16; ++di)
  {
    printf("%02x", digest[di]);
  }*/
  //printf("\t%s\n", filename);
  free(buffer);
  fclose(file);
  return true;
}
#elif defined(GDCM_BUILD_TESTING)
inline bool process_file(const char *filename, md5_byte_t *digest)
{
  if( !filename || !digest ) return false;

  FILE *file = fopen(filename, "rb");
  if(!file)
    {
    return false;
    }

  size_t file_size = System::FileSize(filename);
  void *buffer = malloc(file_size);
  if(!buffer)
    {
    fclose(file);
    return false;
    }
  size_t read = fread(buffer, 1, file_size, file);
  if( read != file_size ) return false;

  md5_state_t state;
  md5_init(&state);
  md5_append(&state, (const md5_byte_t *)buffer, (int)file_size);
  md5_finish(&state, digest);
  /*printf("MD5 (\"%s\") = ", test[i]); */
  /*for (int di = 0; di < 16; ++di)
  {
    printf("%02x", digest[di]);
  }*/
  //printf("\t%s\n", filename);
  free(buffer);
  fclose(file);
  return true;
}
#else
inline bool process_file(const char *, unsigned char *)
{
  return false;
}
#endif

bool MD5::ComputeFile(const char *filename, char digest_str[33])
{
  // If not file exist
  // return false;
#ifdef GDCM_USE_SYSTEM_OPENSSL
  unsigned char digest[16];
#elif defined(GDCM_BUILD_TESTING)
  md5_byte_t digest[16];
#else
  unsigned char digest[16] = {};
#endif
  /* Do the file */
  if( !process_file(filename, digest) )
    {
    return false;
    }

  for (int di = 0; di < 16; ++di)
    {
    sprintf(digest_str+2*di, "%02x", digest[di]);
    }
  digest_str[2*16] = '\0';
  return true;
}


} // end namespace gdcm
