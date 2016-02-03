/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "gdcmOpenSSLCryptographicMessageSyntax.h"

#include <limits> // numeric_limits
#include <string.h> // memcpy

#include <openssl/cms.h>
#include <openssl/evp.h>
#include <openssl/bio.h>
#include <openssl/pem.h>
#include <openssl/err.h>
#include <openssl/rand.h>

namespace gdcm
{

OpenSSLCryptographicMessageSyntax::OpenSSLCryptographicMessageSyntax() :
  recips(sk_X509_new_null()),
  pkey(NULL),
  password(NULL)
{
  cipherType = AES128_CIPHER;
  internalCipherType = CreateCipher(cipherType);
}

OpenSSLCryptographicMessageSyntax::~OpenSSLCryptographicMessageSyntax()
{
  EVP_PKEY_free(pkey);
  sk_X509_free(recips);
  if (password) delete[] password;
}

void OpenSSLCryptographicMessageSyntax::SetCipherType( CryptographicMessageSyntax::CipherTypes type )
{
  internalCipherType = CreateCipher(type);
  cipherType = type;
}

CryptographicMessageSyntax::CipherTypes OpenSSLCryptographicMessageSyntax::GetCipherType() const
{
  return cipherType;
}

bool OpenSSLCryptographicMessageSyntax::SetPassword(const char * pass, size_t passLen)
  {
    assert(pass);

    if (password)
      {
      delete[] password;
      }

    passwordLength = passLen;
    password = new char[passLen];
    memcpy(password, pass, passLen);
    
    return true;
  }

bool OpenSSLCryptographicMessageSyntax::Encrypt(char *output, size_t &outlen, const char *array, size_t len) const
{
  BIO *in = NULL, *out = NULL;
  CMS_ContentInfo *cms = NULL;
  int flags = CMS_BINARY | CMS_PARTIAL;
  bool ret = false;

  if (!password && ::sk_X509_num(recips) == 0)
  {
    gdcmErrorMacro( "No password or recipients added." );
    goto err;
  }

  // RAND_status() and RAND_event() return 1 if the PRNG has been seeded with
  // enough data, 0 otherwise.
  if( !RAND_status() )
    {
    gdcmErrorMacro( "PRNG was not seeded properly" );
    goto err;
    }

  if( len > (size_t)std::numeric_limits<int>::max() )
    {
    gdcmErrorMacro( "len is too big: " << len );
    goto err;
    }

  in = BIO_new_mem_buf((void*)array, (int)len);
  if(!in)
    {
    gdcmErrorMacro( "Error at creating the input memory buffer." );
    goto err;
    }

  out = BIO_new(BIO_s_mem());
  if (!out)
    {
    gdcmErrorMacro( "Error at creating the output memory buffer." );
    goto err;
    }


  cms = CMS_encrypt(recips, in, internalCipherType, flags);
  if (!cms)
    {
    gdcmErrorMacro( "Error at creating the CMS strucutre." );
    goto err;
    }

  if (password)
    {
    unsigned char* pwri_tmp = (unsigned char *)BUF_memdup(password, passwordLength);
    
    if (!pwri_tmp)
      goto err;

    if (!CMS_add0_recipient_password(cms, -1, NID_undef, NID_undef, pwri_tmp, passwordLength, NULL))
      goto err;
    pwri_tmp = NULL;
    }

  if (!CMS_final(cms, in, NULL, flags))
    goto err;

  if (! i2d_CMS_bio(out, cms))
    {
    gdcmErrorMacro( "Error at writing CMS structure to output." );
    goto err;
    }

  BUF_MEM *bptr;
  BIO_get_mem_ptr(out, &bptr);

  if (bptr->length > outlen)
    {
    gdcmErrorMacro( "Supplied output buffer too small: " << bptr->length << " bytes needed." );
    goto err;
    }
  memcpy(output, bptr->data, bptr->length);
  outlen = bptr->length;
  
  ret = true;

err:
  if (!ret)
    {
    outlen = 0;
    gdcmErrorMacro( ERR_error_string(ERR_peek_error(), NULL) );
    }

  if (cms)
    CMS_ContentInfo_free(cms);
  if (in)
    BIO_free(in);
  if (out)
    BIO_free(out);

  return ret;
}

bool OpenSSLCryptographicMessageSyntax::Decrypt(char *output, size_t &outlen, const char *array, size_t len) const
{
  BIO *in = NULL, *out = NULL;
  CMS_ContentInfo *cms = NULL;
  bool ret = false;
  int flags = /*CMS_DETACHED | */CMS_BINARY;

  if (!password && pkey == NULL)
    {
    gdcmErrorMacro( "No password or private key specified." );
    goto err;
    }

  in = BIO_new_mem_buf((void*)array, (int)len);
  if (!in)
    {
    gdcmErrorMacro( "Error at creating the input memory buffer." );
    goto err;
    }

  cms = d2i_CMS_bio(in, NULL);
  if (!cms)
    {
    gdcmErrorMacro( "Error when parsing the CMS structure." );
    goto err;
    }

  out = BIO_new(BIO_s_mem());
  if (!out)
    {
    gdcmErrorMacro( "Error at creating the output memory buffer." );
    goto err;
    }

  if (password)
    if (!CMS_decrypt_set1_password(cms, (unsigned char*)password, passwordLength))
      {
      gdcmErrorMacro( "Error at setting the decryption password." );
      goto err;
      }

  if (!CMS_decrypt(cms, pkey, NULL, NULL, out, flags))
    {
    gdcmErrorMacro( "Error at decrypting CMS structure" );
    goto err;
    }

  BUF_MEM *bptr;
  BIO_get_mem_ptr(out, &bptr);

  if (bptr->length > outlen)
    {
    gdcmErrorMacro( "Supplied output buffer too small: " << bptr->length << " bytes needed." );
    goto err;
    }
  memcpy(output, bptr->data, bptr->length);
  outlen = bptr->length;
  
  ret = true;

err:
  if (!ret)
    {
    outlen = 0;
    gdcmErrorMacro( ERR_error_string(ERR_peek_error(), NULL) );
    }

  if (cms)
    CMS_ContentInfo_free(cms);
  if (in)
    BIO_free(in);
  if (out)
    BIO_free(out);

  return ret;
}

bool OpenSSLCryptographicMessageSyntax::ParseKeyFile( const char *keyfile)
{
  ::BIO *in;
  ::EVP_PKEY *new_pkey;
  if ((in=::BIO_new_file(keyfile,"r")) == NULL)
    {
    return false;
    }
  (void)BIO_reset(in);
  if ((new_pkey=PEM_read_bio_PrivateKey(in,NULL,NULL,NULL)) == NULL)
    {
    return false;
    }
  BIO_free(in);

  if (pkey != NULL)
    {
    EVP_PKEY_free(pkey);
    }
  
  this->pkey = new_pkey;
  return true;
}

bool OpenSSLCryptographicMessageSyntax::ParseCertificateFile( const char *keyfile)
{
  assert( recips );
  ::X509 *x509 = NULL;

  ::BIO *in;
  if (!(in=::BIO_new_file(keyfile,"r")))
    {
    return false;
    }
  // -> LEAK reported by valgrind...
  if (!(x509=::PEM_read_bio_X509(in,NULL,NULL,NULL)))
    {
    return false;
    }
  ::BIO_free(in); in = NULL;
  ::sk_X509_push(recips, x509);
  return true;
}

const EVP_CIPHER* OpenSSLCryptographicMessageSyntax::CreateCipher( CryptographicMessageSyntax::CipherTypes ciphertype)
{
  const EVP_CIPHER *cipher = 0;
  switch( ciphertype )
    {
  case CryptographicMessageSyntax::DES3_CIPHER:   // Triple DES
    cipher = EVP_des_ede3_cbc();
    break;
  case CryptographicMessageSyntax::AES128_CIPHER: // CBC AES
    cipher = EVP_aes_128_cbc();
    break;
  case CryptographicMessageSyntax::AES192_CIPHER: // '   '
    cipher = EVP_aes_192_cbc();
    break;
  case CryptographicMessageSyntax::AES256_CIPHER: // '   '
    cipher = EVP_aes_256_cbc();
    break;
    }
  return cipher;
}

} // end namespace gdcm

