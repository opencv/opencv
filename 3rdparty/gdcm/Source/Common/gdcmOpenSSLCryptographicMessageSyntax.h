/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMOPENSSLCRYPTOGRAPHICMESSAGESYNTAX_H
#define GDCMOPENSSLCRYPTOGRAPHICMESSAGESYNTAX_H

#include "gdcmCryptographicMessageSyntax.h"
#include <openssl/cms.h>
#include <openssl/evp.h>

namespace gdcm
{

class GDCM_EXPORT OpenSSLCryptographicMessageSyntax : public CryptographicMessageSyntax
{
public:
  OpenSSLCryptographicMessageSyntax();
  ~OpenSSLCryptographicMessageSyntax();
  
  // X.509
  bool ParseCertificateFile( const char *filename );
  bool ParseKeyFile( const char *filename );

  // PBE
  bool SetPassword(const char * pass, size_t passLen);

  /// Set Cipher Type.
  /// Default is: AES256_CIPHER
  void SetCipherType(CipherTypes type);
  CipherTypes GetCipherType() const;
  /// create a CMS envelopedData structure
  bool Encrypt(char *output, size_t &outlen, const char *array, size_t len) const;
  /// decrypt content from a PKCS#7 envelopedData structure
  bool Decrypt(char *output, size_t &outlen, const char *array, size_t len) const;

private:
//#ifdef GDCM_HAVE_CMS_RECIPIENT_PASSWORD
//  ::stack_st_X509 *recips;
//#else
  STACK_OF(X509) *recips;
//#endif
  ::EVP_PKEY *pkey;
  const EVP_CIPHER *internalCipherType;
  char * password;
  size_t passwordLength;
  CipherTypes cipherType;

private:
  OpenSSLCryptographicMessageSyntax(const OpenSSLCryptographicMessageSyntax&);  // Not implemented.
  void operator=(const OpenSSLCryptographicMessageSyntax&);  // Not implemented.
  const EVP_CIPHER *CreateCipher( CryptographicMessageSyntax::CipherTypes ciphertype);

};

} // end namespace gdcm

#endif //GDCMOPENSSLCRYPTOGRAPHICMESSAGESYNTAX_H
