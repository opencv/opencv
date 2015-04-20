/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMCRYPTOGRAPHICMESSAGESYNTAX_H
#define GDCMCRYPTOGRAPHICMESSAGESYNTAX_H

#include "gdcmTypes.h"

namespace gdcm
{

class GDCM_EXPORT CryptographicMessageSyntax
{
public:
  CryptographicMessageSyntax() {}

  virtual ~CryptographicMessageSyntax() {}

  typedef enum {
    DES3_CIPHER,   // Triple DES
    AES128_CIPHER, // CBC AES
    AES192_CIPHER, // '   '
    AES256_CIPHER  // '   '
  } CipherTypes;

    // X.509
  virtual bool ParseCertificateFile( const char *filename ) = 0;
  virtual bool ParseKeyFile( const char *filename ) = 0;

  // PBE
  virtual bool SetPassword(const char * pass, size_t passLen) = 0;

  /// create a CMS envelopedData structure
  virtual bool Encrypt(char *output, size_t &outlen, const char *array, size_t len) const = 0;
  /// decrypt content from a CMS envelopedData structure
  virtual bool Decrypt(char *output, size_t &outlen, const char *array, size_t len) const = 0;

  virtual void SetCipherType(CipherTypes type) = 0;

  virtual CipherTypes GetCipherType() const = 0;

private:
  CryptographicMessageSyntax(const CryptographicMessageSyntax&);  // Not implemented.
  void operator=(const CryptographicMessageSyntax&);  // Not implemented.
};

} // end namespace gdcm

#endif //GDCMCRYPTOGRAPHICMESSAGESYNTAX_H
