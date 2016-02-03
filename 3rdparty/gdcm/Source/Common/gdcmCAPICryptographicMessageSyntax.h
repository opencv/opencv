/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMCAPICRYPTOGRAPHICMESSAGESYNTAX_H
#define GDCMCAPICRYPTOGRAPHICMESSAGESYNTAX_H

#include "gdcmCryptographicMessageSyntax.h"
#include <windows.h>
#include <wincrypt.h>
#include <vector>

namespace gdcm
{

class GDCM_EXPORT CAPICryptographicMessageSyntax : public CryptographicMessageSyntax
{
public:
  CAPICryptographicMessageSyntax();
  ~CAPICryptographicMessageSyntax();

  // X.509
  bool ParseCertificateFile( const char *filename );
  bool ParseKeyFile( const char *filename );

  // PBE
  bool SetPassword(const char * pass, size_t passLen);

  void SetCipherType(CipherTypes type);

  CipherTypes GetCipherType() const;

  /// create a CMS envelopedData structure
  bool Encrypt(char *output, size_t &outlen, const char *array, size_t len) const;
  /// decrypt content from a CMS envelopedData structure
  bool Decrypt(char *output, size_t &outlen, const char *array, size_t len) const;

  bool GetInitialized() const
  {
    return initialized;
  }

private:
  bool Initialize();
  static ALG_ID GetAlgIdByObjId(const char * pszObjId);
  const char *GetCipherObjId() const;
  static void ReverseBytes(BYTE* data, DWORD len);
  static bool LoadFile(const char * filename, BYTE* & buffer, DWORD & bufLen);

private:
  bool initialized;
  HCRYPTPROV hProv;
  std::vector<PCCERT_CONTEXT> certifList;
  HCRYPTKEY hRsaPrivK;
  CipherTypes cipherType;
};

} // end namespace gdcm

#endif //GDCMCAPICRYPTOGRAPHICMESSAGESYNTAX_H
