/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMCRYPTOFACTORY_H
#define GDCMCRYPTOFACTORY_H

#include "gdcmCryptographicMessageSyntax.h"
#include <map>

namespace gdcm
{

/**
 * \brief Class to do handle the crypto factory
 * \details GDCM needs to access in a platform independant way
 * the user specified crypto engine. It can be:
 * \li CAPI (windows only)
 * \li OPENSSL (portable)
 * \li OPENSSLP7 (portable)
 * By default the factory will try:
 * CAPI if on windows
 * OPENSSL if possible
 * OPENSSLP7 when older OpenSSL is used.
 */
class GDCM_EXPORT CryptoFactory
{
public:
  enum CryptoLib {DEFAULT = 0, OPENSSL = 1, CAPI = 2, OPENSSLP7 = 3};

  virtual CryptographicMessageSyntax* CreateCMSProvider() = 0;
  static CryptoFactory* GetFactoryInstance(CryptoLib id = DEFAULT);

protected:
  CryptoFactory(CryptoLib id)
  {
    AddLib(id, this);
  }

private:
  static std::map<CryptoLib, CryptoFactory*>& getInstanceMap()
  {
    static std::map<CryptoLib, CryptoFactory*> libs;
    return libs;
  }

  static void AddLib(CryptoLib id, CryptoFactory* f)
  {
    if (getInstanceMap().insert(std::pair<CryptoLib, CryptoFactory*>(id, f)).second == false)
      {
      gdcmErrorMacro( "Library already registered under id " << (int)id );
      }
  }

protected:
  CryptoFactory(){}
  ~CryptoFactory(){}
};

} // end namespace gdcm

#endif // GDCMCRYPTOFACTORY_H
