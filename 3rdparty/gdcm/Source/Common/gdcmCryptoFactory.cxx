/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmCryptoFactory.h"

#ifdef WIN32
#include "gdcmCAPICryptoFactory.h"
#endif

#ifdef GDCM_USE_SYSTEM_OPENSSL
#ifdef GDCM_HAVE_CMS_RECIPIENT_PASSWORD
#include "gdcmOpenSSLCryptoFactory.h"
#endif
#include "gdcmOpenSSLP7CryptoFactory.h"
#endif

namespace gdcm
{

CryptoFactory* CryptoFactory::GetFactoryInstance(CryptoLib id)
{
#ifdef WIN32
  static CAPICryptoFactory capi(CryptoFactory::CAPI);
#endif
#ifdef GDCM_USE_SYSTEM_OPENSSL
#ifdef GDCM_HAVE_CMS_RECIPIENT_PASSWORD
  static OpenSSLCryptoFactory ossl(CryptoFactory::OPENSSL);
#endif
  static OpenSSLP7CryptoFactory osslp7(CryptoFactory::OPENSSLP7);
#endif

  // If user specified DEFAULT:
  if( id == DEFAULT )
    {
#ifdef GDCM_USE_SYSTEM_OPENSSL
#ifdef GDCM_HAVE_CMS_RECIPIENT_PASSWORD
    id = CryptoFactory::OPENSSL;
#else
    id = CryptoFactory::OPENSSLP7;
#endif // GDCM_HAVE_CMS_RECIPIENT_PASSWORD
#endif // GDCM_USE_SYSTEM_OPENSSL
// We always prefer native API (by default):
#ifdef WIN32
    id = CryptoFactory::CAPI;
#endif // WIN32
    }

  std::map<CryptoLib, CryptoFactory*>::iterator it = getInstanceMap().find(id);
  if (it == getInstanceMap().end())
    {
    gdcmErrorMacro( "No crypto factory registered with id " << (int)id );
    return NULL;
    }
  assert(it->second);
  return it->second;
}

} // end native gdcm
