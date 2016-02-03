/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMOPENSSLCRYPTOFACTORY_H
#define GDCMOPENSSLCRYPTOFACTORY_H

#include "gdcmCryptoFactory.h"
#include "gdcmOpenSSLCryptographicMessageSyntax.h"

namespace gdcm
{

class GDCM_EXPORT OpenSSLCryptoFactory : public CryptoFactory
{
public:
  OpenSSLCryptoFactory(CryptoLib id) : CryptoFactory(id)
  {
    gdcmDebugMacro( "OpenSSL Factory registered." );
  }
    
public:
  CryptographicMessageSyntax* CreateCMSProvider()
  {
    InitOpenSSL();
    return new OpenSSLCryptographicMessageSyntax();
  }

protected:
  void InitOpenSSL();

private:
  OpenSSLCryptoFactory(){}
};

} // end namespace gdcm

#endif //GDCMOPENSSLCRYPTOFACTORY_H
