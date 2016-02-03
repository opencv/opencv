/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmOpenSSLCryptoFactory.h"

#include <openssl/evp.h>
#include <openssl/err.h>

namespace gdcm
{

void OpenSSLCryptoFactory::InitOpenSSL()
{
  static bool Initialized = false;
  if (!Initialized)
    {
    OpenSSL_add_all_algorithms();
    ERR_load_crypto_strings();
    Initialized = true;
    }
}

} // end namespace gdcm
