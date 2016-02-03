/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMCAPICRYPTOFACTORY_H
#define GDCMCAPICRYPTOFACTORY_H

#include "gdcmCryptoFactory.h"

namespace gdcm
{

class GDCM_EXPORT CAPICryptoFactory : public CryptoFactory
{
public:
  CAPICryptoFactory(CryptoLib id);
  CryptographicMessageSyntax* CreateCMSProvider();

private:
  CAPICryptoFactory() {}
};

} // end namespace gdcm

#endif //GDCMCAPICRYPTOFACTORY_H
