/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmASN1.h"

struct MyASN1 : public gdcm::ASN1
{
  int TestPBKDF2()
    {
    return this->gdcm::ASN1::TestPBKDF2();
    }
};

int TestASN1(int argc, char *argv[])
{
  if( argc < 1 )
    {
    return 1;
    }
  const char *filename = argv[1];
  MyASN1 asn1;
  asn1.ParseDumpFile( filename );

  asn1.TestPBKDF2();

  return 0;
}
