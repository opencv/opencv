/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmDummyValueGenerator.h"
#include "gdcmTesting.h"

int TestDummyValueGenerator(int , char *[])
{
  gdcm::DummyValueGenerator dvg; (void)dvg;
  const char patientid1[] = "hello";
  const char patientid2[] = "hello ";
  // Because patientid1 & patientid2 are equivalent in DICOM we need to be able to generate
  // identical replacement value in case of de-identifier operation:

  const char *ptr1 = gdcm::DummyValueGenerator::Generate( patientid1 );
  const char *ptr2 = gdcm::DummyValueGenerator::Generate( patientid2 );
  if( !ptr1 || !ptr2 ) return 1;

  std::string str1 = ptr1;
  std::string str2 = ptr2;

  if( str1 != str2 )
    {
    return 1;
    }

  return 0;
}
