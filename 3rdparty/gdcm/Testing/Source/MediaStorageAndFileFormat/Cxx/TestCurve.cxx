/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmCurve.h"

int TestCurve(int, char *[])
{
  gdcm::Curve c;
  c.SetTypeOfData( "TAC" );
  //c.SetTypeOfData( "PROF" );
  //c.SetTypeOfData( "PRESSURE" );
  //c.SetTypeOfData( "RESP" );
  //c.SetTypeOfData( "dummy" );
  std::cout << c.GetTypeOfData() << std::endl;
  std::cout << c.GetTypeOfDataDescription() << std::endl;

  return 0;
}
