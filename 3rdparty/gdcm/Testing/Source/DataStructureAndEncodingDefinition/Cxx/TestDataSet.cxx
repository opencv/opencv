/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmDataSet.h"
#include "gdcmDataElement.h"
#include "gdcmExplicitDataElement.h"
#include "gdcmImplicitDataElement.h"


int TestDataSet(int , char *[])
{
  gdcm::DataSet ds;
  std::cout << sizeof ds << std::endl;
  gdcm::DataElement d;
  ds.Insert(d);
  const gdcm::DataElement& r =
    ds.GetDataElement( gdcm::Tag(0,0) );
  std::cout << r << std::endl;

  const gdcm::Tag t2 = gdcm::Tag(0x1234, 0x5678);
  gdcm::DataElement d2(t2);
  std::cout << d2 << std::endl;
  ds.Insert(d2);
  const gdcm::DataElement& r2 =
    ds.GetDataElement( t2 );
  std::cout << r2 << std::endl;

  const gdcm::Tag t3 = gdcm::Tag(0x1234, 0x5679);
  gdcm::DataElement d3(t3);
  d3.SetVR( gdcm::VR::UL );
  std::cout << d3 << std::endl;
  ds.Insert(d3);
  const gdcm::DataElement& r3 =
    ds.GetDataElement( t3 );
  std::cout << r3 << std::endl;

  std::cout << "Size:" << ds.Size() << std::endl;
  if( ds.Size() != 3 )
    {
    //return 1;
    }

  std::cout << "Print Dataset:" << std::endl;
  std::cout << ds << std::endl;

  const gdcm::DataElement &de1 =  ds[ gdcm::Tag(0x0020,0x0037) ];
  const gdcm::DataElement &de2 =  ds(0x0020,0x0037);
  if( &de1 != &de2 ) return 1;

  std::cout << ds << std::endl;

  gdcm::DataElement de3;
  std::cout << de3 << std::endl;
  const gdcm::DataElement &de4 = ds(0x0,0x0);
  std::cout << de4 << std::endl;
  const gdcm::DataElement &de5 = ds(0x0,0x1);
  std::cout << de5 << std::endl;
  const gdcm::DataElement &de6 = ds(0x1,0x1);
  std::cout << de6 << std::endl;

  return 0;
}
