/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmStringFilter.h"
#include "gdcmAttribute.h"
#include "gdcmSmartPointer.h"

// reproduce a bug with multi value binary attribute
int TestStringFilter3(int argc, char *argv[])
{
  gdcm::SmartPointer<gdcm::File> f = new gdcm::File;
  gdcm::DataSet & ds = f->GetDataSet();

  // (0020,0032) DS [-85.000000\ 21.600000\108.699997] # 32,3 Image Position (Patient)
    {
    gdcm::Attribute<0x20,0x32> at = { -85, 21.6, 108.699997 };
    gdcm::DataElement de = at.GetAsDataElement();
    const gdcm::ByteValue * bv = de.GetByteValue();
    const std::string ref( bv->GetPointer(), bv->GetLength() );
    ds.Insert( de );

    gdcm::StringFilter sf;
    sf.SetFile( *f );
    const gdcm::Tag & t = at.GetTag();
    std::string s1 = sf.ToString( t );
    std::string s2 = sf.FromString(t, &s1[0], s1.size() );

    std::cout << s1 << std::endl;
    //std::cout << s2 << std::endl;
    if( s2 != ref ) return 1;
    }

  // (0018,1310) US 0\256\256\0                                        # 8,4 Acquisition Matrix
    {
    gdcm::Attribute<0x18,0x1310> at = { 0, 256, 256, 0 };
    gdcm::DataElement de = at.GetAsDataElement();
    const gdcm::ByteValue * bv = de.GetByteValue();
    const std::string ref( bv->GetPointer(), bv->GetLength() );
    ds.Insert( de );

    gdcm::StringFilter sf;
    sf.SetFile( *f );
    const gdcm::Tag & t = at.GetTag();
    std::string s1 = sf.ToString( t );
    std::string s2 = sf.FromString(t, &s1[0], s1.size() );

    std::cout << "[" << s1 << "]" << std::endl;
    //std::cout << s2 << std::endl;
    if( s2 != ref ) return 1;
    }

  return 0;
}
