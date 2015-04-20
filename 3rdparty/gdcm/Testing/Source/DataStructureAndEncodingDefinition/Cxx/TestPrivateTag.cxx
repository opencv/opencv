/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmPrivateTag.h"
#include "gdcmDataSet.h"

int TestPrivateTag(int , char * [])
{
  gdcm::PrivateTag pt(0x29,0x1018,"SIEMENS CSA HEADER");
  if( pt != gdcm::Tag(0x0029,0x18) )
    {
    std::cerr << pt << std::endl;
    return 1;
    }
  if( pt.GetOwner() != std::string("SIEMENS CSA HEADER") )
    {
    std::cerr << "[" << pt.GetOwner() << "]" << std::endl;
    return 1;
    }

  const char str[] = "0029,1019,SIEMENS CSA HEADER";

  if( !pt.ReadFromCommaSeparatedString( str ) ) return 1;

  if( pt != gdcm::Tag(0x0029,0x19) )
    {
    std::cerr << pt << std::endl;
    return 1;
    }
  if( pt.GetOwner() != std::string("SIEMENS CSA HEADER") )
    {
    std::cerr << "[" << pt.GetOwner() << "]" << std::endl;
    return 1;
    }

  const gdcm::PrivateTag pt1(0x1,0x2,"BLA");
  const char str0[] = "";
  const char str1[] = "1,2,BLA";
  const char str2[] = "1,65536,BLU";
  const char str3[] = "65536,2,BLU";
  const char str4[] = "65536,2";
  if( pt.ReadFromCommaSeparatedString( NULL ) )
    {
    return 1;
    }
  if( pt.ReadFromCommaSeparatedString( str0 ) )
    {
    return 1;
    }
  if( !pt.ReadFromCommaSeparatedString( str1 ) )
    {
    return 1;
    }
  if( pt != pt1 )
    {
    return 1;
    }
  if( pt.ReadFromCommaSeparatedString( str2 ) )
    {
    return 1;
    }
  if( pt.ReadFromCommaSeparatedString( str3 ) )
    {
    return 1;
    }
  if( pt.ReadFromCommaSeparatedString( str4 ) )
    {
    return 1;
    }

  gdcm::PrivateTag null(0x0,0x0,0);
  if( null.GetOwner() == 0 )
    {
    return 1;
    }

  const gdcm::PrivateTag nospaces(0x12,0x34,"Philips MR Imaging DD 001");
  gdcm::PrivateTag spaces(0x12,0x34," Philips MR Imaging DD 001 ");
  if( nospaces != spaces ) return 1;
  spaces.SetOwner("    Philips MR Imaging DD 001    ");
  if( nospaces != spaces ) return 1;

    {
    // hand-craft Dataset fragment
    gdcm::DataSet ds;
    gdcm::DataElement de(gdcm::Tag(0x0029,0x0011), 0,gdcm::VR::LO);
    ds.Insert(de);
    // get private tag
    gdcm::PrivateTag pt0(0x0029, 0x0023, "Titi");
    ds.GetDataElement(pt0);
    }

  return 0;
}
