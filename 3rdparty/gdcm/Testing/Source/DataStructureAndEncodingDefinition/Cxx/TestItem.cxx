/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmItem.h"
//#include "gdcmStringStream.h"
#include "gdcmVR.h"
#include "gdcmExplicitDataElement.h"
#include "gdcmSwapper.h"

void CreateDataElement(gdcm::ExplicitDataElement &de, int offset)
{
  std::stringstream ss;

  gdcm::Tag tag(0x1234, (uint16_t)(0x5678+offset));
  gdcm::VR vr = gdcm::VR::UN;
  const char str[] = "GDCM";
  uint32_t len = strlen(str);
  assert( sizeof(uint32_t) == 4 );
  gdcm::ByteValue val(str, len);
  tag.Write<gdcm::SwapperNoOp>(ss);
  vr.Write(ss);
  const char *slen = reinterpret_cast<char*>(&len);
  ss.write( slen, 4);
  val.Write<gdcm::SwapperNoOp>(ss);
#ifdef GDCM_WORDS_BIGENDIAN
  de.Read<gdcm::SwapperDoOp>( ss );
#else
  de.Read<gdcm::SwapperNoOp>( ss );
#endif

  std::cout << de << std::endl;
}

int TestItem(int , char *[])
{
  gdcm::Item item;
  std::cout << item << std::endl;

  gdcm::Item it1;
  gdcm::Item it2;

  gdcm::DataSet ds;
  gdcm::ExplicitDataElement xde;
  CreateDataElement(xde,0);
  ds.Insert( xde );
  CreateDataElement(xde,1);
  ds.Insert( xde );
  CreateDataElement(xde,2);
  ds.Insert( xde );
  CreateDataElement(xde,3);
  ds.Insert( xde );

  std::cout << ds << std::endl;

  // undefined by default:
  gdcm::Item it3;
  gdcm::VL vl = it3.GetVL();
  if( !vl.IsUndefined() )
    {
    return 1;
    }
  if( !it3.IsUndefinedLength() )
    {
    return 1;
    }

  return 0;
}
