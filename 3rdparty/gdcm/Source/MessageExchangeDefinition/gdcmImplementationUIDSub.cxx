/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmImplementationUIDSub.h"

namespace gdcm
{
namespace network
{
const uint8_t ImplementationUIDSub::ItemType = 0x52;
const uint8_t ImplementationUIDSub::Reserved2 = 0x00;

ImplementationUIDSub::ImplementationUIDSub()
{
  ImplementationClassUID = "FOO";
  ItemLength = (uint16_t)ImplementationClassUID.size();
}

const std::ostream &ImplementationUIDSub::Write(std::ostream &os) const
{
  os.write( (char*)&ItemType, sizeof(ItemType) );
  os.write( (char*)&Reserved2, sizeof(Reserved2) );
  os.write( (char*)&ItemLength, sizeof(ItemLength) );
  os.write( ImplementationClassUID.c_str(), ImplementationClassUID.size() );

  return os;
}

} // end namespace network
} // end namespace gdcm
