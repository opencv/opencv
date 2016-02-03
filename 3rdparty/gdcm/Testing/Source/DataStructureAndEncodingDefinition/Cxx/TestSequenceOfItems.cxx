/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmSequenceOfItems.h"

int TestSequenceOfItems(int, char *[])
{
  gdcm::SequenceOfItems si;
  std::cout << si << std::endl;

  gdcm::VL vl = si.GetLength();
  if( !vl.IsUndefined() )
    {
    return 1;
    }
  if( !si.IsUndefinedLength() )
    {
    return 1;
    }

  gdcm::SmartPointer<gdcm::SequenceOfItems> sq = new gdcm::SequenceOfItems();
  gdcm::DataElement des( gdcm::Tag(0xdead,0xbeef) );
  des.SetVR(gdcm::VR::SQ);
  des.SetValue(*sq);

  if( !des.GetValueAsSQ()->IsUndefinedLength() )
    {
    return 1;
    }

  return 0;
}
