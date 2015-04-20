/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmDictEntry.h"
#include "gdcmDict.h"

int TestDictEntry(int, char *[])
{
  //gdcm::DictEntry de(0,0,0,0);
  //std::cout << sizeof(de) << std::endl;

  //gdcm::PrivateDictEntry pde(0,0,0);
  //std::cout << sizeof(pde) << std::endl;
  gdcm::PrivateTag pt1(0,0,"bla");
  gdcm::PrivateTag pt2(0,0,"foo");
  gdcm::PrivateTag pt3(0,1,"bla");
  gdcm::PrivateTag pt4(0,1,"foo");
  if( ! (pt1 < pt2) )
    {
    return 1;
    }
  if( ! (pt2 < pt3) )
    {
    return 1;
    }
  if( ! (pt3 < pt4) )
    {
    return 1;
    }

  if( ! (pt1 < pt3) )
    {
    return 1;
    }
  if( ! (pt1 < pt4) )
    {
    return 1;
    }

  if( ! (pt2 < pt3) )
    {
    return 1;
    }
  if( ! (pt2 < pt4) )
    {
    return 1;
    }

  if( pt4 < pt3 )
    {
    return 1;
    }
  if( pt4 < pt2 )
    {
    return 1;
    }
  if( pt4 < pt1 )
    {
    return 1;
    }

  if( pt3 < pt2 )
    {
    return 1;
    }
  if( pt3 < pt1 )
    {
    return 1;
    }

  if( pt2 < pt1 )
    {
    return 1;
    }

  return 0;
}
