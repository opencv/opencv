/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmTagKeywords.h"

int TestTagKeywords(int, char *[])
{
  gdcm::Keywords::FileMetaInformationVersion at1;
  (void)at1;
  if( at1.GetTag() != gdcm::Tag( 0x2, 0x1 ) ) return 1;
  return 0;
}
