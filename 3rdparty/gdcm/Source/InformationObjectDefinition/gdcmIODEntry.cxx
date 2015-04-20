/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmIODEntry.h"

#include <stdlib.h> // abort

namespace gdcm
{

Usage::UsageType IODEntry::GetUsageType() const
{
  assert( !usage.empty() );
  if( usage == "M" )
    {
    return Usage::Mandatory;
    }
  else if( usage == "U" )
    {
    return Usage::UserOption;
    }
  else if( usage.find( "U - " ) <  usage.size() )
    {
    return Usage::UserOption;
    }
  else if( usage.find( "C- " ) <  usage.size() )
    {
    return Usage::Conditional;
    }
  else if( usage.find( "C - " ) <  usage.size() )
    {
    return Usage::Conditional;
    }
  //else
  assert(0); // Keep me so that I can debug Part3.xml
  return Usage::Invalid;
}

}
